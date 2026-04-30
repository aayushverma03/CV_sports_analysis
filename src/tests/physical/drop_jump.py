"""Drop Jump pipeline.

Pose-only test. Athlete steps off a box, lands, rebounds vertically.
Metrics: rebound flight time -> jump height; ground contact between
landings -> reactive strength index (RSI = jump_height_m / contact_s).
Only `rsi` is benchmark-scored; the others ride along as informational.

State machine on min(left_ankle_y, right_ankle_y), pixel-y (smaller =
higher in image):

    on_box -> dropping -> contact_1 -> rebound -> contact_2 -> done

The ground baseline is unknown until first landing; we lock it at the
first frame where the ankle stabilises after the drop. This avoids
needing the box height as an input.

Pose backend: `pose_default` (YOLO-pose). Spec mandates `pose_biomech`
(RTMPose-x), but our v1 metrics don't use joint angles — switch to
RTMPose when biomech metrics (trunk lean, etc.) are added.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import cv2
import numpy as np

from src.core.annotation.overlays import (
    draw_bbox,
    draw_hud,
    draw_skeleton,
    event_flash,
    render_endcard,
)
from src.core.detection.player_detector import detect_players
from src.core.pose.estimator import create_pose_estimator
from src.core.utils.video_io import frame_iter, video_info
from src.metrics.jump.flight_time_s import flight_time_s
from src.metrics.jump.ground_contact_time_s import ground_contact_time_s
from src.metrics.jump.jump_height_cm import jump_height_cm
from src.metrics.jump.rsi import rsi
from src.scoring.grade import format_band
from src.tests.base import (
    AnalysisDiagnostics,
    AnalysisResult,
    AthleteProfile,
    BaseTest,
    MetricValue,
    ProtocolError,
    score_test,
)

# --- Tunables ----------------------------------------------------------

_ON_BOX_FRAMES = 15                  # samples to lock the box level
_DROP_THRESHOLD_FRAC = 0.05          # ankle moves >5% bbox-h below box -> dropping
_LANDING_PLATEAU_FRAMES = 3          # K stable frames -> landed
_LANDING_VELOCITY_FRAC = 0.02        # max frame-to-frame delta to count as plateau
_REBOUND_LIFTOFF_FRAC = 0.05         # ankle rises >5% bbox-h above ground -> airborne
_REBOUND_LAND_TOLERANCE_FRAC = 0.03  # ankle returns within 3% of ground -> landed
_POSE_CONF_MIN = 0.30
_ENDCARD_HOLD_S = 2.0


_State = Literal[
    "on_box", "dropping", "contact_1", "rebound", "contact_2", "done"
]


@dataclass
class _DropJumpDetector:
    """Streaming state machine — locates the 4 frames a Drop Jump produces.

    Each `update(frame_idx, ankle_y, bbox_h)` advances the state given the
    current ankle-y reading (in image pixels; smaller = higher). bbox-h
    is the athlete's bbox height; thresholds scale with it so the same
    tunables work across athletes / framings.
    """

    state: _State = "on_box"
    box_y: float | None = None
    ground_y: float | None = None
    step_off_frame: int | None = None
    first_landing_frame: int | None = None
    rebound_takeoff_frame: int | None = None
    rebound_landing_frame: int | None = None

    _box_buffer: list[float] = field(default_factory=list)
    _last_y: float | None = None
    _stable_count: int = 0
    _provisional_landing_frame: int | None = None

    def update(self, frame_idx: int, ankle_y: float | None, bbox_h: float | None) -> None:
        if ankle_y is None or bbox_h is None:
            self._last_y = None
            return

        if self.state == "on_box":
            self._box_buffer.append(ankle_y)
            if len(self._box_buffer) >= _ON_BOX_FRAMES:
                self.box_y = float(np.median(self._box_buffer))
                # Once locked, watch for ankle_y dropping below box level.
            if self.box_y is not None and ankle_y > self.box_y + _DROP_THRESHOLD_FRAC * bbox_h:
                self.state = "dropping"
                self.step_off_frame = frame_idx

        elif self.state == "dropping":
            # Plateau detection: K consecutive frames with small delta.
            if self._last_y is not None:
                delta = abs(ankle_y - self._last_y)
                if delta < _LANDING_VELOCITY_FRAC * bbox_h:
                    if self._stable_count == 0:
                        self._provisional_landing_frame = frame_idx - 1
                    self._stable_count += 1
                else:
                    self._stable_count = 0
                    self._provisional_landing_frame = None
            if self._stable_count >= _LANDING_PLATEAU_FRAMES:
                self.state = "contact_1"
                self.first_landing_frame = self._provisional_landing_frame
                self.ground_y = ankle_y
                self._stable_count = 0
                self._provisional_landing_frame = None

        elif self.state == "contact_1":
            # Watch for liftoff: ankle rises above ground baseline.
            assert self.ground_y is not None
            if ankle_y < self.ground_y - _REBOUND_LIFTOFF_FRAC * bbox_h:
                self.state = "rebound"
                self.rebound_takeoff_frame = frame_idx

        elif self.state == "rebound":
            # Watch for landing: ankle returns near ground baseline.
            assert self.ground_y is not None
            if ankle_y >= self.ground_y - _REBOUND_LAND_TOLERANCE_FRAC * bbox_h:
                self.state = "contact_2"
                self.rebound_landing_frame = frame_idx
                self.state = "done"  # we're done — only need this one rebound

        self._last_y = ankle_y


# --- Pipeline ----------------------------------------------------------


class DropJumpTest(BaseTest):
    """Drop Jump pipeline: rebound RSI from ankle-y kinematics."""

    test_id = "drop-jump"

    def __init__(self) -> None:
        self._pose = create_pose_estimator("pose_default")

    def run(
        self,
        video_path: Path,
        athlete: AthleteProfile,
        output_dir: Path,
    ) -> AnalysisResult:
        info = video_info(video_path)
        fps = info.fps
        out_path = output_dir / f"{self.test_id}.mp4"

        writer = cv2.VideoWriter(
            str(out_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (info.width, info.height),
        )
        if not writer.isOpened():
            raise ProtocolError(f"could not open VideoWriter at {out_path}")

        detector = _DropJumpDetector()
        n_frames = 0
        n_low_conf = 0

        try:
            for frame in frame_iter(video_path):
                n_frames += 1
                img = frame.image

                dets = detect_players(img)
                bbox = dets[0].bbox_xyxy if dets else None
                pose = (
                    self._pose.estimate_bbox(img, bbox)
                    if bbox is not None else None
                )

                ankle_y, bbox_h = _ankle_features(pose, bbox)
                if pose is not None and pose.mean_confidence < _POSE_CONF_MIN:
                    n_low_conf += 1

                detector.update(frame.idx, ankle_y, bbox_h)

                if pose is not None:
                    draw_skeleton(img, pose.keypoints)
                if bbox is not None:
                    draw_bbox(img, bbox)
                draw_hud(img, _hud_fields(detector, frame.idx, fps))

                events = (
                    detector.step_off_frame, detector.first_landing_frame,
                    detector.rebound_takeoff_frame, detector.rebound_landing_frame,
                )
                if frame.idx in events and bbox is not None:
                    event_flash(img, bbox)

                writer.write(img)

            metrics = self._compute_metrics(detector, fps)
            scores, test_score = score_test(metrics, self.test_id, athlete.gender)

            endcard_rows = [
                (mid.replace("_", " "), f"{mv.raw:.3f} {mv.unit}",
                 int(round(scores[mid].score)) if mid in scores else 0)
                for mid, mv in metrics.items()
            ]
            endcard = render_endcard(
                title="Drop Jump",
                athlete=f"{athlete.gender} age {athlete.age}",
                metric_rows=endcard_rows,
                test_score=int(round(test_score.score)),
                band=format_band(test_score.band),
                size=(info.width, info.height),
            )
            for _ in range(int(fps * _ENDCARD_HOLD_S)):
                writer.write(endcard)
        finally:
            writer.release()

        return AnalysisResult(
            test_id=self.test_id,
            athlete=athlete,
            metrics=metrics,
            scores=scores,
            test_score=test_score,
            annotated_video_path=out_path,
            diagnostics=AnalysisDiagnostics(
                fps_input=fps, duration_s=n_frames / fps if fps > 0 else 0.0
            ),
        )

    def _compute_metrics(
        self, detector: _DropJumpDetector, fps: float
    ) -> dict[str, MetricValue]:
        if detector.first_landing_frame is None:
            raise ProtocolError(
                "no first landing detected — athlete may not have stepped "
                "off the box, or pose tracking lost the ankle"
            )
        if detector.rebound_takeoff_frame is None:
            raise ProtocolError(
                "no rebound takeoff detected — athlete did not rebound after "
                "first ground contact"
            )
        if detector.rebound_landing_frame is None:
            raise ProtocolError(
                "no rebound landing detected — athlete did not return to "
                "the ground after the rebound"
            )

        gct_s = ground_contact_time_s(
            detector.first_landing_frame, detector.rebound_takeoff_frame, fps
        )
        flight_s = flight_time_s(
            detector.rebound_takeoff_frame, detector.rebound_landing_frame, fps
        )
        height_cm = jump_height_cm(flight_s)
        rsi_value = rsi(height_cm, gct_s) if gct_s > 0 else 0.0

        return {
            "rsi": MetricValue(raw=rsi_value, unit="m_per_s"),
            "ground_contact_time_s": MetricValue(raw=gct_s, unit="s"),
            "flight_time_s": MetricValue(raw=flight_s, unit="s"),
            "jump_height_cm": MetricValue(raw=height_cm, unit="cm"),
        }


# --- helpers ------------------------------------------------------------


def _ankle_features(pose, bbox) -> tuple[float | None, float | None]:
    """Return (min_ankle_y, bbox_height) or (None, None) if unusable."""
    if pose is None or bbox is None:
        return None, None
    la = pose.position("left_ankle")
    ra = pose.position("right_ankle")
    la_c = pose.confidence_of("left_ankle")
    ra_c = pose.confidence_of("right_ankle")
    candidates = []
    if la_c >= _POSE_CONF_MIN:
        candidates.append(float(la[1]))
    if ra_c >= _POSE_CONF_MIN:
        candidates.append(float(ra[1]))
    if not candidates:
        return None, None
    return min(candidates), float(bbox[3] - bbox[1])


def _hud_fields(det: _DropJumpDetector, frame_idx: int, fps: float) -> dict[str, str]:
    if det.state == "on_box":
        return {"phase": "on box"}
    if det.state == "dropping":
        return {"phase": "dropping"}
    if det.state == "contact_1" and det.first_landing_frame is not None:
        elapsed = (frame_idx - det.first_landing_frame) / fps
        return {"phase": "contact", "contact": f"{elapsed:.2f} s"}
    if det.state == "rebound" and det.rebound_takeoff_frame is not None:
        elapsed = (frame_idx - det.rebound_takeoff_frame) / fps
        return {
            "phase": "airborne",
            "flight": f"{elapsed:.2f} s",
            "height": f"{jump_height_cm(elapsed):.1f} cm",
        }
    if det.state == "done" and det.rebound_landing_frame is not None and det.rebound_takeoff_frame is not None:
        flight = flight_time_s(
            det.rebound_takeoff_frame, det.rebound_landing_frame, fps
        )
        height = jump_height_cm(flight)
        gct = (
            ground_contact_time_s(
                det.first_landing_frame, det.rebound_takeoff_frame, fps
            ) if det.first_landing_frame is not None else 0.0
        )
        rsi_v = rsi(height, gct) if gct > 0 else 0.0
        return {
            "phase": "done",
            "rsi": f"{rsi_v:.2f}",
            "height": f"{height:.1f} cm",
            "contact": f"{gct:.2f} s",
        }
    return {"phase": det.state}
