"""Counter Movement Jump pipeline.

Pose-only test: flight time -> jump height via projectile formula. No
calibration needed (height comes from time, not pixels).

Single-pass design (hard rule #3): each frame is detected, posed,
state-machined, annotated, and written to the output video before the
next frame is read. Events (takeoff / landing) fire late by a few frames
because we require a stable airborne run; this is fine for HUD purposes.
"""
from __future__ import annotations

from dataclasses import dataclass
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
from src.metrics.jump.jump_height_cm import jump_height_cm
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

# Tunables
_BASELINE_FRAMES = 30        # initial standing window for ground reference
_AIRBORNE_THRESHOLD_FRAC = 0.05  # ankle must rise >5% of athlete bbox height
_MIN_AIRBORNE_FRAMES = 3     # debounce: airborne must persist this long
_POSE_CONF_MIN = 0.30        # ankle confidence below this -> frame ignored
_ENDCARD_HOLD_S = 2.0


_State = Literal["standing", "airborne_provisional", "airborne_confirmed"]


@dataclass
class _StreamingDetector:
    """Per-frame state machine that locates takeoff + landing live.

    Tracks min(left_ankle_y, right_ankle_y) — smaller pixel-y = higher in
    image. Builds a baseline from the first `_BASELINE_FRAMES` valid samples,
    then flags airborne when the ankle rises a fraction of bbox-height
    above the baseline. Debounce window of `_MIN_AIRBORNE_FRAMES` rejects
    one-frame spikes.

    Multiple airborne runs are collected (walking steps trigger short ones,
    the actual jump triggers a long one). `best_jump()` returns the longest.
    """

    state: _State = "standing"
    candidates: list[tuple[int, int]] = None  # type: ignore[assignment]
    _baseline_y: float | None = None
    _threshold_y: float | None = None
    _standing_ankles: list[float] = None      # type: ignore[assignment]
    _standing_heights: list[float] = None     # type: ignore[assignment]
    _provisional_start: int | None = None
    _confirmed_takeoff: int | None = None

    def __post_init__(self) -> None:
        self.candidates = []
        self._standing_ankles = []
        self._standing_heights = []

    def update(self, frame_idx: int, ankle_y: float | None, bbox_h: float | None) -> None:
        if ankle_y is None or bbox_h is None:
            return

        if self._threshold_y is None:
            self._standing_ankles.append(ankle_y)
            self._standing_heights.append(bbox_h)
            if len(self._standing_ankles) >= _BASELINE_FRAMES:
                self._baseline_y = float(np.median(self._standing_ankles))
                scale = float(np.median(self._standing_heights))
                self._threshold_y = self._baseline_y - _AIRBORNE_THRESHOLD_FRAC * scale
            return

        airborne = ankle_y < self._threshold_y
        if self.state == "standing":
            if airborne:
                self.state = "airborne_provisional"
                self._provisional_start = frame_idx
        elif self.state == "airborne_provisional":
            if airborne:
                if frame_idx - (self._provisional_start or frame_idx) + 1 >= _MIN_AIRBORNE_FRAMES:
                    self.state = "airborne_confirmed"
                    self._confirmed_takeoff = self._provisional_start
            else:
                self.state = "standing"
                self._provisional_start = None
        elif self.state == "airborne_confirmed":
            if not airborne:
                # Close out this candidate and reset to listen for more.
                if self._confirmed_takeoff is not None:
                    self.candidates.append((self._confirmed_takeoff, frame_idx))
                self.state = "standing"
                self._confirmed_takeoff = None
                self._provisional_start = None

    def best_jump(self) -> tuple[int, int] | None:
        """Return (takeoff, landing) of the longest airborne run, or None."""
        if not self.candidates:
            return None
        return max(self.candidates, key=lambda p: p[1] - p[0])

    @property
    def latest_takeoff(self) -> int | None:
        """Most recent confirmed-takeoff frame (for live HUD)."""
        if self._confirmed_takeoff is not None:
            return self._confirmed_takeoff
        return self.candidates[-1][0] if self.candidates else None

    @property
    def latest_landing(self) -> int | None:
        """Most recent landing frame, only set when not currently airborne."""
        if self.state == "airborne_confirmed":
            return None
        return self.candidates[-1][1] if self.candidates else None


class CounterMovementJumpTest(BaseTest):
    """CMJ pipeline: pose-only, projectile-formula jump height."""

    test_id = "counter-movement-jump"

    def __init__(self) -> None:
        self._pose = create_pose_estimator("pose_biomech")

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

        detector = _StreamingDetector()
        n_frames = 0
        n_low_conf = 0
        try:
            for frame in frame_iter(video_path):
                n_frames += 1
                img = frame.image  # mutated in place by overlays

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

                # Overlays (draw before any event flash)
                if pose is not None:
                    draw_skeleton(img, pose.keypoints)
                if bbox is not None:
                    draw_bbox(img, bbox)
                draw_hud(img, _hud_fields(detector, frame.idx, fps))

                if frame.idx in (detector.latest_takeoff, detector.latest_landing):
                    if bbox is not None:
                        event_flash(img, bbox)

                writer.write(img)

            jump = detector.best_jump()
            if jump is None:
                raise ProtocolError(
                    "could not locate a CMJ jump in the video — "
                    "ankle never rose above the standing-baseline threshold"
                )
            takeoff, landing = jump
            ft = flight_time_s(takeoff, landing, fps)
            jh = jump_height_cm(ft)
            metrics = {
                "flight_time_s": MetricValue(raw=ft, unit="s"),
                "jump_height_cm": MetricValue(raw=jh, unit="cm"),
            }
            scores, test_score = score_test(metrics, self.test_id, athlete.gender)

            endcard = render_endcard(
                title="Counter Movement Jump",
                athlete=f"{athlete.gender} age {athlete.age}",
                metric_rows=[
                    ("Jump Height (cm)", f"{jh:.1f}",
                     int(round(scores["jump_height_cm"].score))),
                    ("Flight Time (s)", f"{ft:.3f}",
                     int(round(scores["flight_time_s"].score))),
                ],
                test_score=int(round(test_score.score)),
                band=format_band(test_score.band),
                size=(info.width, info.height),
            )
            for _ in range(int(fps * _ENDCARD_HOLD_S)):
                writer.write(endcard)
        finally:
            writer.release()

        diagnostics = AnalysisDiagnostics(
            fps_input=fps,
            duration_s=n_frames / fps if fps > 0 else 0.0,
        )
        return AnalysisResult(
            test_id=self.test_id,
            athlete=athlete,
            metrics=metrics,
            scores=scores,
            test_score=test_score,
            annotated_video_path=out_path,
            diagnostics=diagnostics,
        )


# --- helpers ------------------------------------------------------------


def _ankle_features(pose, bbox) -> tuple[float | None, float | None]:
    """Return (min_ankle_y, bbox_height) or (None, None) if unusable."""
    if pose is None or bbox is None:
        return None, None
    la = pose.position("left_ankle")
    ra = pose.position("right_ankle")
    la_c = pose.confidence_of("left_ankle")
    ra_c = pose.confidence_of("right_ankle")
    if max(la_c, ra_c) < _POSE_CONF_MIN:
        return None, None
    # If only one ankle is confident, use it; otherwise take the lower-y
    # (= higher in image) of the two so a single foot leaving the ground
    # already counts as airborne.
    candidates = []
    if la_c >= _POSE_CONF_MIN:
        candidates.append(float(la[1]))
    if ra_c >= _POSE_CONF_MIN:
        candidates.append(float(ra[1]))
    if not candidates:
        return None, None
    return min(candidates), float(bbox[3] - bbox[1])


def _hud_fields(det: _StreamingDetector, frame_idx: int, fps: float) -> dict[str, str]:
    """Live HUD — best-known state at this frame.

    Pre-takeoff: ready. Mid-air: live elapsed (current candidate).
    Post-landing: latest completed candidate's flight + height. The
    "best" jump for the final score is picked across all candidates
    after the loop ends.
    """
    if det.state == "airborne_confirmed" and det.latest_takeoff is not None:
        elapsed = (frame_idx - det.latest_takeoff) / fps
        return {
            "phase": "airborne",
            "flight": f"{elapsed:.2f} s",
            "height": f"{jump_height_cm(elapsed):.1f} cm",
        }
    if det.candidates:
        takeoff, landing = det.candidates[-1]
        ft = flight_time_s(takeoff, landing, fps)
        return {
            "phase": "landed",
            "flight": f"{ft:.3f} s",
            "height": f"{jump_height_cm(ft):.1f} cm",
        }
    return {"phase": "ready", "flight": "-", "height": "-"}
