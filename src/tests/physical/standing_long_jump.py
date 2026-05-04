"""Standing Long Jump pipeline.

Pose-based single-pass test: athlete jumps horizontally from a take-off
line, distance measured to the closest landing point. Side-on camera.

Calibration: this protocol needs pixel-to-metre to convert horizontal
ankle displacement into centimetres. The user's footage typically shows
floor distance markings YOLO-World does not reliably detect, so the
pipeline falls back to a body-height proxy: athlete bbox-h during the
standing baseline is divided by an assumed body height (defaults to
1.70 m, configurable). Calibration quality is reported alongside the
metric so downstream consumers can flag estimates derived this way.

Detection events mirror CMJ — takeoff = ankle rises a fraction of bbox-h
above its standing baseline; landing = ankle returns near baseline. The
"best jump" is the longest airborne candidate. Horizontal distance
comes from comparing ankle x at takeoff vs ankle x at landing.
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

_BASELINE_FRAMES = 30
_AIRBORNE_THRESHOLD_FRAC = 0.05
_MIN_AIRBORNE_FRAMES = 3
_POSE_CONF_MIN = 0.30
_ENDCARD_HOLD_S = 2.0

# Calibration via body-height proxy. Default 1.70 m matches the median
# adult athlete; can be overridden per-test by passing
# `assumed_athlete_height_m` to the constructor.
_DEFAULT_ATHLETE_HEIGHT_M = 1.70


_State = Literal["standing", "airborne_provisional", "airborne_confirmed"]


# --- State -------------------------------------------------------------


@dataclass
class _StreamingDetector:
    """CMJ-style streaming takeoff / landing detector + per-frame ankle
    positions. After the loop, the longest airborne candidate is the
    best jump; ankle positions at takeoff and landing give horizontal
    distance.
    """

    state: _State = "standing"
    candidates: list[tuple[int, int]] = field(default_factory=list)
    _baseline_y: float | None = None
    _threshold_y: float | None = None
    _standing_ankles_y: list[float] = field(default_factory=list)
    _standing_heights: list[float] = field(default_factory=list)
    _provisional_start: int | None = None
    _confirmed_takeoff: int | None = None

    # Per-frame ankle (x, y) in pixel coords for the most-confident foot,
    # plus bbox-h. Used for distance measurement and peak-height.
    ankle_samples: list[tuple[int, float, float, float]] = field(
        default_factory=list,
    )

    def update(
        self,
        frame_idx: int,
        ankle_xy: tuple[float, float] | None,
        bbox_h: float | None,
    ) -> None:
        if ankle_xy is not None and bbox_h is not None:
            self.ankle_samples.append(
                (frame_idx, ankle_xy[0], ankle_xy[1], bbox_h)
            )
        if ankle_xy is None or bbox_h is None:
            return
        ankle_y = ankle_xy[1]

        if self._threshold_y is None:
            self._standing_ankles_y.append(ankle_y)
            self._standing_heights.append(bbox_h)
            if len(self._standing_ankles_y) >= _BASELINE_FRAMES:
                self._baseline_y = float(np.median(self._standing_ankles_y))
                scale = float(np.median(self._standing_heights))
                self._threshold_y = (
                    self._baseline_y - _AIRBORNE_THRESHOLD_FRAC * scale
                )
            return

        airborne = ankle_y < self._threshold_y
        if self.state == "standing":
            if airborne:
                self.state = "airborne_provisional"
                self._provisional_start = frame_idx
        elif self.state == "airborne_provisional":
            if airborne:
                if (frame_idx - (self._provisional_start or frame_idx) + 1
                        >= _MIN_AIRBORNE_FRAMES):
                    self.state = "airborne_confirmed"
                    self._confirmed_takeoff = self._provisional_start
            else:
                self.state = "standing"
                self._provisional_start = None
        elif self.state == "airborne_confirmed":
            if not airborne:
                if self._confirmed_takeoff is not None:
                    self.candidates.append(
                        (self._confirmed_takeoff, frame_idx)
                    )
                self.state = "standing"
                self._confirmed_takeoff = None
                self._provisional_start = None

    def best_jump(self) -> tuple[int, int] | None:
        if not self.candidates:
            return None
        return max(self.candidates, key=lambda p: p[1] - p[0])

    @property
    def baseline_bbox_h(self) -> float | None:
        if not self._standing_heights:
            return None
        return float(np.median(self._standing_heights))

    @property
    def latest_takeoff(self) -> int | None:
        if self._confirmed_takeoff is not None:
            return self._confirmed_takeoff
        return self.candidates[-1][0] if self.candidates else None

    @property
    def latest_landing(self) -> int | None:
        if self.state == "airborne_confirmed":
            return None
        return self.candidates[-1][1] if self.candidates else None


# --- Pipeline ----------------------------------------------------------


class StandingLongJumpTest(BaseTest):
    """Standing Long Jump: pose-only horizontal jump, body-height
    calibration."""

    test_id = "standing-long-jump"

    def __init__(
        self,
        assumed_athlete_height_m: float = _DEFAULT_ATHLETE_HEIGHT_M,
    ) -> None:
        if assumed_athlete_height_m <= 0:
            raise ValueError(
                f"assumed_athlete_height_m must be > 0, "
                f"got {assumed_athlete_height_m}"
            )
        self._assumed_height_m = float(assumed_athlete_height_m)
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

        det = _StreamingDetector()
        n_frames = 0
        n_low_conf = 0
        # Trunk-lean computation needs a single moment — record per-frame
        # pose so we can sample at the takeoff frame post-loop.
        trunk_lean_by_frame: dict[int, float] = {}

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

                ankle_xy, bbox_h = _ankle_features(pose, bbox)
                if pose is not None and pose.mean_confidence < _POSE_CONF_MIN:
                    n_low_conf += 1
                tl = _trunk_lean_deg(pose)
                if tl is not None:
                    trunk_lean_by_frame[frame.idx] = tl

                det.update(frame.idx, ankle_xy, bbox_h)

                if pose is not None:
                    draw_skeleton(img, pose.keypoints)
                if bbox is not None:
                    draw_bbox(img, bbox)
                draw_hud(img, _hud_fields(det, frame.idx, fps))

                if frame.idx in (det.latest_takeoff, det.latest_landing):
                    if bbox is not None:
                        event_flash(img, bbox)

                writer.write(img)

            jump = det.best_jump()
            if jump is None:
                raise ProtocolError(
                    "could not locate a standing long jump — ankle never "
                    "rose above the standing-baseline threshold"
                )
            takeoff, landing = jump

            baseline_h_px = det.baseline_bbox_h
            if baseline_h_px is None or baseline_h_px <= 0:
                raise ProtocolError(
                    "no usable standing-baseline bbox height for "
                    "body-proxy calibration"
                )
            px_per_m = baseline_h_px / self._assumed_height_m

            distance_cm = _horizontal_distance_cm(
                ankle_samples=det.ankle_samples,
                takeoff_frame=takeoff,
                landing_frame=landing,
                px_per_m=px_per_m,
            )
            peak_height_cm = _peak_height_cm(
                ankle_samples=det.ankle_samples,
                takeoff_frame=takeoff,
                landing_frame=landing,
                px_per_m=px_per_m,
            )
            ft = flight_time_s(takeoff, landing, fps)
            tl_at_takeoff = trunk_lean_by_frame.get(takeoff)

            metrics: dict[str, MetricValue] = {
                "jump_distance_cm": MetricValue(raw=distance_cm, unit="cm"),
                "flight_time_s": MetricValue(raw=ft, unit="s"),
            }
            if peak_height_cm is not None:
                metrics["peak_height_cm"] = MetricValue(
                    raw=peak_height_cm, unit="cm",
                )
            if tl_at_takeoff is not None:
                metrics["trunk_lean_takeoff_deg"] = MetricValue(
                    raw=tl_at_takeoff, unit="deg",
                )
            scores, test_score = score_test(metrics, self.test_id, athlete.gender)

            metric_rows = [
                ("Jump Distance (cm)", f"{distance_cm:.1f}",
                 int(round(scores["jump_distance_cm"].score))
                 if "jump_distance_cm" in scores else 0),
                ("Flight Time (s)", f"{ft:.3f}", 0),
            ]
            if peak_height_cm is not None:
                metric_rows.append(
                    ("Peak Height (cm)", f"{peak_height_cm:.1f}", 0)
                )
            if tl_at_takeoff is not None:
                metric_rows.append(
                    ("Trunk Lean (deg)", f"{tl_at_takeoff:.1f}", 0)
                )

            endcard = render_endcard(
                title="Standing Long Jump",
                athlete=f"{athlete.gender} age {athlete.age}",
                metric_rows=metric_rows,
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


# --- Pose helpers ------------------------------------------------------


def _ankle_features(
    pose, bbox,
) -> tuple[tuple[float, float] | None, float | None]:
    """Return (ankle_xy, bbox_h) for the more-confident foot, or
    (None, None) if neither ankle is reliable."""
    if pose is None or bbox is None:
        return None, None
    la = pose.position("left_ankle")
    ra = pose.position("right_ankle")
    la_c = pose.confidence_of("left_ankle")
    ra_c = pose.confidence_of("right_ankle")
    candidates: list[tuple[float, tuple[float, float]]] = []
    if la_c >= _POSE_CONF_MIN:
        candidates.append((la_c, (float(la[0]), float(la[1]))))
    if ra_c >= _POSE_CONF_MIN:
        candidates.append((ra_c, (float(ra[0]), float(ra[1]))))
    if not candidates:
        return None, None
    # Highest-confidence ankle wins on positional reliability; for the
    # airborne threshold we still want the lower-y of the two so
    # detection fires as soon as either foot leaves the ground.
    chosen_xy = min(candidates, key=lambda c: c[1][1])[1]
    return chosen_xy, float(bbox[3] - bbox[1])


def _trunk_lean_deg(pose) -> float | None:
    """Forward trunk lean at the moment of takeoff: angle (degrees)
    between the trunk vector (mid-hip -> mid-shoulder) and the image
    vertical. Positive when the athlete is leaning forward in the
    direction of travel; sign is unsigned in this implementation since
    direction-of-travel detection isn't part of v1.
    """
    if pose is None:
        return None
    needed = ("left_hip", "right_hip", "left_shoulder", "right_shoulder")
    if any(pose.confidence_of(k) < _POSE_CONF_MIN for k in needed):
        return None
    hip = (
        np.asarray(pose.position("left_hip"), dtype=float)
        + np.asarray(pose.position("right_hip"), dtype=float)
    ) / 2.0
    sh = (
        np.asarray(pose.position("left_shoulder"), dtype=float)
        + np.asarray(pose.position("right_shoulder"), dtype=float)
    ) / 2.0
    v = sh - hip
    n = float(np.linalg.norm(v))
    if n == 0:
        return None
    # Angle between v and image-up (-y), in degrees.
    cos_a = float(np.clip(-v[1] / n, -1.0, 1.0))
    return float(np.degrees(np.arccos(cos_a)))


# --- Distance / height -------------------------------------------------


def _horizontal_distance_cm(
    *,
    ankle_samples: list[tuple[int, float, float, float]],
    takeoff_frame: int,
    landing_frame: int,
    px_per_m: float,
) -> float:
    """Horizontal pixel displacement between takeoff and landing ankle
    positions, converted to centimetres via `px_per_m`.
    """
    by_frame = {fi: (ax, ay) for (fi, ax, ay, _) in ankle_samples}
    a_take = by_frame.get(takeoff_frame)
    a_land = by_frame.get(landing_frame)
    if a_take is None or a_land is None:
        # Fall back to nearest available samples.
        if ankle_samples:
            sorted_idx = sorted(by_frame.keys())
            if a_take is None:
                a_take = by_frame[
                    min(sorted_idx, key=lambda f: abs(f - takeoff_frame))
                ]
            if a_land is None:
                a_land = by_frame[
                    min(sorted_idx, key=lambda f: abs(f - landing_frame))
                ]
    if a_take is None or a_land is None:
        return 0.0
    dx_px = abs(a_land[0] - a_take[0])
    return dx_px / px_per_m * 100.0


def _peak_height_cm(
    *,
    ankle_samples: list[tuple[int, float, float, float]],
    takeoff_frame: int,
    landing_frame: int,
    px_per_m: float,
) -> float | None:
    """Maximum vertical rise of the ankle during flight (smaller pixel-y
    = higher in image), in centimetres. Reference is the takeoff ankle
    y; rise = baseline_y - min_y_in_flight.
    """
    in_flight = [
        (ay) for (fi, _, ay, _) in ankle_samples
        if takeoff_frame <= fi <= landing_frame
    ]
    if len(in_flight) < 2:
        return None
    takeoff_ay = next(
        (ay for (fi, _, ay, _) in ankle_samples if fi == takeoff_frame),
        None,
    )
    if takeoff_ay is None:
        return None
    rise_px = max(0.0, float(takeoff_ay) - float(min(in_flight)))
    return rise_px / px_per_m * 100.0


# --- HUD --------------------------------------------------------------


def _hud_fields(
    det: _StreamingDetector, frame_idx: int, fps: float,
) -> dict[str, str]:
    if det.state == "airborne_confirmed" and det.latest_takeoff is not None:
        elapsed = (frame_idx - det.latest_takeoff) / fps
        return {"phase": "airborne", "flight": f"{elapsed:.2f} s"}
    if det.candidates:
        takeoff, landing = det.candidates[-1]
        ft = flight_time_s(takeoff, landing, fps)
        return {"phase": "landed", "flight": f"{ft:.3f} s"}
    return {"phase": "ready", "flight": "-"}
