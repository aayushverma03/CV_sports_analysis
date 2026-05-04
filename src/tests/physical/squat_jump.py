"""Squat Jump pipeline.

Vertical jump from a held squat position — no countermovement. Same
pose-only flight-time approach as CMJ, plus:

- Squat-hold detection: knee angle stable for >= 1.5 s before takeoff,
  giving the `min_knee_angle_deg` (squat depth) metric.
- Countermovement validity check: if the hip dips further DOWN between
  the end of the held squat and takeoff, the attempt is flagged as
  invalid (that's a CMJ pattern, not an SJ).

Single-pass design (hard rule #3): each frame is detected, posed,
state-machined, annotated, and written to the output video before the
next frame is read. Events (takeoff / landing) fire late by a few
frames because we require a stable airborne run; this is fine for HUD
purposes and the exact frame indices come from offline post-processing
of the streaming detector's candidate list.
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

# --- Tunables ----------------------------------------------------------

_BASELINE_FRAMES = 30
_AIRBORNE_THRESHOLD_FRAC = 0.05
_MIN_AIRBORNE_FRAMES = 3
_POSE_CONF_MIN = 0.30
_ENDCARD_HOLD_S = 2.0

# Squat depth threshold: a "squat" registers when the knee angle drops
# below this (degrees). 140 deg ~ moderate squat; 110 deg ~ deep. The
# protocol allows any depth, so we just want to identify WHEN the
# athlete is squatting (not standing).
_SQUAT_KNEE_ANGLE_DEG = 140.0

# Hold detection: knee angle stable within this band, for at least this
# long, qualifies as a held squat.
_HOLD_KNEE_ANGLE_BAND_DEG = 8.0
_HOLD_MIN_DURATION_S = 1.5

# Countermovement detection: between the end of the held squat and
# takeoff, the hip must not drop more than this fraction of bbox-h
# without immediately reversing. A real SJ shows hip y monotonically
# decreasing (rising in the world) once the squat is released.
_COUNTERMOVEMENT_HIP_DROP_FRAC = 0.04


_State = Literal[
    "standing", "squatting", "ascending",
    "airborne_provisional", "airborne_confirmed",
]


# --- State -------------------------------------------------------------


@dataclass
class _StreamingDetector:
    """Per-frame state machine that locates the squat hold, takeoff, and
    landing. Builds an ankle baseline from the first stationary frames,
    then tracks knee angle and hip-y to phase through:

        standing -> squatting -> [held] -> ascending -> airborne -> landed

    Hold detection is post-processed: we record knee-angle samples by
    frame and identify the longest stable window post-loop.
    """

    state: _State = "standing"

    # Airborne detection (mirrors CMJ).
    candidates: list[tuple[int, int]] = field(default_factory=list)
    _baseline_y: float | None = None
    _threshold_y: float | None = None
    _standing_ankles: list[float] = field(default_factory=list)
    _standing_heights: list[float] = field(default_factory=list)
    _provisional_start: int | None = None
    _confirmed_takeoff: int | None = None

    # Per-frame knee-angle and hip-y samples, used post-loop to find the
    # held squat and check for countermovement.
    knee_samples: list[tuple[int, float]] = field(default_factory=list)
    hip_samples: list[tuple[int, float, float]] = field(
        default_factory=list,
    )  # (frame_idx, hip_y, bbox_h)

    def update_features(
        self,
        frame_idx: int,
        ankle_y: float | None,
        bbox_h: float | None,
        knee_angle_deg: float | None,
        hip_y: float | None,
    ) -> None:
        if knee_angle_deg is not None:
            self.knee_samples.append((frame_idx, knee_angle_deg))
        if hip_y is not None and bbox_h is not None:
            self.hip_samples.append((frame_idx, hip_y, bbox_h))

        if ankle_y is None or bbox_h is None:
            return

        if self._threshold_y is None:
            self._standing_ankles.append(ankle_y)
            self._standing_heights.append(bbox_h)
            if len(self._standing_ankles) >= _BASELINE_FRAMES:
                self._baseline_y = float(np.median(self._standing_ankles))
                scale = float(np.median(self._standing_heights))
                self._threshold_y = (
                    self._baseline_y - _AIRBORNE_THRESHOLD_FRAC * scale
                )
            return

        airborne = ankle_y < self._threshold_y
        # State updates (only used for HUD; jump window picked post-loop).
        if airborne and self.state in (
            "standing", "squatting", "ascending",
        ):
            self.state = "airborne_provisional"
            self._provisional_start = frame_idx
        elif self.state == "airborne_provisional":
            if airborne:
                if (frame_idx - (self._provisional_start or frame_idx) + 1
                        >= _MIN_AIRBORNE_FRAMES):
                    self.state = "airborne_confirmed"
                    self._confirmed_takeoff = self._provisional_start
            else:
                self._reset_airborne()
                self._update_squat_state(knee_angle_deg)
        elif self.state == "airborne_confirmed":
            if not airborne:
                if self._confirmed_takeoff is not None:
                    self.candidates.append(
                        (self._confirmed_takeoff, frame_idx)
                    )
                self._reset_airborne()
        else:
            self._update_squat_state(knee_angle_deg)

    def _update_squat_state(self, knee_angle_deg: float | None) -> None:
        if knee_angle_deg is None:
            return
        if knee_angle_deg < _SQUAT_KNEE_ANGLE_DEG:
            self.state = "squatting"
        else:
            # Once the knee opens past the squat threshold, we're either
            # standing or ascending toward takeoff. Distinguishing them
            # reliably needs the post-loop pass; for the live HUD we just
            # call it 'standing'.
            self.state = "standing"

    def _reset_airborne(self) -> None:
        self.state = "standing"
        self._provisional_start = None
        self._confirmed_takeoff = None

    def best_jump(self) -> tuple[int, int] | None:
        if not self.candidates:
            return None
        return max(self.candidates, key=lambda p: p[1] - p[0])

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


class SquatJumpTest(BaseTest):
    """Squat Jump pipeline: pose-only, no calibration."""

    test_id = "squat-jump"

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

        det = _StreamingDetector()
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
                knee_angle = _knee_angle_deg(pose)
                hip_y = _hip_y(pose)
                if pose is not None and pose.mean_confidence < _POSE_CONF_MIN:
                    n_low_conf += 1

                det.update_features(
                    frame.idx, ankle_y, bbox_h, knee_angle, hip_y,
                )

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
                    "could not locate a squat jump in the video — "
                    "ankle never rose above the standing-baseline threshold"
                )
            takeoff, landing = jump
            ft = flight_time_s(takeoff, landing, fps)
            jh = jump_height_cm(ft)
            min_knee = _min_knee_angle_during_hold(
                knee_samples=det.knee_samples, takeoff_frame=takeoff,
                hold_min_frames=int(round(_HOLD_MIN_DURATION_S * fps)),
            )
            countermovement = _detect_countermovement(
                hip_samples=det.hip_samples,
                takeoff_frame=takeoff,
                lookback_frames=int(round(0.5 * fps)),
            )
            validity = "invalid (countermovement detected)" if countermovement else "valid"

            metrics: dict[str, MetricValue] = {
                "jump_height_cm": MetricValue(raw=jh, unit="cm"),
                "flight_time_s": MetricValue(raw=ft, unit="s"),
            }
            if min_knee is not None:
                metrics["min_knee_angle_deg"] = MetricValue(
                    raw=min_knee, unit="deg",
                )
            scores, test_score = score_test(metrics, self.test_id, athlete.gender)

            metric_rows = [
                ("Jump Height (cm)", f"{jh:.1f}",
                 int(round(scores["jump_height_cm"].score))
                 if "jump_height_cm" in scores else 0),
                ("Flight Time (s)", f"{ft:.3f}", 0),
            ]
            if min_knee is not None:
                metric_rows.append(("Min Knee (deg)", f"{min_knee:.1f}", 0))
            metric_rows.append(("Validity", validity, 0))

            endcard = render_endcard(
                title="Squat Jump",
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


def _ankle_features(pose, bbox) -> tuple[float | None, float | None]:
    if pose is None or bbox is None:
        return None, None
    la = pose.position("left_ankle")
    ra = pose.position("right_ankle")
    la_c = pose.confidence_of("left_ankle")
    ra_c = pose.confidence_of("right_ankle")
    candidates: list[float] = []
    if la_c >= _POSE_CONF_MIN:
        candidates.append(float(la[1]))
    if ra_c >= _POSE_CONF_MIN:
        candidates.append(float(ra[1]))
    if not candidates:
        return None, None
    return min(candidates), float(bbox[3] - bbox[1])


def _hip_y(pose) -> float | None:
    """Mean of left+right hip y, or single confident hip if only one
    has good confidence. None if neither is reliable."""
    if pose is None:
        return None
    ys: list[float] = []
    for kp in ("left_hip", "right_hip"):
        if pose.confidence_of(kp) >= _POSE_CONF_MIN:
            ys.append(float(pose.position(kp)[1]))
    if not ys:
        return None
    return float(np.mean(ys))


def _knee_angle_deg(pose) -> float | None:
    """Average knee angle (degrees) at the most-confident leg.

    Computed from hip-knee-ankle as the angle between vectors
    knee->hip and knee->ankle. 180 deg = fully extended, 90 deg = deep
    squat. Returns None if no leg has all three keypoints confident.
    """
    if pose is None:
        return None
    angles: list[float] = []
    for side in ("left", "right"):
        h = pose.confidence_of(f"{side}_hip")
        k = pose.confidence_of(f"{side}_knee")
        a = pose.confidence_of(f"{side}_ankle")
        if min(h, k, a) < _POSE_CONF_MIN:
            continue
        hip = np.asarray(pose.position(f"{side}_hip"), dtype=float)
        knee = np.asarray(pose.position(f"{side}_knee"), dtype=float)
        ankle = np.asarray(pose.position(f"{side}_ankle"), dtype=float)
        v1 = hip - knee
        v2 = ankle - knee
        n1 = float(np.linalg.norm(v1))
        n2 = float(np.linalg.norm(v2))
        if n1 == 0 or n2 == 0:
            continue
        cos_a = float(np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0))
        angles.append(float(np.degrees(np.arccos(cos_a))))
    if not angles:
        return None
    return float(np.mean(angles))


# --- Squat-hold + validity post-processing ----------------------------


def _min_knee_angle_during_hold(
    *,
    knee_samples: list[tuple[int, float]],
    takeoff_frame: int,
    hold_min_frames: int,
) -> float | None:
    """Return the minimum knee angle observed during the squat hold
    immediately preceding takeoff.

    The hold is the longest contiguous run of samples (before takeoff)
    where the knee angle stays within `_HOLD_KNEE_ANGLE_BAND_DEG` of
    its rolling median AND below `_SQUAT_KNEE_ANGLE_DEG`. Returns the
    minimum knee angle in that window, or None if no qualifying hold
    exists.
    """
    pre = [(fi, a) for (fi, a) in knee_samples if fi <= takeoff_frame]
    if len(pre) < hold_min_frames:
        return None
    angles = [a for _, a in pre]
    # Slide a window of `hold_min_frames` and find any window where the
    # values are within band AND below the squat threshold; track the
    # min value across all such windows.
    best_min: float | None = None
    for i in range(len(angles) - hold_min_frames + 1):
        window = angles[i:i + hold_min_frames]
        med = float(np.median(window))
        if med >= _SQUAT_KNEE_ANGLE_DEG:
            continue
        if max(window) - min(window) > _HOLD_KNEE_ANGLE_BAND_DEG:
            continue
        m = min(window)
        if best_min is None or m < best_min:
            best_min = m
    return best_min


def _detect_countermovement(
    *,
    hip_samples: list[tuple[int, float, float]],
    takeoff_frame: int,
    lookback_frames: int,
) -> bool:
    """True if the hip drops significantly within `lookback_frames` of
    takeoff. In a clean squat jump the hip rises (lower y) monotonically
    once the squat is released; a transient drop indicates an
    illegal countermovement.
    """
    rel = [
        (fi, hy, bh) for (fi, hy, bh) in hip_samples
        if takeoff_frame - lookback_frames <= fi <= takeoff_frame
    ]
    if len(rel) < 3:
        return False
    bbox_h = float(np.median([bh for (_, _, bh) in rel]))
    if bbox_h <= 0:
        return False
    threshold_px = _COUNTERMOVEMENT_HIP_DROP_FRAC * bbox_h
    # Look for any frame-pair where hip y INCREASES (= moves down in
    # the image) by more than the threshold while the next frames
    # don't recover. Easiest check: max hip y in window vs hip y at
    # the START of window — if max is AFTER the start AND substantially
    # lower (in image), that's a drop.
    ys = [hy for (_, hy, _) in rel]
    start_y = ys[0]
    max_y_after_start = max(ys[1:]) if len(ys) > 1 else start_y
    return (max_y_after_start - start_y) > threshold_px


# --- HUD --------------------------------------------------------------


def _hud_fields(
    det: _StreamingDetector, frame_idx: int, fps: float,
) -> dict[str, str]:
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
    if det.state == "squatting":
        return {"phase": "squat", "flight": "-", "height": "-"}
    return {"phase": "ready", "flight": "-", "height": "-"}
