"""Foot Tapping Test pipeline.

Athlete stands over a ball and taps it alternately with each foot for
30 s. Measures lower-limb cyclic speed.

Pipeline reuses the juggling patterns:
- ByteTrack on COCO 0 (person) + 32 (sports_ball), single inference
- cumulative-area athlete picker (focal player = closest to camera for
  most of the video) with largest-in-frame fallback
- closest-ball pick (filters out other balls in frame)
- ankle-only touch detection with debounce

Metrics:
- total_taps (scored)
- taps_per_second = total / video_duration (scored)
- left_taps, right_taps (informational)

The spec mandates a 30 s window. The pipeline measures the actual video
duration; if a clip is shorter, total_taps will scale down accordingly,
but taps_per_second is normalised so it stays comparable.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import cv2
import numpy as np

from src.core.annotation.overlays import (
    BALL,
    draw_bbox,
    draw_hud,
    draw_skeleton,
    render_endcard,
)
from src.core.detection.player_detector import PERSON_CLASS_ID
from src.core.pose.estimator import create_pose_estimator
from src.core.pose.orientation import ankle_side, body_center_x
from src.core.tracking.bytetrack_tracker import ByteTrackTracker, TrackedDetection
from src.core.utils.video_io import frame_iter, video_info
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

SPORTS_BALL_CLASS_ID = 32

# --- Tunables ----------------------------------------------------------

_TOUCH_PROXIMITY_FRAC = 0.20    # ball within 20 % of bbox-h of an ankle
_TOUCH_DEBOUNCE_S = 0.15        # min 0.15 s between taps -> max 6.7 Hz cap
_POSE_CONF_MIN = 0.30
_ENDCARD_HOLD_S = 2.5


# --- State -------------------------------------------------------------


@dataclass
class _Tap:
    frame_idx: int
    side: Literal["L", "R"]


@dataclass
class _RunState:
    taps: list[_Tap] = field(default_factory=list)
    last_tap_frame: int = -10**6
    track_area: dict[int, float] = field(default_factory=dict)
    # Zone state per IMAGE-SIDE (not pose-model L/R label) — robust to
    # pose-label flips. Image-left ankle vs image-right ankle, mapped
    # to athlete-anatomical L/R when registering the tap.
    image_left_in_zone: bool = False
    image_right_in_zone: bool = False
    n_low_pose_conf: int = 0
    n_frames_with_athlete: int = 0
    n_frames_with_ball: int = 0


# --- Pipeline ----------------------------------------------------------


class FootTappingTest(BaseTest):
    """Foot-tap counting pipeline: ankle ↔ ball proximity events."""

    test_id = "foot-tapping"

    def __init__(self) -> None:
        self._tracker = ByteTrackTracker(
            classes=[PERSON_CLASS_ID, SPORTS_BALL_CLASS_ID],
            confidence=0.10,
        )
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

        state = _RunState()
        n_frames = 0
        debounce_frames = int(round(_TOUCH_DEBOUNCE_S * fps))

        try:
            for frame in frame_iter(video_path):
                n_frames += 1
                img = frame.image

                tracked = self._tracker.update(img)
                runner = self._pick_athlete(tracked, state)
                ball = _pick_ball(tracked, runner)
                if runner is not None:
                    state.n_frames_with_athlete += 1
                if ball is not None:
                    state.n_frames_with_ball += 1

                pose = (
                    self._pose.estimate_bbox(img, runner.bbox_xyxy)
                    if runner is not None else None
                )
                if pose is not None and pose.mean_confidence < _POSE_CONF_MIN:
                    state.n_low_pose_conf += 1

                # Edge-triggered tap detection (zone state by IMAGE-SIDE).
                if (ball is not None and pose is not None
                        and runner is not None):
                    bcx = body_center_x(pose)
                    img_left_now, img_right_now = _ankles_in_zone(
                        ball, pose, runner, bcx
                    )
                    debounced = (
                        (frame.idx - state.last_tap_frame) >= debounce_frames
                    )
                    new_side: str | None = None
                    new_x: float | None = None
                    if img_left_now and not state.image_left_in_zone and debounced:
                        new_side = "image_left"
                        # find x of ankle on image-left side that triggered
                        new_x = _ankle_x_on_side(pose, bcx, image_left=True)
                    elif img_right_now and not state.image_right_in_zone and debounced:
                        new_side = "image_right"
                        new_x = _ankle_x_on_side(pose, bcx, image_left=False)
                    if new_side is not None and new_x is not None and bcx is not None:
                        side = ankle_side(new_x, bcx)
                        state.taps.append(_Tap(frame_idx=frame.idx, side=side))
                        state.last_tap_frame = frame.idx
                    state.image_left_in_zone = img_left_now
                    state.image_right_in_zone = img_right_now
                else:
                    state.image_left_in_zone = False
                    state.image_right_in_zone = False

                # Annotation
                if runner is not None:
                    draw_bbox(img, runner.bbox_xyxy)
                if ball is not None:
                    draw_bbox(img, ball.bbox_xyxy, color=BALL)
                if pose is not None:
                    draw_skeleton(img, pose.keypoints)
                draw_hud(img, _hud_fields(state, frame.idx, fps))
                writer.write(img)
        except Exception:
            writer.release()
            raise

        if not state.taps:
            writer.release()
            raise ProtocolError(
                "no taps detected — ball or pose tracking failed; "
                "check ball visibility and pose confidence"
            )

        duration_s = n_frames / fps if fps > 0 else 0.0
        metrics = self._compute_metrics(state, duration_s)
        scores, test_score = score_test(metrics, self.test_id, athlete.gender)

        endcard_rows = [
            (mid.replace("_", " "), f"{mv.raw:.3f} {mv.unit}",
             int(round(scores[mid].score)) if mid in scores else 0)
            for mid, mv in metrics.items()
        ]
        endcard = render_endcard(
            title="Foot Tapping (30 s)",
            athlete=f"{athlete.gender} age {athlete.age}",
            metric_rows=endcard_rows,
            test_score=int(round(test_score.score)),
            band=format_band(test_score.band),
            size=(info.width, info.height),
        )
        for _ in range(int(fps * _ENDCARD_HOLD_S)):
            writer.write(endcard)
        writer.release()

        return AnalysisResult(
            test_id=self.test_id,
            athlete=athlete,
            metrics=metrics,
            scores=scores,
            test_score=test_score,
            annotated_video_path=out_path,
            diagnostics=AnalysisDiagnostics(
                fps_input=fps, duration_s=duration_s
            ),
        )

    # --- internal helpers ---------------------------------------------

    def _pick_athlete(
        self, tracked: list[TrackedDetection], state: _RunState
    ) -> TrackedDetection | None:
        """Cumulative-area dominant track, with largest-in-frame fallback."""
        people = [t for t in tracked if t.class_id == PERSON_CLASS_ID]
        for t in people:
            state.track_area[t.track_id] = (
                state.track_area.get(t.track_id, 0.0) + t.height * t.width
            )
        if not people:
            return None
        if state.track_area:
            dominant_id = max(state.track_area, key=state.track_area.get)
            for t in people:
                if t.track_id == dominant_id:
                    return t
        return max(people, key=lambda t: t.height * t.width)

    def _compute_metrics(
        self, state: _RunState, duration_s: float
    ) -> dict[str, MetricValue]:
        total = len(state.taps)
        tps = (total / duration_s) if duration_s > 0 else 0.0
        left = sum(1 for t in state.taps if t.side == "L")
        right = total - left
        return {
            "total_taps": MetricValue(raw=total, unit="count"),
            "taps_per_second": MetricValue(raw=tps, unit="hz"),
            "left_taps": MetricValue(raw=left, unit="count"),
            "right_taps": MetricValue(raw=right, unit="count"),
        }


# --- module-level helpers ---------------------------------------------


def _pick_ball(
    tracked: list[TrackedDetection],
    athlete: TrackedDetection | None,
) -> TrackedDetection | None:
    """Closest ball to the athlete (filters out other balls in frame)."""
    balls = [t for t in tracked if t.class_id == SPORTS_BALL_CLASS_ID]
    if not balls:
        return None
    if athlete is None:
        return balls[0]
    ax, ay = athlete.center
    return min(
        balls,
        key=lambda b: (b.center[0] - ax) ** 2 + (b.center[1] - ay) ** 2,
    )


def _ankles_in_zone(
    ball: TrackedDetection,
    pose,
    runner: TrackedDetection,
    bcx: float | None,
) -> tuple[bool, bool]:
    """Return (image_left_in_zone, image_right_in_zone).

    Either ankle keypoint within touch-proximity of the ball is
    classified by IMAGE-X position relative to body-center, not by the
    pose model's L/R label. This is robust to pose-label flips: the
    same physical foot always classifies to the same image side.

    If the body center can't be determined (both hips low confidence),
    we fall back to pose-model labels.
    """
    bx, by = ball.center
    threshold = _TOUCH_PROXIMITY_FRAC * runner.height
    img_left_in = False
    img_right_in = False
    for kp_name in ("left_ankle", "right_ankle"):
        if pose.confidence_of(kp_name) < _POSE_CONF_MIN:
            continue
        akp = pose.position(kp_name)
        if float(np.hypot(bx - akp[0], by - akp[1])) >= threshold:
            continue
        if bcx is None:
            # Fallback: trust the pose label
            if kp_name == "left_ankle":
                img_right_in = True  # left_ankle is athlete's left = image-right
            else:
                img_left_in = True
        else:
            if float(akp[0]) < bcx:
                img_left_in = True
            else:
                img_right_in = True
    return img_left_in, img_right_in


def _ankle_x_on_side(
    pose,
    bcx: float | None,
    *,
    image_left: bool,
) -> float | None:
    """Return the image-x of the in-zone ankle on the requested image side."""
    if bcx is None:
        return None
    candidates: list[float] = []
    for kp_name in ("left_ankle", "right_ankle"):
        if pose.confidence_of(kp_name) < _POSE_CONF_MIN:
            continue
        x = float(pose.position(kp_name)[0])
        if image_left and x < bcx:
            candidates.append(x)
        elif not image_left and x >= bcx:
            candidates.append(x)
    if not candidates:
        return None
    # If both ankles ended up on the same side (rare — overlapping feet),
    # pick the one closer to body-center.
    return min(candidates, key=lambda x: abs(x - bcx))


def _hud_fields(state: _RunState, frame_idx: int, fps: float) -> dict[str, str]:
    elapsed_s = (frame_idx + 1) / fps
    total = len(state.taps)
    tps = (total / elapsed_s) if elapsed_s > 0 else 0.0
    return {
        "phase": "tapping",
        "total": str(total),
        "rate": f"{tps:.2f}/s",
        "elapsed": f"{elapsed_s:.1f}s",
    }
