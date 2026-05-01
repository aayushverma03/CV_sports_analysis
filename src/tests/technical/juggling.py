"""Juggling Test pipeline.

Continuous ball juggling without ground contact. Pipeline:

- ByteTrack jointly tracks athlete (COCO 0) + sports_ball (COCO 32)
- per-frame pose for ankles, knees, head
- a touch = ball within `touch_proximity_frac * bbox_h` of any tracked
  body part, with a debounce window so consecutive frames of contact
  count as one touch
- a drop (streak end) fires when no touch has been registered for
  `drop_no_touch_s` seconds — simpler and more robust than ground-plane
  detection and works without calibration

Metrics: max_consecutive_touches (scored), touches_per_second (scored),
plus informational total_ball_touches and left_leg_utilisation_pct.
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
from src.core.tracking.bytetrack_tracker import ByteTrackTracker, TrackedDetection
from src.core.utils.video_io import frame_iter, video_info
from src.metrics.ball.max_consecutive_touches import max_consecutive_touches
from src.metrics.ball.touches_per_second import touches_per_second
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

# Touch detection
_TOUCH_BODY_PARTS: tuple[tuple[str, str], ...] = (
    ("left_ankle", "L"),
    ("right_ankle", "R"),
    ("left_knee", "L"),
    ("right_knee", "R"),
    ("nose", "head"),
)
_TOUCH_PROXIMITY_FRAC = 0.20    # ball within 20 % of bbox-h of body part
_TOUCH_DEBOUNCE_S = 0.15        # prevents consecutive contact frames double-counting
_POSE_CONF_MIN = 0.30

# Streak / drop detection
_DROP_NO_TOUCH_S = 0.7          # >= 0.7 s without a touch = ball dropped

_ENDCARD_HOLD_S = 2.5


# --- State -------------------------------------------------------------


@dataclass
class _Touch:
    frame_idx: int
    side: Literal["L", "R", "head"]


@dataclass
class _RunState:
    touches: list[_Touch] = field(default_factory=list)
    streaks: list[int] = field(default_factory=list)
    current_streak: int = 0
    last_touch_frame: int = -10**6
    first_touch_frame: int | None = None
    athlete_track_id: int | None = None
    n_low_pose_conf: int = 0
    n_frames_with_athlete: int = 0
    n_frames_with_ball: int = 0


# --- Pipeline ----------------------------------------------------------


class JugglingTest(BaseTest):
    """Juggling pipeline — touches counted as ball-near-body-part events."""

    test_id = "juggling"

    def __init__(self) -> None:
        # Person + ball tracked in one ByteTrack call (single inference per frame).
        # Lower confidence on the ball — small + motion-blurred at 30 fps.
        self._tracker = ByteTrackTracker(
            classes=[PERSON_CLASS_ID, SPORTS_BALL_CLASS_ID],
            confidence=0.20,
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
        drop_frames = int(round(_DROP_NO_TOUCH_S * fps))

        try:
            for frame in frame_iter(video_path):
                n_frames += 1
                img = frame.image

                tracked = self._tracker.update(img)
                runner = self._pick_athlete(tracked, state)
                ball = _pick_ball(tracked)
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

                # Touch detection
                if (ball is not None and pose is not None
                        and runner is not None
                        and (frame.idx - state.last_touch_frame) >= debounce_frames):
                    side = _detect_touch(ball, pose, runner)
                    if side is not None:
                        state.touches.append(_Touch(frame_idx=frame.idx, side=side))
                        state.last_touch_frame = frame.idx
                        if state.first_touch_frame is None:
                            state.first_touch_frame = frame.idx
                        state.current_streak += 1

                # Drop / streak end: no touch for `drop_frames` after at least one touch
                if (state.current_streak > 0
                        and (frame.idx - state.last_touch_frame) >= drop_frames):
                    state.streaks.append(state.current_streak)
                    state.current_streak = 0

                # Annotation
                if runner is not None:
                    draw_bbox(img, runner.bbox_xyxy)
                if ball is not None:
                    draw_bbox(img, ball.bbox_xyxy, color=BALL)
                if pose is not None:
                    draw_skeleton(img, pose.keypoints)
                draw_hud(img, _hud_fields(state, frame.idx, fps))
                writer.write(img)

            # Close any open streak at end-of-video
            if state.current_streak > 0:
                state.streaks.append(state.current_streak)
                state.current_streak = 0
        except Exception:
            writer.release()
            raise

        if not state.touches:
            writer.release()
            raise ProtocolError(
                "no touches detected — ball or pose tracking failed; "
                "check ball visibility and lighting"
            )

        metrics = self._compute_metrics(state, fps)
        scores, test_score = score_test(metrics, self.test_id, athlete.gender)

        endcard_rows = [
            (mid.replace("_", " "), f"{mv.raw:.3f} {mv.unit}",
             int(round(scores[mid].score)) if mid in scores else 0)
            for mid, mv in metrics.items()
        ]
        endcard = render_endcard(
            title="Juggling Test",
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
                fps_input=fps, duration_s=n_frames / fps if fps > 0 else 0.0
            ),
        )

    # --- internal helpers ---------------------------------------------

    def _pick_athlete(
        self, tracked: list[TrackedDetection], state: _RunState
    ) -> TrackedDetection | None:
        """Largest person in frame = closest to camera = focal athlete.

        Re-locks if the previously locked track is no longer present
        (multi-person scenes confuse ByteTrack; we don't want to lose the
        athlete forever just because their ID got reassigned).
        """
        people = [t for t in tracked if t.class_id == PERSON_CLASS_ID]
        if not people:
            return None
        if state.athlete_track_id is not None:
            for t in people:
                if t.track_id == state.athlete_track_id:
                    return t
        largest = max(people, key=lambda t: t.height * t.width)
        state.athlete_track_id = largest.track_id
        return largest

    def _compute_metrics(
        self, state: _RunState, fps: float
    ) -> dict[str, MetricValue]:
        max_streak = max_consecutive_touches(state.streaks)
        total_touches = len(state.touches)
        # active duration = first touch -> last touch
        last_frame = state.touches[-1].frame_idx
        first_frame = state.first_touch_frame or 0
        active_s = (last_frame - first_frame) / fps if last_frame > first_frame else 0.0
        tps = touches_per_second(total_touches, active_s)

        out: dict[str, MetricValue] = {
            "max_consecutive_touches": MetricValue(raw=max_streak, unit="count"),
            "touches_per_second": MetricValue(raw=tps, unit="hz"),
            "total_ball_touches": MetricValue(raw=total_touches, unit="count"),
        }
        # left_leg_utilisation_pct: foot+knee touches only (head excluded)
        leg_touches = [t for t in state.touches if t.side in ("L", "R")]
        if leg_touches:
            left_pct = sum(1 for t in leg_touches if t.side == "L") / len(leg_touches) * 100.0
            out["left_leg_utilisation_pct"] = MetricValue(raw=left_pct, unit="percent")
        return out


# --- module-level helpers ---------------------------------------------


def _pick_ball(tracked: list[TrackedDetection]) -> TrackedDetection | None:
    balls = [t for t in tracked if t.class_id == SPORTS_BALL_CLASS_ID]
    return balls[0] if balls else None


def _detect_touch(
    ball: TrackedDetection,
    pose,
    runner: TrackedDetection,
) -> Literal["L", "R", "head"] | None:
    """Return touch side if ball is within proximity of a tracked body part, else None.

    Picks the body part with the smallest pixel distance below threshold.
    """
    bx, by = ball.center
    threshold = _TOUCH_PROXIMITY_FRAC * runner.height
    best_side: str | None = None
    best_d = float("inf")
    for kp_name, side_label in _TOUCH_BODY_PARTS:
        if pose.confidence_of(kp_name) < _POSE_CONF_MIN:
            continue
        kp = pose.position(kp_name)
        d = float(np.hypot(bx - kp[0], by - kp[1]))
        if d < threshold and d < best_d:
            best_d = d
            best_side = side_label
    return best_side  # type: ignore[return-value]


def _hud_fields(state: _RunState, frame_idx: int, fps: float) -> dict[str, str]:
    elapsed = (
        (frame_idx - state.first_touch_frame) / fps
        if state.first_touch_frame is not None else 0.0
    )
    total = len(state.touches)
    tps = (total / elapsed) if elapsed > 0 else 0.0
    return {
        "phase": "juggling" if state.current_streak > 0 else "ready",
        "streak": str(state.current_streak),
        "total": str(total),
        "rate": f"{tps:.2f}/s",
    }
