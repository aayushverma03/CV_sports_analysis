"""Juggling Test pipeline.

Continuous ball juggling without ground contact. Two-pass design with
shared player_picker (multi-person handling):

- Pass 1: ByteTrack jointly tracks athlete (COCO 0) + sports_ball
  (COCO 32). Record per-track person history + per-frame ball positions.
- Post-loop: pick the player track via shared player_picker (area
  dominance → ball proximity + longest sustained motion fallback).
- Pass 2: re-iterate; pose + touch detection only on the picked
  player's bbox; ball is per-frame closest-to-player.

Touch detection: a touch = ball within `touch_proximity_frac * bbox_h`
of any tracked body part, with a debounce window so consecutive frames
of contact count as one touch. A drop (streak end) fires when the ball
is confidently far from the athlete OR no touch for the timeout window.

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
from src.core.pose.orientation import ankle_side, body_center_x
from src.core.tracking.bytetrack_tracker import ByteTrackTracker, TrackedDetection
from src.core.tracking.player_picker import pick_player
from src.core.utils.video_io import frame_iter, video_info
from src.metrics.ball.max_consecutive_touches import max_consecutive_touches
from src.metrics.ball.touches_per_second import touches_per_second
from src.scoring.grade import format_band
from src.tests.base import (
    AnalysisDiagnostics,
    AnalysisResult,
    AthleteProfile,
    BaseTest,
    DetectionError,
    MetricValue,
    ProtocolError,
    score_test,
)

SPORTS_BALL_CLASS_ID = 32

# --- Tunables ----------------------------------------------------------

# Touch detection
# Foot + thigh (knee proxy) only. Including the head was over-counting:
# a single juggle has the ball travel through ankle->knee->head and back,
# triggering 3+ touches per actual contact. Foot/thigh only matches the
# typical juggling-drill protocol and removes the worst false positives.
# Body parts to check for ball contact. Side is now derived from
# image-x position relative to body-center (see _detect_touch),
# not the keypoint's L/R label, so we only need the keypoint names.
_TOUCH_BODY_PARTS: tuple[str, ...] = (
    "left_ankle",
    "right_ankle",
    "left_knee",
    "right_knee",
)
_TOUCH_PROXIMITY_FRAC = 0.20    # ball within 20 % of bbox-h of body part
# Debounce >= half the slowest expected cadence (elite 2.5 Hz -> 0.4 s
# gaps); 0.25 s comfortably below that and well above the 0.15 s we had,
# which was double-counting through the ankle->knee transitions.
_TOUCH_DEBOUNCE_S = 0.25
_POSE_CONF_MIN = 0.30

# Streak / drop detection
# A "drop" needs evidence: either we directly see the ball far from the
# athlete, or we've gone a long stretch without any sign of a touch. A
# brief disappearance (athlete turns away from camera, ball briefly
# occluded by body) should NOT end the streak.
_BALL_FAR_FRAC = 1.5            # |ball_x - athlete_x| >= 1.5 * bbox_h = "far"
_DROP_BALL_FAR_S = 0.5          # ball confirmed far for this long -> drop
_DROP_TIMEOUT_S = 3.0           # absolute fallback: no touches for this long -> drop
_BALL_RECENCY_S = 0.5           # ball-far check requires ball visible within this window

_ENDCARD_HOLD_S = 2.5

# Background tracks too short to be the focal player.
_MIN_TRACK_HISTORY_FRAMES = 60


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
    # Ball-far drop signal: count consecutive frames the ball has been
    # confidently far from the athlete.
    ball_far_run: int = 0
    last_ball_seen_frame: int = -10**6
    n_low_pose_conf: int = 0
    n_frames_with_athlete: int = 0
    n_frames_with_ball: int = 0


# --- Pipeline ----------------------------------------------------------


class JugglingTest(BaseTest):
    """Juggling pipeline — touches counted as ball-near-body-part events."""

    test_id = "juggling"

    def __init__(self) -> None:
        # Person + ball tracked in one ByteTrack call (single inference per frame).
        # Confidence lowered (0.10) so high-toss / motion-blurred ball frames
        # still register; person detections are robust enough at low conf.
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

        # === PASS 1: detect + track all persons + balls ===
        track_history: dict[int, list[tuple[int, float, float, float, float]]] = {}
        people_per_frame: dict[int, list[TrackedDetection]] = {}
        balls_per_frame: dict[int, list[TrackedDetection]] = {}
        n_frames = 0

        for frame in frame_iter(video_path):
            n_frames += 1
            tracked = self._tracker.update(frame.image)
            for p in tracked:
                if p.class_id == PERSON_CLASS_ID:
                    track_history.setdefault(p.track_id, []).append(
                        (frame.idx, float(p.center[0]), float(p.center[1]),
                         p.height, p.width)
                    )
                    people_per_frame.setdefault(frame.idx, []).append(p)
                elif p.class_id == SPORTS_BALL_CLASS_ID:
                    balls_per_frame.setdefault(frame.idx, []).append(p)

        # === Pick THE player track (area dominance, then ball proximity) ===
        ball_positions = {
            fi: [(b.center[0], b.center[1]) for b in balls]
            for fi, balls in balls_per_frame.items()
        }
        player_track_id = pick_player(
            track_history,
            object_positions=ball_positions or None,
            min_history_frames=_MIN_TRACK_HISTORY_FRAMES,
            verbose=True,
        )
        if player_track_id is None:
            if not track_history:
                raise DetectionError("no people were detected in the video")
            raise ProtocolError(
                "could not identify a single player track — "
                "neither pixel-area dominance nor ball-proximity fallback "
                "yielded a winner"
            )

        # === PASS 2: re-iterate, touch detection on the picked player ===
        writer = cv2.VideoWriter(
            str(out_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (info.width, info.height),
        )
        if not writer.isOpened():
            raise ProtocolError(f"could not open VideoWriter at {out_path}")

        state = _RunState()
        debounce_frames = int(round(_TOUCH_DEBOUNCE_S * fps))
        drop_far_frames = int(round(_DROP_BALL_FAR_S * fps))
        drop_timeout_frames = int(round(_DROP_TIMEOUT_S * fps))
        ball_recency_frames = int(round(_BALL_RECENCY_S * fps))

        try:
            for frame in frame_iter(video_path):
                img = frame.image
                runner = _find_track_in_frame(
                    people_per_frame.get(frame.idx, []), player_track_id
                )
                ball = _pick_ball(balls_per_frame.get(frame.idx, []), runner)
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
                        state.ball_far_run = 0

                # Track ball-far signal — only horizontal separation. During
                # juggling the ball travels far vertically (up over the
                # athlete's head and back down), but stays directly above
                # them in x. A real drop = ball rolls away laterally, which
                # is exactly what a horizontal-only distance captures.
                if ball is not None:
                    state.last_ball_seen_frame = frame.idx
                    if runner is not None:
                        bx, _ = ball.center
                        rx, _ = runner.center
                        if abs(bx - rx) > _BALL_FAR_FRAC * runner.height:
                            state.ball_far_run += 1
                        else:
                            state.ball_far_run = 0

                # Drop / streak end: ball confirmed far for >= drop_far_frames,
                # OR absolute timeout with no touch for drop_timeout_frames.
                if state.current_streak > 0:
                    no_touch_for = frame.idx - state.last_touch_frame
                    ball_seen_recently = (
                        frame.idx - state.last_ball_seen_frame
                    ) <= ball_recency_frames
                    drop_signal = False
                    if (ball_seen_recently
                            and state.ball_far_run >= drop_far_frames):
                        drop_signal = True
                    elif no_touch_for >= drop_timeout_frames:
                        drop_signal = True
                    if drop_signal:
                        state.streaks.append(state.current_streak)
                        state.current_streak = 0
                        state.ball_far_run = 0

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


def _find_track_in_frame(
    people: list[TrackedDetection], track_id: int
) -> TrackedDetection | None:
    """Locate the picked-track-id detection in this frame's people."""
    for p in people:
        if p.track_id == track_id:
            return p
    return None


def _pick_ball(
    balls: list[TrackedDetection],
    athlete: TrackedDetection | None,
) -> TrackedDetection | None:
    """Pick the ball closest to the athlete.

    Multi-player training drills typically have several balls in frame
    (other players' balls, balls on the ground). The one geometrically
    closest to our athlete is the one being juggled.
    """
    if not balls:
        return None
    if athlete is None:
        return balls[0]  # highest-conf as fallback
    ax, ay = athlete.center
    return min(
        balls,
        key=lambda b: (b.center[0] - ax) ** 2 + (b.center[1] - ay) ** 2,
    )


def _detect_touch(
    ball: TrackedDetection,
    pose,
    runner: TrackedDetection,
) -> Literal["L", "R"] | None:
    """Return touch side if any tracked body part is in proximity of the ball.

    Side is derived from the contacting keypoint's IMAGE-X relative to
    body center (mid-hip), not the pose model's L/R label — robust to
    pose-model label flips.
    """
    bx, by = ball.center
    threshold = _TOUCH_PROXIMITY_FRAC * runner.height
    best_x: float | None = None
    best_d = float("inf")
    for kp_name in _TOUCH_BODY_PARTS:
        if pose.confidence_of(kp_name) < _POSE_CONF_MIN:
            continue
        kp = pose.position(kp_name)
        d = float(np.hypot(bx - kp[0], by - kp[1]))
        if d < threshold and d < best_d:
            best_d = d
            best_x = float(kp[0])
    if best_x is None:
        return None
    bcx = body_center_x(pose)
    if bcx is None:
        # Fallback: image-side relative to ball x
        return "R" if best_x < bx else "L"
    return ankle_side(best_x, bcx)


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
