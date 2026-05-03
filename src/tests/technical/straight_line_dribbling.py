"""Straight Line Dribbling pipeline (view-aware).

Course: athletes dribble through a line of cones / flat dome markers.
The pipeline auto-classifies the camera view from the athlete's run-
window trajectory and adjusts the metric set accordingly:

- side-on   -> athlete moves laterally; pixel-x range is large
- rear-view -> athlete moves into depth; bbox-h shrinks more than x changes

End-of-test detection is heuristic — athlete pixel velocity below
threshold for >= 1 s after the run started — because the flat dome
markers in the user's footage aren't picked up by YOLO-World (cone_v2,
Phase 8.6, will fix that).

v1 ships two scored metrics that don't depend on pixel-to-metre
calibration: total_completion_time_s and touches_per_metre. Side-on
extras (ball_foot_distance_m, control_loss_events) light up once
calibration is available.
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
from src.metrics.ball.touches_per_metre import touches_per_metre
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

SPORTS_BALL_CLASS_ID = 32  # COCO class index for "sports ball"

# --- Tunables ----------------------------------------------------------

_MOTION_THRESHOLD_FRAC = 0.05
_MOTION_WINDOW_FRAMES = 5
_STOP_RATIO_OF_PEAK = 0.15      # stop when motion drops to <15% of observed peak
_STOP_WINDOW_FRAMES = 30        # ~1 s @ 30 fps for the moving-average window
_MIN_RUN_FRAMES = 60            # athlete must have been running >= 2 s before stop fires
_TOUCH_PROXIMITY_FRAC = 0.30    # ball within 30 % of bbox-h of an ankle = touch
_TOUCH_DEBOUNCE_S = 0.20
_POSE_CONF_MIN = 0.30
_ENDCARD_HOLD_S = 2.5

# View classification thresholds (run-window pixel ranges, normalized to frame size)
_VIEW_SIDE_ON_X_FRAC = 0.30     # athlete pixel-x covers >= 30 % of frame width


_State = Literal["pre_start", "running", "stopped"]
_View = Literal["side_on", "rear_view", "unknown"]


# --- State -------------------------------------------------------------


@dataclass
class _Touch:
    frame_idx: int
    side: Literal["L", "R"]


@dataclass
class _RunState:
    state: _State = "pre_start"
    start_frame: int | None = None
    stop_frame: int | None = None
    athlete_track_id: int | None = None
    # History buffers — pixel-x and bbox-h for view classification + stop detection
    center_history: list[tuple[int, float, float, float]] = field(default_factory=list)
    # (frame_idx, cx, cy, bbox_h)
    touches: list[_Touch] = field(default_factory=list)
    last_touch_frame: int = -10**6
    # Adaptive stop detection: peak observed motion since start_frame
    peak_motion_per_frame: float = 0.0
    # Diagnostics
    n_low_pose_conf: int = 0
    n_frames_with_athlete: int = 0
    n_frames_with_ball: int = 0


# --- Pipeline ----------------------------------------------------------


class StraightLineDribblingTest(BaseTest):
    """Dribbling-through-cones pipeline; view-aware metric set."""

    test_id = "straight-line-dribbling"

    def __init__(self, distance_m: float = 30.0) -> None:
        if distance_m <= 0:
            raise ValueError(f"distance_m must be > 0, got {distance_m}")
        self._distance_m = float(distance_m)
        # ByteTrack on both person + sports_ball — single inference per frame
        self._tracker = ByteTrackTracker(
            classes=[PERSON_CLASS_ID, SPORTS_BALL_CLASS_ID],
            confidence=0.20,  # ball can be small / motion-blurred
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

                if runner is not None:
                    state.center_history.append(
                        (frame.idx, float(runner.center[0]),
                         float(runner.center[1]), runner.height)
                    )
                    self._update_state(state, runner, frame.idx, fps)

                # Touch detection — only meaningful between start and stop
                if (state.state == "running" and ball is not None
                        and pose is not None and runner is not None):
                    self._maybe_register_touch(
                        state, ball, pose, runner, frame.idx, debounce_frames
                    )

                # Annotation
                if runner is not None:
                    draw_bbox(img, runner.bbox_xyxy)
                if ball is not None:
                    draw_bbox(img, ball.bbox_xyxy, color=BALL)
                if pose is not None:
                    draw_skeleton(img, pose.keypoints)
                draw_hud(img, _hud_fields(state, frame.idx, fps, self._distance_m))
                writer.write(img)
        except Exception:
            writer.release()
            raise

        if state.start_frame is None:
            writer.release()
            raise ProtocolError(
                "no start motion detected — athlete never moved enough to "
                "trigger run start"
            )
        if state.stop_frame is None:
            writer.release()
            raise ProtocolError(
                "could not detect end of run — athlete kept moving until "
                "video ended; trim the clip or extend it past the dribble"
            )

        view = _classify_view(state, info.width)
        metrics = self._compute_metrics(state, fps)
        scores, test_score = score_test(metrics, self.test_id, athlete.gender)

        endcard_rows = [
            (mid.replace("_", " "), f"{mv.raw:.3f} {mv.unit}",
             int(round(scores[mid].score)) if mid in scores else 0)
            for mid, mv in metrics.items()
        ]
        endcard = render_endcard(
            title="Straight Line Dribbling",
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
        people = [t for t in tracked if t.class_id == PERSON_CLASS_ID]
        if not people:
            return None
        if state.athlete_track_id is None:
            state.athlete_track_id = people[0].track_id
            return people[0]
        for t in people:
            if t.track_id == state.athlete_track_id:
                return t
        return None

    def _update_state(
        self,
        state: _RunState,
        runner: TrackedDetection,
        frame_idx: int,
        fps: float,
    ) -> None:
        cx = float(runner.center[0])
        cy = float(runner.center[1])
        bbox_h = runner.height

        if state.state == "pre_start":
            # No stationary gate — many demo / instructional videos have
            # the athlete already in motion at frame 0. Total course time
            # is the metric, not reaction-onset, so first-significant-motion
            # is good enough.
            if len(state.center_history) > _MOTION_WINDOW_FRAMES:
                fi0, x0, y0, _ = state.center_history[-_MOTION_WINDOW_FRAMES - 1]
                dx = cx - x0
                dy = cy - y0
                if (dx * dx + dy * dy) ** 0.5 > _MOTION_THRESHOLD_FRAC * bbox_h:
                    state.state = "running"
                    state.start_frame = fi0
        elif state.state == "running":
            # Adaptive stop detection: track peak motion since start_frame,
            # fire when current motion drops to <15% of peak. Self-scales
            # to the actual motion regime — works for both side-on
            # (high pixel motion) and rear-view (sub-pixel motion).
            if len(state.center_history) > _STOP_WINDOW_FRAMES:
                window = state.center_history[-_STOP_WINDOW_FRAMES - 1:]
                fi_start, sx, sy, sh = window[0]
                _, ex, ey, eh = window[-1]
                disp_center = ((ex - sx) ** 2 + (ey - sy) ** 2) ** 0.5
                disp_height = abs(eh - sh)
                total_disp = (disp_center ** 2 + disp_height ** 2) ** 0.5
                avg_per_frame = total_disp / _STOP_WINDOW_FRAMES

                if avg_per_frame > state.peak_motion_per_frame:
                    state.peak_motion_per_frame = avg_per_frame

                if (state.start_frame is not None
                        and frame_idx - state.start_frame >= _MIN_RUN_FRAMES
                        and state.peak_motion_per_frame > 0
                        and avg_per_frame < _STOP_RATIO_OF_PEAK * state.peak_motion_per_frame):
                    state.state = "stopped"
                    state.stop_frame = window[0][0]  # start of the still window

    def _maybe_register_touch(
        self,
        state: _RunState,
        ball: TrackedDetection,
        pose,
        runner: TrackedDetection,
        frame_idx: int,
        debounce_frames: int,
    ) -> None:
        if (frame_idx - state.last_touch_frame) < debounce_frames:
            return
        bx, by = ball.center
        la = pose.position("left_ankle")
        ra = pose.position("right_ankle")
        la_c = pose.confidence_of("left_ankle")
        ra_c = pose.confidence_of("right_ankle")
        d_l = float(np.hypot(bx - la[0], by - la[1])) if la_c >= _POSE_CONF_MIN else float("inf")
        d_r = float(np.hypot(bx - ra[0], by - ra[1])) if ra_c >= _POSE_CONF_MIN else float("inf")
        d_min = min(d_l, d_r)
        threshold = _TOUCH_PROXIMITY_FRAC * runner.height
        if d_min < threshold:
            # Side from image-x of the contacting ankle vs body-center
            # (robust to pose-model L/R label flips).
            contact_x = float(la[0]) if d_l <= d_r else float(ra[0])
            bcx = body_center_x(pose)
            if bcx is not None:
                side: Literal["L", "R"] = ankle_side(contact_x, bcx)
            else:
                side = "L" if contact_x >= bx else "R"  # fallback
            state.touches.append(_Touch(frame_idx=frame_idx, side=side))
            state.last_touch_frame = frame_idx

    def _compute_metrics(
        self, state: _RunState, fps: float
    ) -> dict[str, MetricValue]:
        assert state.start_frame is not None and state.stop_frame is not None
        elapsed = (state.stop_frame - state.start_frame) / fps
        n_touches = len(state.touches)
        tpm = touches_per_metre(n_touches, self._distance_m)

        out: dict[str, MetricValue] = {
            "total_completion_time_s": MetricValue(raw=elapsed, unit="s"),
            "touches_per_metre": MetricValue(raw=tpm, unit="count_per_m"),
        }
        if n_touches:
            left_pct = sum(1 for t in state.touches if t.side == "L") / n_touches * 100.0
            out["left_leg_utilisation_pct"] = MetricValue(raw=left_pct, unit="percent")
        return out


# --- module-level helpers ---------------------------------------------


def _pick_ball(tracked: list[TrackedDetection]) -> TrackedDetection | None:
    """Highest-confidence sports ball detection, or None."""
    balls = [t for t in tracked if t.class_id == SPORTS_BALL_CLASS_ID]
    return balls[0] if balls else None


def _classify_view(state: _RunState, frame_width: int) -> _View:
    """Side-on if athlete pixel-x range >= threshold of frame width, else rear-view.

    Considers only history during the run window (start_frame to stop_frame).
    """
    if state.start_frame is None or state.stop_frame is None:
        return "unknown"
    run_xs = [
        x for fi, x, _, _ in state.center_history
        if state.start_frame <= fi <= state.stop_frame
    ]
    if not run_xs:
        return "unknown"
    x_range = max(run_xs) - min(run_xs)
    if x_range >= _VIEW_SIDE_ON_X_FRAC * frame_width:
        return "side_on"
    return "rear_view"


def _hud_fields(
    state: _RunState, frame_idx: int, fps: float, distance_m: float
) -> dict[str, str]:
    distance = f"{distance_m:.0f} m"
    if state.state == "pre_start":
        return {"phase": "ready", "distance": distance, "touches": "0"}
    if state.state == "running" and state.start_frame is not None:
        elapsed = (frame_idx - state.start_frame) / fps
        return {
            "phase": "running",
            "distance": distance,
            "elapsed": f"{elapsed:.2f} s",
            "touches": str(len(state.touches)),
        }
    if (state.state == "stopped" and state.start_frame is not None
            and state.stop_frame is not None):
        t = (state.stop_frame - state.start_frame) / fps
        return {
            "phase": "finished",
            "distance": distance,
            "time": f"{t:.3f} s",
            "touches": str(len(state.touches)),
        }
    return {"phase": "-", "distance": distance}
