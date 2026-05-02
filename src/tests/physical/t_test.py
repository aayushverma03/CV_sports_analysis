"""T-Test (Agility) pipeline.

T-shape course: athlete sprints forward to centre cone, side-shuffles
left, side-shuffles across right, side-shuffles back to centre, then
backpedals to start. Total path A->B->C->B->D->B->A.

v1 ships only the scored metric `total_completion_time_s`. Cone detection
and segment_completion_times are spec-mandated for v1.x, deferred until
the cone-handoff is more reliable across marker types.

Pattern: same start/stop heuristics as Linear Sprint and Straight Line
Dribbling — stationary gate before motion onset, adaptive stop detection
based on peak motion observed during the run.
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
    render_endcard,
)
from src.core.detection.player_detector import PERSON_CLASS_ID
from src.core.pose.estimator import create_pose_estimator
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

# --- Tunables ----------------------------------------------------------

_STATIONARY_WINDOW_FRAMES = 15
_STATIONARY_RANGE_FRAC = 0.03
_MOTION_THRESHOLD_FRAC = 0.05
_MOTION_WINDOW_FRAMES = 5
_STOP_RATIO_OF_PEAK = 0.15
_STOP_WINDOW_FRAMES = 30
# T-Test elite male is 8.8 s; below 6 s is implausible. Require 6 s of
# run before stop fires, which filters out demo-clip blips and
# between-run transitions in compilation videos.
_MIN_RUN_FRAMES = 180         # 6 s @ 30 fps
_POSE_INTERVAL_FRAMES = 3
_ENDCARD_HOLD_S = 2.5


_State = Literal["pre_start", "running", "stopped"]


@dataclass
class _RunState:
    state: _State = "pre_start"
    start_frame: int | None = None
    stop_frame: int | None = None
    track_area: dict[int, float] = field(default_factory=dict)
    stationary_confirmed: bool = False
    center_history: list[tuple[int, float, float, float]] = field(default_factory=list)
    peak_motion_per_frame: float = 0.0


# --- Pipeline ----------------------------------------------------------


class TTestTest(BaseTest):
    """T-Test: total time from motion-onset to motion-stop."""

    test_id = "t-test"

    def __init__(self) -> None:
        self._tracker = ByteTrackTracker(
            classes=[PERSON_CLASS_ID],
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
        last_pose = None

        try:
            for frame in frame_iter(video_path):
                n_frames += 1
                img = frame.image

                tracked = self._tracker.update(img)
                runner = self._pick_athlete(tracked, state)
                pose = last_pose
                if runner is not None and frame.idx % _POSE_INTERVAL_FRAMES == 0:
                    pose = self._pose.estimate_bbox(img, runner.bbox_xyxy)
                    last_pose = pose

                if runner is not None:
                    state.center_history.append(
                        (frame.idx, float(runner.center[0]),
                         float(runner.center[1]), runner.height)
                    )
                    self._update_state(state, runner, frame.idx)

                if runner is not None:
                    draw_bbox(img, runner.bbox_xyxy)
                if pose is not None:
                    draw_skeleton(img, pose.keypoints)
                draw_hud(img, _hud_fields(state, frame.idx, fps))
                writer.write(img)
        except Exception:
            writer.release()
            raise

        if state.start_frame is None:
            writer.release()
            if not state.stationary_confirmed:
                raise ProtocolError(
                    "athlete was never stationary at the start — re-record "
                    "with the athlete still behind start cone A for >= 0.5 s"
                )
            raise ProtocolError("no start motion detected — athlete never moved")
        if state.stop_frame is None:
            writer.release()
            raise ProtocolError(
                "could not detect end of run — athlete kept moving until "
                "video ended; trim the clip past the run completion"
            )

        time_s = (state.stop_frame - state.start_frame) / fps
        metrics = {
            "total_completion_time_s": MetricValue(raw=time_s, unit="s"),
        }
        scores, test_score = score_test(metrics, self.test_id, athlete.gender)

        endcard_rows = [
            (mid.replace("_", " "), f"{mv.raw:.3f} {mv.unit}",
             int(round(scores[mid].score)) if mid in scores else 0)
            for mid, mv in metrics.items()
        ]
        endcard = render_endcard(
            title="T-Test (Agility)",
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

    def _update_state(
        self,
        state: _RunState,
        runner: TrackedDetection,
        frame_idx: int,
    ) -> None:
        cx = float(runner.center[0])
        cy = float(runner.center[1])
        bbox_h = runner.height

        if state.state == "pre_start":
            if (not state.stationary_confirmed
                    and len(state.center_history) >= _STATIONARY_WINDOW_FRAMES):
                recent = state.center_history[-_STATIONARY_WINDOW_FRAMES:]
                xs = [x for _, x, _, _ in recent]
                ys = [y for _, _, y, _ in recent]
                spread = max(
                    max(xs) - min(xs),
                    max(ys) - min(ys),
                )
                if spread < _STATIONARY_RANGE_FRAC * bbox_h:
                    state.stationary_confirmed = True
            if (state.stationary_confirmed
                    and len(state.center_history) > _MOTION_WINDOW_FRAMES):
                fi0, x0, y0, _ = state.center_history[-_MOTION_WINDOW_FRAMES - 1]
                dx = cx - x0
                dy = cy - y0
                if (dx * dx + dy * dy) ** 0.5 > _MOTION_THRESHOLD_FRAC * bbox_h:
                    state.state = "running"
                    state.start_frame = fi0
        elif state.state == "running":
            if len(state.center_history) > _STOP_WINDOW_FRAMES:
                window = state.center_history[-_STOP_WINDOW_FRAMES - 1:]
                _, sx, sy, sh = window[0]
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
                        and avg_per_frame
                            < _STOP_RATIO_OF_PEAK * state.peak_motion_per_frame):
                    state.state = "stopped"
                    state.stop_frame = window[0][0]


# --- helpers ----------------------------------------------------------


def _hud_fields(state: _RunState, frame_idx: int, fps: float) -> dict[str, str]:
    if state.state == "pre_start":
        return {"phase": "ready"}
    if state.state == "running" and state.start_frame is not None:
        elapsed = (frame_idx - state.start_frame) / fps
        return {"phase": "running", "elapsed": f"{elapsed:.2f} s"}
    if (state.state == "stopped" and state.start_frame is not None
            and state.stop_frame is not None):
        t = (state.stop_frame - state.start_frame) / fps
        return {"phase": "finished", "time": f"{t:.3f} s"}
    return {"phase": "-"}
