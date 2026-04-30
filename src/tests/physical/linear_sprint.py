"""Linear Sprint pipeline — single declared distance.

The operator declares which sprint distance is being run (10 / 20 / 30 /
40 m) and places one cone at the finish line. The pipeline:

- detects the finish cone with YOLO-World during the first second
  (calibration window), picking the most-detected stable cluster
- tracks the athlete with ByteTrack from frame 0
- start event = athlete pixel-displacement exceeds a fraction of bbox
  height (motion onset)
- direction of run = sign of the first significant motion
- finish event = athlete center crosses the cone's x-coordinate in the
  run direction
- metric = (finish_frame - start_frame) / fps -> single split keyed by
  declared distance (`time_10m_s`, etc.)

No pixel-to-metre calibration is required: distance is declared, time is
counted in frames. Multi-split (one run -> 10/20/30/40 splits) is deferred
to v2 — needs either field-line homography or a working flat-marker
detector.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import cv2
import numpy as np

from src.core.annotation.overlays import (
    draw_bbox,
    draw_gate,
    draw_hud,
    render_endcard,
)
from src.core.detection.marker_detector import MarkerDetector
from src.core.detection.player_detector import Detection
from src.core.tracking.bytetrack_tracker import ByteTrackTracker, TrackedDetection
from src.core.utils.video_io import frame_iter, video_info
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

# --- Tunables ----------------------------------------------------------

SUPPORTED_DISTANCES_M = (10.0, 20.0, 30.0, 40.0)

_FINISH_CONE_FRAMES = 30          # ~1 s @ 30 fps for cone consensus
_CONE_CLUSTER_RADIUS_PX = 40.0
_CONE_MIN_DETECTIONS = 5
_MOTION_THRESHOLD_FRAC = 0.05     # 5% of bbox-height over 5 frames
_MOTION_WINDOW_FRAMES = 5
# "Stationary" gate: require the athlete's pixel-x to vary < this fraction
# of bbox-height across the last N frames before we accept any motion as
# the start event. Filters out videos where the athlete enters frame
# already running.
_STATIONARY_WINDOW_FRAMES = 15
_STATIONARY_RANGE_FRAC = 0.03
_ENDCARD_HOLD_S = 2.5


# --- Helpers -----------------------------------------------------------


@dataclass
class _Cluster:
    sum_x: float = 0.0
    sum_y: float = 0.0
    count: int = 0

    @property
    def centroid(self) -> np.ndarray:
        return np.array([self.sum_x / self.count, self.sum_y / self.count])


def _consensus_clusters(
    detections: list[Detection],
    radius_px: float,
    min_count: int,
) -> list[tuple[np.ndarray, int]]:
    """Greedy spatial cluster. Returns (centroid, count) sorted by count desc."""
    clusters: list[_Cluster] = []
    r2 = radius_px * radius_px
    for d in detections:
        cx, cy = d.center
        attached = False
        for c in clusters:
            mx, my = c.centroid
            if (mx - cx) ** 2 + (my - cy) ** 2 < r2:
                c.sum_x += cx
                c.sum_y += cy
                c.count += 1
                attached = True
                break
        if not attached:
            clusters.append(_Cluster(cx, cy, 1))
    out = [(c.centroid, c.count) for c in clusters if c.count >= min_count]
    out.sort(key=lambda x: x[1], reverse=True)
    return out


_State = Literal["pre_start", "running", "finished"]


@dataclass
class _RunState:
    state: _State = "pre_start"
    finish_cone_px: np.ndarray | None = None
    start_frame: int | None = None
    finish_frame: int | None = None
    direction_sign: int = 0          # +1 or -1; 0 until determined
    athlete_track_id: int | None = None
    athlete_x_history: list[tuple[int, float]] = field(default_factory=list)
    # Set when the athlete has been stationary for STATIONARY_WINDOW_FRAMES;
    # only after this gate do we accept motion as the start event.
    stationary_confirmed: bool = False


# --- Pipeline ----------------------------------------------------------


class LinearSprintTest(BaseTest):
    """Linear Sprint over a declared distance with one finish cone."""

    test_id = "linear-sprint"

    def __init__(self, distance_m: float = 10.0) -> None:
        if distance_m not in SUPPORTED_DISTANCES_M:
            raise ValueError(
                f"distance_m must be one of {SUPPORTED_DISTANCES_M}, got {distance_m}"
            )
        self._distance_m = float(distance_m)
        self._marker = MarkerDetector()
        self._tracker = ByteTrackTracker()

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
        cone_buffer: list[Detection] = []
        n_frames = 0

        try:
            for frame in frame_iter(video_path):
                n_frames += 1
                img = frame.image

                # --- Finish-cone consensus during the first 1s
                if frame.idx < _FINISH_CONE_FRAMES and state.finish_cone_px is None:
                    cone_buffer.extend(self._marker.detect(img))
                    if frame.idx == _FINISH_CONE_FRAMES - 1:
                        state.finish_cone_px = self._lock_finish_cone(cone_buffer)

                # --- Athlete tracking + state-machine update
                tracked = self._tracker.update(img)
                runner = self._pick_athlete(tracked, state)
                if runner is not None and state.finish_cone_px is not None:
                    self._update_run_state(state, runner, frame.idx, fps)

                # --- Annotations
                self._annotate(img, state, runner, frame.idx, fps)
                writer.write(img)
        except Exception:
            writer.release()
            raise

        if state.finish_cone_px is None:
            writer.release()
            raise DetectionError(
                "finish cone never detected — re-record with the cone in frame "
                "and unobstructed during the first second"
            )
        if state.start_frame is None:
            writer.release()
            if not state.stationary_confirmed:
                raise ProtocolError(
                    "athlete was never stationary at the start — re-record "
                    "with the athlete static behind the start line for at "
                    "least half a second before initiating the sprint"
                )
            raise ProtocolError("no start motion detected — athlete never moved")
        if state.finish_frame is None:
            writer.release()
            raise ProtocolError(
                f"athlete never crossed the finish cone at "
                f"{int(self._distance_m)} m"
            )

        time_s = (state.finish_frame - state.start_frame) / fps
        metric_id = f"time_{int(self._distance_m)}m_s"
        metrics = {metric_id: MetricValue(raw=time_s, unit="s")}
        scores, test_score = score_test(metrics, self.test_id, athlete.gender)

        endcard_rows = (
            [(metric_id.replace("_", " "), f"{time_s:.3f} s",
              int(round(scores[metric_id].score)))]
            if metric_id in scores else []
        )
        endcard = render_endcard(
            title=f"Linear Sprint ({int(self._distance_m)} m)",
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

    # --- Internal helpers ---------------------------------------------

    def _lock_finish_cone(self, buffer: list[Detection]) -> np.ndarray:
        if not buffer:
            raise DetectionError(
                "no cone detections during the first second — "
                "is the finish cone visible at frame 0?"
            )
        clusters = _consensus_clusters(
            buffer, _CONE_CLUSTER_RADIUS_PX, _CONE_MIN_DETECTIONS
        )
        if not clusters:
            raise DetectionError(
                "no stable cone cluster found in the first second — "
                "increase confidence by ensuring an unobstructed cone view"
            )
        return clusters[0][0]  # most-detected centroid

    def _pick_athlete(
        self, tracked: list[TrackedDetection], state: _RunState
    ) -> TrackedDetection | None:
        if not tracked:
            return None
        if state.athlete_track_id is None:
            state.athlete_track_id = tracked[0].track_id
            return tracked[0]
        for t in tracked:
            if t.track_id == state.athlete_track_id:
                return t
        return None

    def _update_run_state(
        self,
        state: _RunState,
        runner: TrackedDetection,
        frame_idx: int,
        fps: float,
    ) -> None:
        athlete_cx = float(runner.center[0])
        cone_cx = float(state.finish_cone_px[0])  # type: ignore[index]
        bbox_h = runner.height

        state.athlete_x_history.append((frame_idx, athlete_cx))

        if state.state == "pre_start":
            # Confirm a stationary period before accepting motion. Once
            # confirmed, the flag stays set — we only want this to gate
            # the *first* start event, not re-fire if the athlete pauses
            # mid-run.
            if (not state.stationary_confirmed
                    and len(state.athlete_x_history) >= _STATIONARY_WINDOW_FRAMES):
                recent_xs = [
                    x for _, x in state.athlete_x_history[-_STATIONARY_WINDOW_FRAMES:]
                ]
                spread = max(recent_xs) - min(recent_xs)
                if spread < _STATIONARY_RANGE_FRAC * bbox_h:
                    state.stationary_confirmed = True

            if (state.stationary_confirmed
                    and len(state.athlete_x_history) > _MOTION_WINDOW_FRAMES):
                fi0, x0 = state.athlete_x_history[-_MOTION_WINDOW_FRAMES - 1]
                dx = athlete_cx - x0
                if abs(dx) > _MOTION_THRESHOLD_FRAC * bbox_h:
                    state.state = "running"
                    state.start_frame = fi0
                    state.direction_sign = 1 if dx > 0 else -1
        elif state.state == "running":
            if state.direction_sign > 0 and athlete_cx >= cone_cx:
                state.finish_frame = frame_idx
                state.state = "finished"
            elif state.direction_sign < 0 and athlete_cx <= cone_cx:
                state.finish_frame = frame_idx
                state.state = "finished"

    def _annotate(
        self,
        img: np.ndarray,
        state: _RunState,
        runner: TrackedDetection | None,
        frame_idx: int,
        fps: float,
    ) -> None:
        if state.finish_cone_px is not None:
            x = int(state.finish_cone_px[0])
            h = img.shape[0]
            gate_state = (
                "passed" if state.state == "finished"
                else "active" if state.state == "running"
                else "pending"
            )
            draw_gate(img, (x, 0), (x, h), state=gate_state)
        if runner is not None:
            draw_bbox(img, runner.bbox_xyxy)
        draw_hud(img, self._hud_fields(state, frame_idx, fps))

    def _hud_fields(
        self, state: _RunState, frame_idx: int, fps: float
    ) -> dict[str, str]:
        distance = f"{int(self._distance_m)} m"
        if state.state == "pre_start":
            return {"phase": "ready", "distance": distance, "time": "-"}
        if state.state == "running" and state.start_frame is not None:
            elapsed = (frame_idx - state.start_frame) / fps
            return {
                "phase": "running", "distance": distance,
                "time": f"{elapsed:.2f} s",
            }
        if (state.state == "finished" and state.start_frame is not None
                and state.finish_frame is not None):
            t = (state.finish_frame - state.start_frame) / fps
            return {
                "phase": "finished", "distance": distance,
                "time": f"{t:.3f} s",
            }
        return {"phase": "-", "distance": distance}
