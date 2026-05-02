"""T-Test (Agility) pipeline.

T-shape course: athlete sprints forward to centre cone, side-shuffles
left, side-shuffles across right, side-shuffles back to centre, then
backpedals to start. Total path A->B->C->B->D->B->A.

v1 ships only the scored metric `total_completion_time_s`. Cone
detection and segment_completion_times are spec-mandated for v1.x,
deferred until the cone-handoff is more reliable across marker types.

Multi-person handling: the user's recordings often contain a coach
plus the test player in the same frame. Rather than depend on the
coach having less cumulative bbox area (which fails when the coach is
in frame longer / closer to camera), we track ALL detected persons
and post-loop pick the track whose smoothed motion-magnitude has the
longest contiguous run above a fraction of the track's mean bbox-h.
The actual test-doer is whoever sprints / shuffles for the longest
sustained stretch — coaches don't.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from src.core.annotation.overlays import (
    ATHLETE_BBOX,
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
    DetectionError,
    MetricValue,
    ProtocolError,
    score_test,
)

# --- Tunables ----------------------------------------------------------

# Per-track motion analysis (post-loop longest-run finder).
# 60-frame (2 s) smoothing absorbs brief cone-touch pauses without
# breaking the run into shorter segments. T-Test players briefly
# decelerate at each cone touch (5 of them); a 1 s window wasn't
# wide enough to bridge those.
_MOTION_SMOOTH_WINDOW_FRAMES = 60
# Threshold: per-frame smoothed motion > this fraction of mean bbox-h
# is "in motion". 3% of a 200-px bbox = 6 px/frame, ~brisk jog speed.
_MOTION_THRESHOLD_FRAC = 0.03
# Adjacent above-threshold segments separated by a sub-threshold gap
# shorter than this are merged — the gap is treated as a brief
# in-test deceleration, not a real stop.
_GAP_MERGE_FRAMES = 30
# T-Test elite male is 8.8 s; below 6 s is implausible.
_MIN_RUN_FRAMES = 180

# Background tracks too small to consider (likely far-background people).
_MIN_TRACK_HISTORY_FRAMES = 60         # 2 s minimum history to be considered
_NON_FOCAL_BBOX_COLOR = (110, 110, 110)
_POSE_INTERVAL_FRAMES = 3
_ENDCARD_HOLD_S = 2.5


# --- State -------------------------------------------------------------


@dataclass
class _RunState:
    # Per-track (track_id -> list of (frame_idx, cx, cy, bbox_h))
    track_history: dict[int, list[tuple[int, float, float, float]]] = field(
        default_factory=dict
    )
    track_area: dict[int, float] = field(default_factory=dict)


@dataclass(frozen=True)
class _TestRun:
    track_id: int
    start_frame: int
    stop_frame: int

    @property
    def duration_frames(self) -> int:
        return self.stop_frame - self.start_frame


# --- Pipeline ----------------------------------------------------------


class TTestTest(BaseTest):
    """T-Test: multi-track motion analysis -> total completion time."""

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
        last_focal_pose = None
        last_focal_id: int | None = None

        try:
            for frame in frame_iter(video_path):
                n_frames += 1
                img = frame.image

                tracked = self._tracker.update(img)
                people = [t for t in tracked if t.class_id == PERSON_CLASS_ID]

                # Record every track's center history & cumulative area
                for p in people:
                    state.track_history.setdefault(p.track_id, []).append(
                        (frame.idx, float(p.center[0]),
                         float(p.center[1]), p.height)
                    )
                    state.track_area[p.track_id] = (
                        state.track_area.get(p.track_id, 0.0)
                        + p.height * p.width
                    )

                # Live focal = cumulatively-largest track (only for the
                # in-flight HUD/skeleton; the actual test-doer is decided
                # post-loop based on longest motion run).
                focal = _live_focal(people, state)
                if focal is not None:
                    if focal.track_id != last_focal_id:
                        last_focal_pose = None
                        last_focal_id = focal.track_id
                    if frame.idx % _POSE_INTERVAL_FRAMES == 0:
                        last_focal_pose = self._pose.estimate_bbox(
                            img, focal.bbox_xyxy
                        )

                # Annotate every detected person; focal gets a brighter box
                for p in people:
                    color = (
                        ATHLETE_BBOX if focal is not None and p.track_id == focal.track_id
                        else _NON_FOCAL_BBOX_COLOR
                    )
                    draw_bbox(img, p.bbox_xyxy, color=color)
                if last_focal_pose is not None:
                    draw_skeleton(img, last_focal_pose.keypoints)
                draw_hud(img, _live_hud_fields(state, frame.idx, fps))
                writer.write(img)
        except Exception:
            writer.release()
            raise

        # Post-loop: pick the track with the longest sustained motion run.
        best = _find_test_run(
            state.track_history, fps, min_run_frames=_MIN_RUN_FRAMES
        )
        if best is None:
            writer.release()
            if not state.track_history:
                raise DetectionError("no people were detected in the video")
            raise ProtocolError(
                "no sustained motion segment found (>= 6 s) on any track — "
                "could not identify the test run"
            )

        time_s = best.duration_frames / fps
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


# --- Track-history analysis -------------------------------------------


def _find_test_run(
    track_history: dict[int, list[tuple[int, float, float, float]]],
    fps: float,
    min_run_frames: int,
) -> _TestRun | None:
    """Across all tracks, find the longest contiguous high-motion run.

    Returns the run as a `_TestRun` keyed by the winning track_id, or
    None if no track has a qualifying segment.
    """
    candidates: list[_TestRun] = []
    for track_id, history in track_history.items():
        if len(history) < _MIN_TRACK_HISTORY_FRAMES:
            continue
        run = _longest_motion_run(history)
        if run is None:
            continue
        start, stop = run
        if (stop - start) >= min_run_frames:
            candidates.append(_TestRun(track_id=track_id, start_frame=start, stop_frame=stop))
    if not candidates:
        return None
    return max(candidates, key=lambda r: r.duration_frames)


def _longest_motion_run(
    history: list[tuple[int, float, float, float]],
) -> tuple[int, int] | None:
    """Return (start_frame, stop_frame) of the longest contiguous stretch
    where smoothed per-frame motion exceeds the bbox-relative threshold.
    """
    if len(history) < _MOTION_SMOOTH_WINDOW_FRAMES + 2:
        return None
    frame_idxs = [h[0] for h in history]
    centers = np.array([(h[1], h[2]) for h in history], dtype=float)
    heights = np.array([h[3] for h in history], dtype=float)

    diffs = np.diff(centers, axis=0)
    motion = np.linalg.norm(diffs, axis=1)  # length N-1

    window = _MOTION_SMOOTH_WINDOW_FRAMES
    kernel = np.ones(window, dtype=float) / window
    smoothed = np.convolve(motion, kernel, mode="same")

    threshold = _MOTION_THRESHOLD_FRAC * float(np.mean(heights))
    above = smoothed > threshold

    # Merge above-threshold segments separated by short sub-threshold
    # gaps (treat brief decelerations as part of the same run).
    merged = above.copy()
    i = 0
    while i < len(merged):
        if not merged[i]:
            j = i
            while j < len(merged) and not merged[j]:
                j += 1
            gap_len = j - i
            # Fill the gap only if it's bounded by True on both sides
            # (i.e., not at the start or end of the trace).
            if i > 0 and j < len(merged) and gap_len < _GAP_MERGE_FRAMES:
                merged[i:j] = True
            i = j
        else:
            i += 1

    best_start_i = -1
    best_len = 0
    cur_start = -1
    for i, a in enumerate(merged):
        if a:
            if cur_start < 0:
                cur_start = i
        else:
            if cur_start >= 0:
                cur_len = i - cur_start
                if cur_len > best_len:
                    best_start_i = cur_start
                    best_len = cur_len
                cur_start = -1
    if cur_start >= 0:
        cur_len = len(merged) - cur_start
        if cur_len > best_len:
            best_start_i = cur_start
            best_len = cur_len

    if best_start_i < 0 or best_len < 1:
        return None
    # `motion` is offset by 1 from `frame_idxs` (it's the diff between
    # frame i and i+1). Map back conservatively.
    start_frame = frame_idxs[best_start_i]
    stop_frame = frame_idxs[min(best_start_i + best_len, len(frame_idxs) - 1)]
    return (start_frame, stop_frame)


# --- Live (in-loop) helpers -------------------------------------------


def _live_focal(
    people: list[TrackedDetection], state: _RunState
) -> TrackedDetection | None:
    """Cumulatively-largest track present in the current frame.

    Used only for live HUD / skeleton — the final result is decided
    post-loop based on the longest motion run across all tracks.
    """
    if not people:
        return None
    if state.track_area:
        dominant_id = max(state.track_area, key=state.track_area.get)
        for t in people:
            if t.track_id == dominant_id:
                return t
    return max(people, key=lambda t: t.height * t.width)


def _live_hud_fields(
    state: _RunState, frame_idx: int, fps: float
) -> dict[str, str]:
    return {
        "phase": "scanning",
        "tracks": str(len(state.track_history)),
        "elapsed": f"{(frame_idx + 1) / fps:.1f} s",
    }
