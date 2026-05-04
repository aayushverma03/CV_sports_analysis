"""T-Test (Agility) pipeline.

T-shape course: athlete sprints forward to centre cone, side-shuffles
left, side-shuffles across right, side-shuffles back to centre, then
backpedals to start. Total path A->B->C->B->D->B->A.

v1 ships only the scored metric `total_completion_time_s`. Cone
detection and segment_completion_times are spec-mandated for v1.x.

Multi-person handling: the user's recordings often contain a coach
plus the test player in the same frame. The pipeline runs in two
passes (deliberate exception to hard rule #3 — multi-person agility
needs post-hoc track selection):

- Pass 1: detect + track every person across the video, sample cone
  detections at stride. No pose, no annotation.
- Post-loop: pick the player track via shared player_picker (area
  dominance → cone proximity + longest sustained motion fallback),
  then find the run window on that track only. Run-window detection
  uses teleport-aware breaks so a track contaminated by ID swaps
  can't read as one continuous run; instead the run-window finder
  returns None and the pipeline raises ProtocolError.
- Pass 2: re-iterate the video and render annotations for ONLY the
  chosen player track (bbox + skeleton + run-relative HUD). Pose runs
  only on the player's bbox, so we save inference vs running pose on
  every track in pass 1.

Known limitation: in close-proximity multi-attempt videos (e.g. coach +
two athletes, all crossing within a few bbox-widths of each other)
ByteTrack produces tracks contaminated by ID swaps. The teleport-aware
run-window finder will reject these and surface a ProtocolError rather
than silently report a duration spanning multiple attempts.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from src.core.annotation.overlays import (
    draw_bbox,
    draw_hud,
    draw_skeleton,
    render_endcard,
)
from src.core.detection.marker_detector import MarkerDetector
from src.core.detection.player_detector import PERSON_CLASS_ID
from src.core.pose.estimator import create_pose_estimator
from src.core.tracking.bytetrack_tracker import ByteTrackTracker
from src.core.tracking.player_picker import pick_player
from src.core.tracking.run_window import (
    cluster_object_positions,
    find_run_on_track,
)
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

# T-Test elite male is 8.8 s; below 6 s is implausible.
_MIN_RUN_FRAMES = 180

# Background tracks too small to consider.
_MIN_TRACK_HISTORY_FRAMES = 60
_POSE_INTERVAL_FRAMES = 3
_ENDCARD_HOLD_S = 2.5

# Cone detection: sample frames at this stride during pass 1 (cones
# are stationary, so we only need a handful of detections to lock
# their positions for the proximity fallback).
_CONE_SAMPLE_STRIDE = 60
_CONE_CLUSTER_RADIUS_PX = 40.0
_CONE_MIN_DETECTIONS = 3


# --- State -------------------------------------------------------------


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
    """T-Test: 2-pass multi-track motion analysis -> total completion time."""

    test_id = "t-test"

    def __init__(self) -> None:
        self._tracker = ByteTrackTracker(
            classes=[PERSON_CLASS_ID],
            confidence=0.20,
        )
        self._pose = create_pose_estimator("pose_default")
        # Lazy-init: only created when needed (cone-fallback in step 2)
        self._marker: MarkerDetector | None = None

    def run(
        self,
        video_path: Path,
        athlete: AthleteProfile,
        output_dir: Path,
    ) -> AnalysisResult:
        info = video_info(video_path)
        fps = info.fps
        out_path = output_dir / f"{self.test_id}.mp4"

        # === PASS 1: detect + track all persons; sample cone detections ===
        # Track history shape (frame_idx, cx, cy, bbox_h, bbox_w) — width
        # is needed by player_picker.pick_by_area_dominance.
        track_history: dict[int, list[tuple[int, float, float, float, float]]] = {}
        track_bboxes: dict[int, dict[int, np.ndarray]] = {}
        cone_detections: list[tuple[float, float]] = []
        n_frames = 0

        for frame in frame_iter(video_path):
            n_frames += 1
            tracked = self._tracker.update(frame.image)
            for p in tracked:
                if p.class_id != PERSON_CLASS_ID:
                    continue
                track_history.setdefault(p.track_id, []).append(
                    (frame.idx, float(p.center[0]), float(p.center[1]),
                     p.height, p.width)
                )
                track_bboxes.setdefault(p.track_id, {})[frame.idx] = (
                    p.bbox_xyxy.copy()
                )
            # Sample cone detections at stride for the proximity fallback
            if frame.idx % _CONE_SAMPLE_STRIDE == 0:
                if self._marker is None:
                    self._marker = MarkerDetector()
                for det in self._marker.detect(frame.image):
                    cone_detections.append(
                        (float(det.center[0]), float(det.center[1]))
                    )

        # === Pick THE player track (area dominance, then cone proximity) ===
        cone_positions = cluster_object_positions(
            cone_detections,
            radius_px=_CONE_CLUSTER_RADIUS_PX,
            min_count=_CONE_MIN_DETECTIONS,
        )
        # Cones are stationary — same set for every frame in the
        # proximity-fallback step.
        object_positions = (
            {fi: cone_positions for fi in range(n_frames)}
            if cone_positions else None
        )
        print(f"[t_test] {len(cone_positions)} cone clusters detected")
        player_track_id = pick_player(
            track_history,
            object_positions=object_positions,
            min_history_frames=_MIN_TRACK_HISTORY_FRAMES,
            verbose=True,
        )
        if player_track_id is None:
            if not track_history:
                raise DetectionError("no people were detected in the video")
            raise ProtocolError(
                "could not identify a single player track — "
                "neither pixel-area dominance nor cone-proximity fallback "
                "yielded a winner"
            )

        # Find the run window on the chosen player's track only.
        run_window = find_run_on_track(
            track_history[player_track_id],
            min_run_frames=_MIN_RUN_FRAMES,
        )
        if run_window is None:
            raise ProtocolError(
                f"player track {player_track_id} found but no sustained "
                f"motion segment >= {_MIN_RUN_FRAMES / fps:.0f} s — "
                "could not time the run"
            )
        best = _TestRun(
            track_id=player_track_id,
            start_frame=run_window[0],
            stop_frame=run_window[1],
        )

        # === PASS 2: re-iterate video, annotate only the chosen player ===
        writer = cv2.VideoWriter(
            str(out_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (info.width, info.height),
        )
        if not writer.isOpened():
            raise ProtocolError(f"could not open VideoWriter at {out_path}")

        player_bboxes_by_frame = track_bboxes.get(best.track_id, {})
        last_pose = None

        try:
            for frame in frame_iter(video_path):
                img = frame.image
                player_bbox = player_bboxes_by_frame.get(frame.idx)

                if player_bbox is not None:
                    if frame.idx % _POSE_INTERVAL_FRAMES == 0:
                        last_pose = self._pose.estimate_bbox(img, player_bbox)
                    draw_bbox(img, player_bbox)
                    if last_pose is not None:
                        draw_skeleton(img, last_pose.keypoints)

                draw_hud(img, _player_hud_fields(frame.idx, fps, best))
                writer.write(img)

            # End-card
            time_s = best.duration_frames / fps
            metrics = {
                "total_completion_time_s": MetricValue(raw=time_s, unit="s"),
            }
            scores, test_score = score_test(
                metrics, self.test_id, athlete.gender
            )
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
        finally:
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


# --- HUD --------------------------------------------------------------


def _player_hud_fields(frame_idx: int, fps: float, run: _TestRun) -> dict[str, str]:
    """HUD scoped to the chosen player's run.

    Pre-start: phase=ready. During run: phase=running, time=elapsed. Post-run: phase=finished, time=total.
    """
    if frame_idx < run.start_frame:
        return {"phase": "ready", "time": "-"}
    if frame_idx <= run.stop_frame:
        elapsed_s = (frame_idx - run.start_frame) / fps
        return {"phase": "running", "time": f"{elapsed_s:.2f} s"}
    total_s = run.duration_frames / fps
    return {"phase": "finished", "time": f"{total_s:.3f} s"}
