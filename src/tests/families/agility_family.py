"""Shared 2-pass scaffolding for agility tests.

Tests in this family share an identical pipeline shape:

1. Pass 1 — detect + ByteTrack every person; sample cone detections at
   stride (no pose, no annotation).
2. Pick the player track via shared `player_picker` (area dominance,
   then cone-proximity fallback).
3. Find the run window on that track via `find_run_on_track`.
4. Pass 2 — re-iterate the video and render bbox + skeleton + HUD only
   for the chosen player track. Pose runs only on the player's bbox.

T-Test and Illinois Agility differ only in the values of a few class-
level constants (test_id, end-card title, min_run_frames, teleport
threshold). Per-test metric formulas and HUD fields are exposed as
overridable methods.

5x10m Sprint with COD does NOT inherit here — it adds camera-motion
compensation and a 3rd pose-collection pass for ankle-based per-rep
detection. It uses the same shared primitives directly.
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

# --- Defaults ----------------------------------------------------------

_DEFAULT_MIN_TRACK_HISTORY_FRAMES = 60
_DEFAULT_POSE_INTERVAL_FRAMES = 3
_DEFAULT_ENDCARD_HOLD_S = 2.5
_DEFAULT_CONE_SAMPLE_STRIDE = 60
_DEFAULT_CONE_CLUSTER_RADIUS_PX = 40.0
_DEFAULT_CONE_MIN_DETECTIONS = 3


# --- State -------------------------------------------------------------


@dataclass(frozen=True)
class AgilityRun:
    track_id: int
    start_frame: int
    stop_frame: int

    @property
    def duration_frames(self) -> int:
        return self.stop_frame - self.start_frame


# --- Base --------------------------------------------------------------


class AgilityFamilyTest(BaseTest):
    """Common 2-pass agility pipeline. Subclass and set class-level
    constants + override `_compute_metrics` if more than the default
    `total_completion_time_s` should be reported."""

    # --- Subclass-overridable constants ---
    test_id: str = ""
    endcard_title: str = ""
    min_run_frames: int = 0     # MUST be set by subclass
    teleport_frac: float = 0.5  # 5.0 effectively disables teleport defence

    # Optional per-test overrides for marker detection
    marker_prompts: tuple[str, ...] | None = None
    marker_confidence: float | None = None

    # Class constants (rarely overridden)
    min_track_history_frames: int = _DEFAULT_MIN_TRACK_HISTORY_FRAMES
    pose_interval_frames: int = _DEFAULT_POSE_INTERVAL_FRAMES
    endcard_hold_s: float = _DEFAULT_ENDCARD_HOLD_S
    cone_sample_stride: int = _DEFAULT_CONE_SAMPLE_STRIDE
    cone_cluster_radius_px: float = _DEFAULT_CONE_CLUSTER_RADIUS_PX
    cone_min_detections: int = _DEFAULT_CONE_MIN_DETECTIONS

    def __init__(self) -> None:
        if not self.test_id or self.min_run_frames <= 0:
            raise NotImplementedError(
                "subclass must set `test_id` and `min_run_frames`"
            )
        self._tracker = ByteTrackTracker(
            classes=[PERSON_CLASS_ID],
            confidence=0.20,
        )
        self._pose = create_pose_estimator("pose_default")
        self._marker: MarkerDetector | None = None

    # --- Subclass hooks ---

    def _compute_metrics(
        self, run: AgilityRun, fps: float,
    ) -> dict[str, MetricValue]:
        """Default: only `total_completion_time_s`. Override to add more."""
        time_s = run.duration_frames / fps
        return {
            "total_completion_time_s": MetricValue(raw=time_s, unit="s"),
        }

    def _hud_fields(
        self, frame_idx: int, fps: float, run: AgilityRun,
    ) -> dict[str, str]:
        """Default HUD: phase + elapsed/total time. Override for richer HUD."""
        if frame_idx < run.start_frame:
            return {"phase": "ready", "time": "-"}
        if frame_idx <= run.stop_frame:
            elapsed_s = (frame_idx - run.start_frame) / fps
            return {"phase": "running", "time": f"{elapsed_s:.2f} s"}
        total_s = run.duration_frames / fps
        return {"phase": "finished", "time": f"{total_s:.3f} s"}

    # --- Pipeline ---

    def run(
        self,
        video_path: Path,
        athlete: AthleteProfile,
        output_dir: Path,
    ) -> AnalysisResult:
        info = video_info(video_path)
        fps = info.fps
        out_path = output_dir / f"{self.test_id}.mp4"

        track_history, track_bboxes, cone_detections, n_frames = (
            self._pass1_detect(video_path)
        )

        cone_positions = cluster_object_positions(
            cone_detections,
            radius_px=self.cone_cluster_radius_px,
            min_count=self.cone_min_detections,
        )
        print(f"[{self.test_id}] {len(cone_positions)} cone clusters detected")

        player_track_id = pick_player(
            track_history,
            object_positions=(
                {fi: cone_positions for fi in range(n_frames)}
                if cone_positions else None
            ),
            min_history_frames=self.min_track_history_frames,
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

        run_window = find_run_on_track(
            track_history[player_track_id],
            min_run_frames=self.min_run_frames,
            teleport_frac=self.teleport_frac,
        )
        if run_window is None:
            raise ProtocolError(
                f"player track {player_track_id} found but no sustained "
                f"motion segment >= {self.min_run_frames / fps:.0f} s — "
                "could not time the run"
            )
        run = AgilityRun(
            track_id=player_track_id,
            start_frame=run_window[0],
            stop_frame=run_window[1],
        )
        metrics = self._compute_metrics(run, fps)
        scores, test_score = score_test(metrics, self.test_id, athlete.gender)

        self._pass2_render(
            video_path=video_path,
            out_path=out_path,
            info=info,
            fps=fps,
            run=run,
            track_bboxes=track_bboxes,
            athlete=athlete,
            metrics=metrics,
            scores=scores,
            test_score=test_score,
        )

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

    # --- Internals ---

    def _pass1_detect(
        self, video_path: Path,
    ) -> tuple[
        dict[int, list[tuple[int, float, float, float, float]]],
        dict[int, dict[int, np.ndarray]],
        list[tuple[float, float]],
        int,
    ]:
        track_history: dict[
            int, list[tuple[int, float, float, float, float]]
        ] = {}
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
            if frame.idx % self.cone_sample_stride == 0:
                if self._marker is None:
                    kwargs: dict = {}
                    if self.marker_prompts is not None:
                        kwargs["prompts"] = list(self.marker_prompts)
                    if self.marker_confidence is not None:
                        kwargs["confidence"] = self.marker_confidence
                    self._marker = MarkerDetector(**kwargs)
                for det in self._marker.detect(frame.image):
                    cone_detections.append(
                        (float(det.center[0]), float(det.center[1]))
                    )
        return track_history, track_bboxes, cone_detections, n_frames

    def _pass2_render(
        self,
        *,
        video_path: Path,
        out_path: Path,
        info,
        fps: float,
        run: AgilityRun,
        track_bboxes: dict[int, dict[int, np.ndarray]],
        athlete: AthleteProfile,
        metrics: dict[str, MetricValue],
        scores,
        test_score,
    ) -> None:
        writer = cv2.VideoWriter(
            str(out_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (info.width, info.height),
        )
        if not writer.isOpened():
            raise ProtocolError(f"could not open VideoWriter at {out_path}")

        player_bboxes_by_frame = track_bboxes.get(run.track_id, {})
        last_pose = None
        try:
            for frame in frame_iter(video_path):
                img = frame.image
                player_bbox = player_bboxes_by_frame.get(frame.idx)
                if player_bbox is not None:
                    if frame.idx % self.pose_interval_frames == 0:
                        last_pose = self._pose.estimate_bbox(img, player_bbox)
                    draw_bbox(img, player_bbox)
                    if last_pose is not None:
                        draw_skeleton(img, last_pose.keypoints)
                draw_hud(img, self._hud_fields(frame.idx, fps, run))
                writer.write(img)

            endcard_rows = [
                (mid.replace("_", " "), f"{mv.raw:.3f} {mv.unit}",
                 int(round(scores[mid].score)) if mid in scores else 0)
                for mid, mv in metrics.items()
            ]
            endcard = render_endcard(
                title=self.endcard_title or self.test_id,
                athlete=f"{athlete.gender} age {athlete.age}",
                metric_rows=endcard_rows,
                test_score=int(round(test_score.score)),
                band=format_band(test_score.band),
                size=(info.width, info.height),
            )
            for _ in range(int(fps * self.endcard_hold_s)):
                writer.write(endcard)
        finally:
            writer.release()
