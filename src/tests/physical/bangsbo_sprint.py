"""Bangsbo Sprint Test (7 × 34.2 m) pipeline.

Repeated-sprint protocol: athlete completes 7 sprints of 34.2 m with
25 s rest between sprints. Each sprint timed independently. Scored on
total_completion_time_s (sum of all 7 sprint times) and
pct_sprint_decrement (Glaister formula — how much the athlete slows
down across the 7 reps).

The 25-second rest gap between sprints makes burst-detection very
clean: each sprint is a contiguous motion segment above threshold,
separated by long sub-threshold rest periods. We reuse the teleport-
aware smoothed-motion machinery from `run_window.py` but instead of
finding the SINGLE longest run we collect ALL above-threshold runs
that meet a minimum-duration filter, treating each as one sprint.

Pipeline (3-pass), reuses the 5x10m / yo-yo scaffolding:
- Pass 1: ByteTrack person, sample cones, estimate camera transform.
  Stabilize positions to frame-0 coords.
- Pick player, calibrate (cone-pair / 34.2 m or body-height proxy).
- Pass 2: find 7 motion bursts on the picked athlete's trajectory;
  derive rep times.
- Pass 3: render annotated video with HUD ticker.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from src.core.annotation.overlays import (
    draw_bbox,
    draw_hud,
    render_endcard,
)
from src.core.calibration.camera_motion import CameraMotion
from src.core.detection.marker_detector import CustomMarkerDetector
from src.core.detection.player_detector import PERSON_CLASS_ID
from src.core.tracking.bytetrack_tracker import ByteTrackTracker
from src.core.tracking.player_picker import pick_player
from src.core.tracking.run_window import cluster_object_positions
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

_N_SPRINTS = 7
_SPRINT_DISTANCE_M = 34.2
_TOTAL_DISTANCE_M = _N_SPRINTS * _SPRINT_DISTANCE_M

# Burst detection — same shape as run_window's longest_motion_run, but
# we collect ALL bursts >= the minimum duration filter rather than
# picking the longest.
_MOTION_SMOOTH_FRAMES = 60
_MOTION_THRESHOLD_FRAC = 0.03   # 3% of mean bbox-h after smoothing
_TELEPORT_FRAC = 0.5            # caps single-frame jumps from ID swaps
# Minimum frames per sprint: even a fast 34.2 m at 12 m/s = 2.85 s = ~85
# frames at 30 fps; below 1.5 s implausible.
_MIN_SPRINT_FRAMES = 45
# Minimum frames between sprints: rest is 25 s but we accept anything
# above 5 s (150 frames) as a clear separator.
_MIN_REST_FRAMES = 150

_MIN_TRACK_HISTORY_FRAMES = 60
_ENDCARD_HOLD_S = 2.5

_MARKER_MODEL_KEYS = (
    "detector_yellow_pole_v1",
    "detector_green_dome_v1",
)
_CONE_SAMPLE_STRIDE = 30
_CONE_CLUSTER_RADIUS_PX = 60.0
_CONE_MIN_DETECTIONS = 3

_DEFAULT_ATHLETE_HEIGHT_M = 1.70


# --- State -------------------------------------------------------------


@dataclass(frozen=True)
class _Burst:
    start_frame: int
    stop_frame: int

    @property
    def duration_frames(self) -> int:
        return self.stop_frame - self.start_frame


@dataclass(frozen=True)
class _RunWindow:
    track_id: int
    bursts: tuple[_Burst, ...]

    @property
    def start_frame(self) -> int:
        return self.bursts[0].start_frame if self.bursts else 0

    @property
    def stop_frame(self) -> int:
        return self.bursts[-1].stop_frame if self.bursts else 0


# --- Pipeline ----------------------------------------------------------


class BangsboSprintTest(BaseTest):
    """Bangsbo Sprint: 7 × 34.2 m, multi-burst trajectory detection."""

    test_id = "bangsbo-sprint"

    def __init__(
        self,
        *,
        assumed_athlete_height_m: float = _DEFAULT_ATHLETE_HEIGHT_M,
    ) -> None:
        self._assumed_height_m = float(assumed_athlete_height_m)
        self._tracker = ByteTrackTracker(
            classes=[PERSON_CLASS_ID],
            confidence=0.20,
        )
        self._marker: CustomMarkerDetector | None = None

    def run(
        self,
        video_path: Path,
        athlete: AthleteProfile,
        output_dir: Path,
    ) -> AnalysisResult:
        info = video_info(video_path)
        fps = info.fps
        out_path = output_dir / f"{self.test_id}.mp4"

        # === PASS 1 ===
        track_history_raw: dict[
            int, list[tuple[int, float, float, float, float]]
        ] = {}
        track_bboxes_raw: dict[int, dict[int, np.ndarray]] = {}
        cone_detections_per_frame: list[tuple[int, float, float]] = []
        n_frames = 0
        motion = CameraMotion()

        for frame in frame_iter(video_path):
            n_frames += 1
            tracked = self._tracker.update(frame.image)
            person_bboxes_this_frame: list[np.ndarray] = []
            for p in tracked:
                if p.class_id != PERSON_CLASS_ID:
                    continue
                person_bboxes_this_frame.append(p.bbox_xyxy)
                track_history_raw.setdefault(p.track_id, []).append(
                    (frame.idx, float(p.center[0]), float(p.center[1]),
                     p.height, p.width)
                )
                track_bboxes_raw.setdefault(p.track_id, {})[frame.idx] = (
                    p.bbox_xyxy.copy()
                )
            if frame.idx % _CONE_SAMPLE_STRIDE == 0:
                if self._marker is None:
                    self._marker = CustomMarkerDetector(
                        model_keys=list(_MARKER_MODEL_KEYS),
                    )
                for det in self._marker.detect(frame.image):
                    cone_detections_per_frame.append((
                        frame.idx,
                        float(det.center[0]),
                        float(det.center[1]),
                    ))
            motion.update(
                frame.idx, frame.image,
                exclude_bboxes_xyxy=person_bboxes_this_frame,
            )

        # === Stabilize ===
        track_history: dict[
            int, list[tuple[int, float, float, float, float]]
        ] = {}
        for tid, hist in track_history_raw.items():
            stabilized = []
            for fi, cx, cy, h, w in hist:
                sx, sy = motion.transform_point(fi, (cx, cy))
                stabilized.append((fi, sx, sy, h, w))
            track_history[tid] = stabilized

        cone_detections_stab: list[tuple[float, float]] = [
            motion.transform_point(fi, (cx, cy))
            for (fi, cx, cy) in cone_detections_per_frame
        ]
        cone_positions = cluster_object_positions(
            cone_detections_stab,
            radius_px=_CONE_CLUSTER_RADIUS_PX,
            min_count=_CONE_MIN_DETECTIONS,
        )
        print(f"[bangsbo] {len(cone_positions)} cone clusters detected")

        # === Pick player ===
        object_positions = (
            {fi: cone_positions for fi in range(n_frames)}
            if cone_positions else None
        )
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
                "could not identify a single player track"
            )

        # === Calibration ===
        baseline_bbox_h = float(np.median(
            [h for (_, _, _, h, _) in track_history[player_track_id]]
        ))
        cone_pair = _pick_two_cones(cone_positions)
        px_per_m, calibration_source = _calibrate(
            cone_pair=cone_pair,
            cone_spacing_m=_SPRINT_DISTANCE_M,
            baseline_bbox_h_px=baseline_bbox_h,
            assumed_height_m=self._assumed_height_m,
        )
        print(f"[bangsbo] calibration: {px_per_m:.1f} px/m ({calibration_source})")

        # === Burst detection -> 7 sprints ===
        bursts = _detect_bursts(
            history=track_history[player_track_id],
            fps=fps,
            n_expected=_N_SPRINTS,
        )
        if len(bursts) < _N_SPRINTS:
            raise ProtocolError(
                f"detected only {len(bursts)} sprints; expected {_N_SPRINTS}. "
                "Check that the athlete completes the full 7 × 34.2 m "
                "protocol with clear rest periods between sprints."
            )
        run_window = _RunWindow(
            track_id=player_track_id,
            bursts=tuple(bursts[:_N_SPRINTS]),
        )

        rep_times_s = tuple(b.duration_frames / fps for b in run_window.bursts)
        sprint_best = float(min(rep_times_s))
        sprint_worst = float(max(rep_times_s))
        sprint_mean = float(np.mean(rep_times_s))
        # Glaister sprint-decrement: ((mean / best) - 1) × 100
        pct_decrement = (sprint_mean / sprint_best - 1.0) * 100.0
        total_time = float(sum(rep_times_s))

        metrics: dict[str, MetricValue] = {
            "total_completion_time_s": MetricValue(raw=total_time, unit="s"),
            "pct_sprint_decrement": MetricValue(raw=pct_decrement, unit="pct"),
            "sprint_best_s": MetricValue(raw=sprint_best, unit="s"),
            "sprint_worst_s": MetricValue(raw=sprint_worst, unit="s"),
            "sprint_mean_s": MetricValue(raw=sprint_mean, unit="s"),
            "total_distance_m": MetricValue(raw=_TOTAL_DISTANCE_M, unit="m"),
            "average_speed_ms": MetricValue(
                raw=_TOTAL_DISTANCE_M / total_time if total_time > 0 else 0.0,
                unit="m_per_s",
            ),
            **{
                f"rep{i+1}_time_s": MetricValue(raw=float(rep_times_s[i]), unit="s")
                for i in range(_N_SPRINTS)
            },
        }
        scores, test_score = score_test(metrics, self.test_id, athlete.gender)

        # === PASS 3: render ===
        writer = cv2.VideoWriter(
            str(out_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (info.width, info.height),
        )
        if not writer.isOpened():
            raise ProtocolError(f"could not open VideoWriter at {out_path}")

        player_bboxes_by_frame = track_bboxes_raw.get(player_track_id, {})
        try:
            for frame in frame_iter(video_path):
                img = frame.image
                bbox = player_bboxes_by_frame.get(frame.idx)
                if bbox is not None:
                    draw_bbox(img, bbox)
                draw_hud(img, _hud_fields(
                    frame_idx=frame.idx, fps=fps, run=run_window,
                    rep_times_s=rep_times_s,
                ))
                writer.write(img)

            endcard_rows = [
                (mid.replace("_", " "), f"{mv.raw:.3f} {mv.unit}",
                 int(round(scores[mid].score)) if mid in scores else 0)
                for mid, mv in metrics.items()
            ]
            endcard = render_endcard(
                title="Bangsbo Sprint (7 × 34.2 m)",
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
                fps_input=fps, duration_s=n_frames / fps if fps > 0 else 0.0,
            ),
        )


# --- Cone helpers -----------------------------------------------------


def _pick_two_cones(
    clusters: list[tuple[float, float]],
) -> tuple[tuple[float, float], tuple[float, float]] | None:
    if len(clusters) < 2:
        return None
    s = sorted(clusters, key=lambda c: c[0])
    return s[0], s[-1]


def _calibrate(
    *,
    cone_pair: tuple[tuple[float, float], tuple[float, float]] | None,
    cone_spacing_m: float,
    baseline_bbox_h_px: float,
    assumed_height_m: float,
) -> tuple[float, str]:
    if cone_pair is not None:
        a, b = cone_pair
        d = float(np.hypot(b[0] - a[0], b[1] - a[1]))
        if d > 0:
            return d / cone_spacing_m, "cone-pair"
    if baseline_bbox_h_px > 0:
        return baseline_bbox_h_px / assumed_height_m, "body-height-proxy"
    return 1.0, "no-calibration"


# --- Burst detection --------------------------------------------------


def _detect_bursts(
    *,
    history: list[tuple[int, float, float, float, float]],
    fps: float,
    n_expected: int,
) -> list[_Burst]:
    """Find all motion bursts (contiguous above-threshold smoothed
    motion) in the picked athlete's trajectory. Each burst = one
    sprint. Returns the top `n_expected` longest, sorted by start time.

    Mirrors `run_window.longest_motion_run` but collects EVERY burst
    instead of just the longest, since Bangsbo has 7 separate sprints
    with long rest periods between.
    """
    if len(history) < _MOTION_SMOOTH_FRAMES + 2:
        return []
    frame_idxs = [h[0] for h in history]
    centers = np.array([(h[1], h[2]) for h in history], dtype=float)
    heights = np.array([h[3] for h in history], dtype=float)

    diffs = np.diff(centers, axis=0)
    motion = np.linalg.norm(diffs, axis=1)
    mean_h = float(np.mean(heights))
    teleports = motion > _TELEPORT_FRAC * mean_h
    motion = motion.copy()
    motion[teleports] = 0.0

    kernel = np.ones(_MOTION_SMOOTH_FRAMES, dtype=float) / _MOTION_SMOOTH_FRAMES
    smoothed = np.convolve(motion, kernel, mode="same")
    above = smoothed > _MOTION_THRESHOLD_FRAC * mean_h

    # Collect contiguous True runs.
    bursts: list[_Burst] = []
    cur_start_i = -1
    for i, a in enumerate(above):
        if a:
            if cur_start_i < 0:
                cur_start_i = i
        else:
            if cur_start_i >= 0:
                if i - cur_start_i >= _MIN_SPRINT_FRAMES:
                    bursts.append(_Burst(
                        start_frame=int(frame_idxs[cur_start_i]),
                        stop_frame=int(frame_idxs[min(i, len(frame_idxs) - 1)]),
                    ))
                cur_start_i = -1
    if cur_start_i >= 0:
        run_len = len(above) - cur_start_i
        if run_len >= _MIN_SPRINT_FRAMES:
            bursts.append(_Burst(
                start_frame=int(frame_idxs[cur_start_i]),
                stop_frame=int(frame_idxs[-1]),
            ))

    # Take the top n_expected longest, then re-sort by start.
    bursts.sort(key=lambda b: -b.duration_frames)
    bursts = bursts[:n_expected]
    bursts.sort(key=lambda b: b.start_frame)
    return bursts


# --- HUD --------------------------------------------------------------


def _current_sprint(
    frame_idx: int, run: _RunWindow,
) -> tuple[int, _Burst | None]:
    """Returns (sprint_number, current_burst). Sprint number is 1-based;
    0 if before the first sprint, n_expected+1 after the last.
    """
    if not run.bursts:
        return 0, None
    if frame_idx < run.bursts[0].start_frame:
        return 0, None
    for i, b in enumerate(run.bursts, start=1):
        if frame_idx <= b.stop_frame:
            return i, b
    return len(run.bursts) + 1, None


def _hud_fields(
    *,
    frame_idx: int,
    fps: float,
    run: _RunWindow,
    rep_times_s: tuple[float, ...],
) -> dict[str, str]:
    sprint_num, current = _current_sprint(frame_idx, run)
    if sprint_num == 0:
        return {
            "phase": "ready",
            "sprint": f"0/{_N_SPRINTS}",
            "time": "-",
            "fatigue": "-",
        }
    if sprint_num <= _N_SPRINTS and current is not None:
        elapsed_s = (frame_idx - current.start_frame) / fps
        # Live fatigue index — mean of completed reps vs best so far.
        completed = rep_times_s[: sprint_num - 1]
        if completed:
            best_so_far = min(completed)
            mean_so_far = sum(completed) / len(completed)
            fatigue = (mean_so_far / best_so_far - 1.0) * 100.0
            fatigue_str = f"{fatigue:.1f}%"
        else:
            fatigue_str = "-"
        return {
            "phase": "running",
            "sprint": f"{sprint_num}/{_N_SPRINTS}",
            "time": f"{elapsed_s:.2f} s",
            "fatigue": fatigue_str,
        }
    # After last sprint: show summary.
    total = sum(rep_times_s)
    best = min(rep_times_s)
    mean = sum(rep_times_s) / len(rep_times_s)
    final_fatigue = (mean / best - 1.0) * 100.0
    return {
        "phase": "finished",
        "sprint": f"{_N_SPRINTS}/{_N_SPRINTS}",
        "time": f"{total:.3f} s",
        "fatigue": f"{final_fatigue:.1f}%",
    }
