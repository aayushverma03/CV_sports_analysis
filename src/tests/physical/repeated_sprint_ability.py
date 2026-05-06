"""Repeated Sprint Ability (RSA) pipeline.

Series of maximal sprints with short rests. Default protocol: 6 × 30 m
with 20 s rest between sprints. Operator can override N and distance
via constructor.

Same multi-burst architecture as `bangsbo_sprint.py` — find all
above-threshold motion bursts on the picked athlete's stabilized
trajectory, keep the top N by duration, time each as one sprint.

Scored on Glaister sprint-decrement plus sprint_best / sprint_mean.
RSA differs from Bangsbo only in protocol parameters (N, distance,
rest); the detection logic is identical.
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

# --- Defaults ----------------------------------------------------------

_DEFAULT_N_SPRINTS = 6
_DEFAULT_SPRINT_DISTANCE_M = 30.0

_MOTION_SMOOTH_FRAMES = 60
_MOTION_THRESHOLD_FRAC = 0.03
_TELEPORT_FRAC = 0.5
_MIN_SPRINT_FRAMES = 30           # 1 s @ 30 fps; below is implausible

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


class RepeatedSprintAbilityTest(BaseTest):
    """Repeated Sprint Ability: N × distance, multi-burst detection."""

    test_id = "repeated-sprint-ability"

    def __init__(
        self,
        *,
        n_sprints: int = _DEFAULT_N_SPRINTS,
        sprint_distance_m: float = _DEFAULT_SPRINT_DISTANCE_M,
        assumed_athlete_height_m: float = _DEFAULT_ATHLETE_HEIGHT_M,
    ) -> None:
        if n_sprints < 2:
            raise ValueError(f"n_sprints must be >= 2, got {n_sprints}")
        if sprint_distance_m <= 0:
            raise ValueError(
                f"sprint_distance_m must be > 0, got {sprint_distance_m}"
            )
        self._n_sprints = n_sprints
        self._sprint_distance_m = float(sprint_distance_m)
        self._total_distance_m = self._n_sprints * self._sprint_distance_m
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
        print(f"[rsa] {len(cone_positions)} cone clusters detected")

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
            cone_spacing_m=self._sprint_distance_m,
            baseline_bbox_h_px=baseline_bbox_h,
            assumed_height_m=self._assumed_height_m,
        )
        print(f"[rsa] calibration: {px_per_m:.1f} px/m ({calibration_source})")

        # === Burst detection ===
        bursts = _detect_bursts(
            history=track_history[player_track_id],
            n_expected=self._n_sprints,
        )
        if len(bursts) < self._n_sprints:
            raise ProtocolError(
                f"detected only {len(bursts)} sprints; expected "
                f"{self._n_sprints}. Check that the athlete completes the "
                "full protocol with clear rest periods between sprints."
            )
        run_window = _RunWindow(
            track_id=player_track_id,
            bursts=tuple(bursts[: self._n_sprints]),
        )

        rep_times_s = tuple(b.duration_frames / fps for b in run_window.bursts)
        sprint_best = float(min(rep_times_s))
        sprint_worst = float(max(rep_times_s))
        sprint_mean = float(np.mean(rep_times_s))
        pct_decrement = (sprint_mean / sprint_best - 1.0) * 100.0
        total_time = float(sum(rep_times_s))

        metrics: dict[str, MetricValue] = {
            "sprint_best_s": MetricValue(raw=sprint_best, unit="s"),
            "sprint_mean_s": MetricValue(raw=sprint_mean, unit="s"),
            "pct_sprint_decrement": MetricValue(raw=pct_decrement, unit="pct"),
            "sprint_worst_s": MetricValue(raw=sprint_worst, unit="s"),
            "total_completion_time_s": MetricValue(raw=total_time, unit="s"),
            "total_distance_m": MetricValue(raw=self._total_distance_m, unit="m"),
            "average_speed_ms": MetricValue(
                raw=self._total_distance_m / total_time if total_time > 0 else 0.0,
                unit="m_per_s",
            ),
            **{
                f"rep{i+1}_time_s": MetricValue(raw=float(rep_times_s[i]), unit="s")
                for i in range(self._n_sprints)
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
                    rep_times_s=rep_times_s, n_sprints=self._n_sprints,
                ))
                writer.write(img)

            endcard_rows = [
                (mid.replace("_", " "), f"{mv.raw:.3f} {mv.unit}",
                 int(round(scores[mid].score)) if mid in scores else 0)
                for mid, mv in metrics.items()
            ]
            endcard = render_endcard(
                title=f"Repeated Sprint Ability ({self._n_sprints} × {self._sprint_distance_m:.0f} m)",
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
    n_expected: int,
) -> list[_Burst]:
    """Same multi-burst detector as Bangsbo: collect all above-threshold
    smoothed-motion runs (>= _MIN_SPRINT_FRAMES), keep the top
    n_expected by duration, re-sort chronologically.
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

    bursts.sort(key=lambda b: -b.duration_frames)
    bursts = bursts[:n_expected]
    bursts.sort(key=lambda b: b.start_frame)
    return bursts


# --- HUD --------------------------------------------------------------


def _current_sprint(
    frame_idx: int, run: _RunWindow,
) -> tuple[int, _Burst | None]:
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
    n_sprints: int,
) -> dict[str, str]:
    sprint_num, current = _current_sprint(frame_idx, run)
    if sprint_num == 0:
        return {
            "phase": "ready",
            "sprint": f"0/{n_sprints}",
            "time": "-",
            "best": "-",
            "fatigue": "-",
        }
    if sprint_num <= n_sprints and current is not None:
        elapsed_s = (frame_idx - current.start_frame) / fps
        completed = rep_times_s[: sprint_num - 1]
        if completed:
            best_so_far = min(completed)
            mean_so_far = sum(completed) / len(completed)
            fatigue = (mean_so_far / best_so_far - 1.0) * 100.0
            return {
                "phase": "running",
                "sprint": f"{sprint_num}/{n_sprints}",
                "time": f"{elapsed_s:.2f} s",
                "best": f"{best_so_far:.2f} s",
                "fatigue": f"{fatigue:.1f}%",
            }
        return {
            "phase": "running",
            "sprint": f"{sprint_num}/{n_sprints}",
            "time": f"{elapsed_s:.2f} s",
            "best": "-",
            "fatigue": "-",
        }
    total = sum(rep_times_s)
    best = min(rep_times_s)
    mean = sum(rep_times_s) / len(rep_times_s)
    final_fatigue = (mean / best - 1.0) * 100.0
    return {
        "phase": "finished",
        "sprint": f"{n_sprints}/{n_sprints}",
        "time": f"{total:.3f} s",
        "best": f"{best:.2f} s",
        "fatigue": f"{final_fatigue:.1f}%",
    }
