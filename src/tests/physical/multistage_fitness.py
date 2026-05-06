"""Multistage Fitness Test (Bleep / Beep Test) pipeline.

Progressive 20 m shuttle test — athlete runs back and forth between two
cones in time with audio bleeps. The pace increases each level (~1 min
per level). Test ends when the athlete fails to reach the line on the
bleep twice consecutively.

Visual-only v1 (audio-beep alignment deferred). Same shape as
`yo_yo_intermittent.py` but:

  - 1 "shuttle" = one 20 m one-way trip (NOT an out-and-back), so
    shuttle count = number of trajectory direction reversals.
  - Continuous test (no recovery jog between shuttles), so a single
    sustained run window covers the whole effort.
  - Different level ladder (Léger 1988 progression) and VO2max formula.

Pipeline (3-pass) reuses the agility/sprint scaffolding: ByteTrack +
camera-motion + shared player_picker + cone-pair calibration. Final
speed level is looked up from the MSFT ladder using the cumulative
shuttle count.
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

# Each shuttle is one 20 m one-way trip.
_SHUTTLE_DISTANCE_M = 20.0

_MIN_RUN_FRAMES = 60
_MIN_TRACK_HISTORY_FRAMES = 60
_ENDCARD_HOLD_S = 2.5
_TELEPORT_FRAC = 5.0

_MARKER_MODEL_KEYS = (
    "detector_yellow_pole_v1",
    "detector_green_dome_v1",
)
_CONE_SAMPLE_STRIDE = 30
_CONE_CLUSTER_RADIUS_PX = 60.0
_CONE_MIN_DETECTIONS = 3

_DEFAULT_ATHLETE_HEIGHT_M = 1.70

# Reversal smoothing — MSFT shuttles take ~5-12 s depending on level,
# so a 1 s window is enough to localize each turn.
_REVERSAL_SMOOTH_S = 1.0
_REVERSAL_MIN_TRAVEL_FRAC = 0.4


# --- MSFT level ladder ------------------------------------------------

# (level, shuttles_at_this_level, cumulative_after_level, speed_km_h).
# Source: Léger 1988 protocol — 1 shuttle = 1 × 20 m one-way trip.
# Speed starts at 8.5 km/h and increases by 0.5 km/h per level.
_MSFT_LADDER: tuple[tuple[float, int, int, float], ...] = (
    (1.0, 7, 7, 8.5),
    (2.0, 8, 15, 9.0),
    (3.0, 8, 23, 9.5),
    (4.0, 9, 32, 10.0),
    (5.0, 9, 41, 10.5),
    (6.0, 10, 51, 11.0),
    (7.0, 10, 61, 11.5),
    (8.0, 11, 72, 12.0),
    (9.0, 11, 83, 12.5),
    (10.0, 11, 94, 13.0),
    (11.0, 12, 106, 13.5),
    (12.0, 12, 118, 14.0),
    (13.0, 13, 131, 14.5),
    (14.0, 13, 144, 15.0),
    (15.0, 13, 157, 15.5),
    (16.0, 14, 171, 16.0),
    (17.0, 14, 185, 16.5),
    (18.0, 15, 200, 17.0),
    (19.0, 15, 215, 17.5),
    (20.0, 16, 231, 18.0),
    (21.0, 16, 247, 18.5),
)


# --- State -------------------------------------------------------------


@dataclass(frozen=True)
class _RunWindow:
    track_id: int
    start_frame: int
    stop_frame: int

    @property
    def duration_frames(self) -> int:
        return self.stop_frame - self.start_frame


# --- Pipeline ----------------------------------------------------------


class MultistageFitnessTest(BaseTest):
    """Multistage Fitness (Bleep Test): visual shuttle counting + MSFT
    level lookup."""

    test_id = "multistage-fitness"

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
        print(f"[msft] {len(cone_positions)} cone clusters detected")

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

        # === Run window ===
        run = find_run_on_track(
            track_history[player_track_id],
            min_run_frames=_MIN_RUN_FRAMES,
            teleport_frac=_TELEPORT_FRAC,
        )
        if run is None:
            raise ProtocolError(
                f"no sustained motion segment >= {_MIN_RUN_FRAMES / fps:.0f}s "
                "on the picked player track"
            )
        run_window = _RunWindow(
            track_id=player_track_id,
            start_frame=run[0],
            stop_frame=run[1],
        )

        # === Calibration ===
        baseline_bbox_h = float(np.median(
            [h for (_, _, _, h, _) in track_history[player_track_id]]
        ))
        cone_pair = _pick_two_cones(cone_positions)
        px_per_m, calibration_source = _calibrate(
            cone_pair=cone_pair,
            cone_spacing_m=_SHUTTLE_DISTANCE_M,
            baseline_bbox_h_px=baseline_bbox_h,
            assumed_height_m=self._assumed_height_m,
        )
        print(f"[msft] calibration: {px_per_m:.1f} px/m ({calibration_source})")

        # === Shuttle counting ===
        n_shuttles = _count_shuttles(
            history=track_history[player_track_id],
            run_window=run_window,
            fps=fps,
        )
        total_distance_m = n_shuttles * _SHUTTLE_DISTANCE_M
        final_level, max_speed_kmh = _level_and_max_speed_for_shuttle_count(n_shuttles)
        vo2_max = _vo2max_leger(max_speed_kmh)
        duration_s = run_window.duration_frames / fps
        avg_speed = total_distance_m / duration_s if duration_s > 0 else 0.0

        metrics: dict[str, MetricValue] = {
            "final_speed_level": MetricValue(raw=final_level, unit="level"),
            "total_distance_m": MetricValue(raw=total_distance_m, unit="m"),
            "average_speed_ms": MetricValue(raw=avg_speed, unit="m_per_s"),
            "vo2max_estimated": MetricValue(raw=vo2_max, unit="ml_per_kg_per_min"),
            "num_shuttles_completed": MetricValue(
                raw=float(n_shuttles), unit="count",
            ),
            "total_completion_time_s": MetricValue(raw=duration_s, unit="s"),
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
                    n_shuttles=n_shuttles, final_level=final_level,
                ))
                writer.write(img)

            endcard_rows = [
                (mid.replace("_", " "), f"{mv.raw:.3f} {mv.unit}",
                 int(round(scores[mid].score)) if mid in scores else 0)
                for mid, mv in metrics.items()
            ]
            endcard = render_endcard(
                title="Multistage Fitness (Bleep Test)",
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


# --- Shuttle counting -------------------------------------------------


def _count_shuttles(
    *,
    history: list[tuple[int, float, float, float, float]],
    run_window: _RunWindow,
    fps: float,
) -> int:
    """Each MSFT shuttle = one 20 m one-way trip = one direction
    reversal in the trajectory. Returns the number of reversals
    detected within the run window.

    For N back-to-back one-way shuttles the trajectory has N reversals
    (one at each end-of-shuttle); the very first frame counts as the
    start of shuttle 1, not a reversal.
    """
    in_window = [
        (fi, cx) for (fi, cx, _, _, _) in history
        if run_window.start_frame <= fi <= run_window.stop_frame
    ]
    if len(in_window) < 10:
        return 0
    xs_raw = np.array([w[1] for w in in_window], dtype=float)
    lane_width = float(xs_raw.max() - xs_raw.min())
    if lane_width < 10.0:
        return 0

    win = max(7, int(round(_REVERSAL_SMOOTH_S * fps)) | 1)
    xs = xs_raw
    if win < len(xs):
        kernel = np.ones(win, dtype=float) / win
        xs = np.convolve(xs, kernel, mode="same")

    min_travel = _REVERSAL_MIN_TRAVEL_FRAC * lane_width
    reversals = 0
    direction = 0
    last_extreme = float(xs[0])
    for x in xs[1:]:
        d = x - last_extreme
        if direction == 0:
            if abs(d) >= min_travel:
                direction = 1 if d > 0 else -1
                last_extreme = float(x)
                # Establishing direction also "completes" shuttle 1
                # (athlete reached the far cone). Count it.
                reversals += 1
        elif direction > 0 and d < 0 and abs(x - last_extreme) >= min_travel:
            reversals += 1
            last_extreme = float(x)
            direction = -1
        elif direction < 0 and d > 0 and abs(x - last_extreme) >= min_travel:
            reversals += 1
            last_extreme = float(x)
            direction = 1
        else:
            if direction > 0 and x > last_extreme:
                last_extreme = float(x)
            elif direction < 0 and x < last_extreme:
                last_extreme = float(x)

    return reversals


# --- Level / vo2max ---------------------------------------------------


def _level_and_max_speed_for_shuttle_count(
    n_shuttles: int,
) -> tuple[float, float]:
    """Return (level, max_speed_km_h) reached given cumulative shuttle
    count. 0 shuttles -> (0.0, 0.0). Caps at the top of the ladder."""
    last_level = 0.0
    last_speed = 0.0
    for level, _, cumulative, speed in _MSFT_LADDER:
        if n_shuttles >= cumulative:
            last_level = level
            last_speed = speed
        else:
            break
    return last_level, last_speed


def _vo2max_leger(max_speed_km_h: float) -> float:
    """Léger (1988) MSFT VO2max estimator (no age correction).
    VO2max (ml/kg/min) = 6 × MAS - 27.4, where MAS is in km/h.
    Returns 0 below MSFT level 1's start speed (athlete never qualified)."""
    if max_speed_km_h <= 0:
        return 0.0
    return 6.0 * max_speed_km_h - 27.4


# --- HUD --------------------------------------------------------------


def _hud_fields(
    *,
    frame_idx: int,
    fps: float,
    run: _RunWindow,
    n_shuttles: int,
    final_level: float,
) -> dict[str, str]:
    if frame_idx < run.start_frame:
        return {
            "phase": "ready",
            "elapsed": "-",
            "shuttles": "0",
            "level": "-",
        }
    if frame_idx <= run.stop_frame:
        return {
            "phase": "running",
            "elapsed": f"{(frame_idx - run.start_frame) / fps:.2f} s",
            "shuttles": str(n_shuttles),
            "level": f"{final_level:.1f}",
        }
    return {
        "phase": "finished",
        "elapsed": f"{run.duration_frames / fps:.3f} s",
        "shuttles": str(n_shuttles),
        "level": f"{final_level:.1f}",
    }
