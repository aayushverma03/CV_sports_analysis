"""Yo-Yo Intermittent Recovery Test Level 2 (IR2) pipeline.

Progressive shuttle test: athlete runs 2 × 20 m at audio-cued pace,
then 10 s of active recovery, then the next shuttle pair at a slightly
faster pace. Test ends when the athlete fails to reach the line on the
cue twice consecutively. Scored on `total_distance_m` and
`final_speed_level`.

Audio-beep detection (the spec's primary stage detector) is deferred to
a follow-up phase. v1 is **visual-only**: the shuttle count comes from
direction reversals on the picked athlete's stabilized x-trajectory,
total distance derives from `shuttle_pairs * 40 m`, and final-speed-
level is looked up from the standard IR2 progression table. This works
for self-paced training-style videos and any session where the audio
cuing is correct (athlete keeps pace = visual count tracks the audio
beep count). For audio-mismatched footage the user should switch to
manual mode (TODO) once that lands.

Pipeline reuses the agility/dribbling 3-pass scaffolding:
- Pass 1: ByteTrack person, sample cones, estimate camera transform
  (Lucas-Kanade + ORB anchor). Stabilize positions to frame-0 coords.
- Pick player, cluster cones, calibrate (cone-pair / 20 m, fall back
  to body-height proxy).
- Pass 2: count direction reversals on the picked athlete's stabilized
  x to derive shuttle count.
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

# IR2 shuttles are 2 × 20 m = 40 m. Cone-pair distance is the 20 m line.
_SHUTTLE_HALF_M = 20.0
_SHUTTLE_TOTAL_M = 40.0

_MIN_RUN_FRAMES = 60          # 2 s at 30 fps; short demos still time
_MIN_TRACK_HISTORY_FRAMES = 60
_ENDCARD_HOLD_S = 2.5
_TELEPORT_FRAC = 5.0          # single-athlete, lenient

_MARKER_MODEL_KEYS = (
    "detector_yellow_pole_v1",
    "detector_green_dome_v1",
)
_CONE_SAMPLE_STRIDE = 30
_CONE_CLUSTER_RADIUS_PX = 60.0
_CONE_MIN_DETECTIONS = 3

_DEFAULT_ATHLETE_HEIGHT_M = 1.70

# Reversal smoothing — IR2 athletes turn at the cones every ~5-15 s,
# so a 1 s window is short enough to localize the turn.
_REVERSAL_SMOOTH_S = 1.0
# Minimum displacement (fraction of lane width) the trajectory must
# travel between consecutive reversals — protects against in-place
# wobble during the recovery jog.
_REVERSAL_MIN_TRAVEL_FRAC = 0.4


# --- IR2 progression table --------------------------------------------

# (level, shuttles_at_this_level, cumulative_shuttles_after_level).
# A "shuttle" here = one out-and-back trip = 40 m.
# Source: standard IR2 audio-cue protocol (Bangsbo 2008 derivative).
# This ladder lets us map shuttle_count -> final_speed_level by finding
# the highest level whose cumulative count is <= shuttles_completed.
_IR2_LADDER: tuple[tuple[float, int, int], ...] = (
    (19.0, 1, 1),
    (20.0, 1, 2),
    (21.0, 2, 4),
    (22.0, 2, 6),
    (23.0, 3, 9),
    (24.0, 4, 13),
    (25.0, 8, 21),
    (26.0, 8, 29),
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


class YoYoIntermittentTest(BaseTest):
    """Yo-Yo IR2: visual shuttle counting + IR2 level lookup."""

    test_id = "yo-yo-intermittent"

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

        # === PASS 1: detect+track + cones + camera motion ===
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

        # === Stabilize tracks + cones to frame-0 coords ===
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
        print(f"[yo-yo] {len(cone_positions)} cone clusters detected")

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
            cone_spacing_m=_SHUTTLE_HALF_M,
            baseline_bbox_h_px=baseline_bbox_h,
            assumed_height_m=self._assumed_height_m,
        )
        print(f"[yo-yo] calibration: {px_per_m:.1f} px/m ({calibration_source})")

        # === Shuttle counting ===
        n_shuttles = _count_shuttles(
            history=track_history[player_track_id],
            run_window=run_window,
            fps=fps,
        )
        total_distance_m = n_shuttles * _SHUTTLE_TOTAL_M
        final_level = _level_for_shuttle_count(n_shuttles)
        vo2_max = _vo2max_from_distance_ir2(total_distance_m)
        duration_s = run_window.duration_frames / fps

        metrics: dict[str, MetricValue] = {
            "total_distance_m": MetricValue(raw=total_distance_m, unit="m"),
            "final_speed_level": MetricValue(raw=final_level, unit="level"),
            "num_shuttles_completed": MetricValue(
                raw=float(n_shuttles), unit="count",
            ),
            "total_completion_time_s": MetricValue(raw=duration_s, unit="s"),
            "vo2max_estimated": MetricValue(raw=vo2_max, unit="ml_per_kg_per_min"),
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
                title="Yo-Yo Intermittent (IR2)",
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
    """Each shuttle = athlete runs out and back = one full direction
    reversal cycle. Counted as the number of LOCAL EXTREMA in the
    smoothed x-trajectory divided by 2 (each shuttle has one max and
    one min; the first half of a shuttle ends at one extremum, the
    second half at the other).
    """
    in_window = [
        (fi, cx) for (fi, cx, _, _, _) in history
        if run_window.start_frame <= fi <= run_window.stop_frame
    ]
    if len(in_window) < 10:
        return 0
    xs_raw = np.array([w[1] for w in in_window], dtype=float)

    # Lane-width gate uses RAW xs to avoid convolution boundary effects.
    # mode="same" zero-pads at both ends, which on a stationary signal
    # creates a fake spike from baseline at the edges.
    lane_width = float(xs_raw.max() - xs_raw.min())
    if lane_width < 10.0:
        return 0

    win = max(7, int(round(_REVERSAL_SMOOTH_S * fps)) | 1)
    xs = xs_raw
    if win < len(xs):
        kernel = np.ones(win, dtype=float) / win
        xs = np.convolve(xs, kernel, mode="same")

    min_travel = _REVERSAL_MIN_TRAVEL_FRAC * lane_width

    extrema = 0
    direction = 0
    last_extreme = xs[0]
    for x in xs[1:]:
        d = x - last_extreme
        if direction == 0:
            if abs(d) >= min_travel:
                direction = 1 if d > 0 else -1
                last_extreme = x
        elif direction > 0 and d < 0 and abs(x - last_extreme) >= min_travel:
            extrema += 1
            last_extreme = x
            direction = -1
        elif direction < 0 and d > 0 and abs(x - last_extreme) >= min_travel:
            extrema += 1
            last_extreme = x
            direction = 1
        else:
            # Track running extreme on the same direction.
            if direction > 0 and x > last_extreme:
                last_extreme = x
            elif direction < 0 and x < last_extreme:
                last_extreme = x

    # For N consecutive out-and-back shuttles there are 2N-1 reversals
    # in the trajectory: N peaks (turn at far cone) + N-1 troughs (turn
    # at start cone, between shuttles). The trajectory often ends AT a
    # trough rather than past one, so the last reversal isn't counted.
    # Compensate with +1 before dividing by 2.
    return (extrema + 1) // 2


# --- Level / vo2max ---------------------------------------------------


def _level_for_shuttle_count(n_shuttles: int) -> float:
    """Map cumulative shuttle count to the corresponding IR2 level.

    Returns the LAST level fully completed (cumulative count <=
    n_shuttles). If the athlete didn't complete level 19, returns 0.0
    so the metric still produces a number for scoring.
    """
    last_level = 0.0
    for level, _, cumulative in _IR2_LADDER:
        if n_shuttles >= cumulative:
            last_level = level
        else:
            break
    return last_level


def _vo2max_from_distance_ir2(total_distance_m: float) -> float:
    """Bangsbo (2008) IR2 -> VO2max linear estimator. ml/kg/min."""
    return 0.0136 * total_distance_m + 45.3


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
