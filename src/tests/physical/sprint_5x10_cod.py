"""5 x 10 m Sprint with Change of Direction pipeline.

Course: two cones 10 m apart. Athlete sprints A->B (1), turns,
B->A (2), turns, A->B (3), turns, B->A (4), turns, A->B (5). Total
50 m of running with four 180-degree turns.

Same 2-pass + shared player_picker design as the agility tests, but
uses the cone PAIR for both player-picker proximity AND pixel-to-metre
calibration (cone separation = 10 m). Per-rep timing is detected from
the picked athlete's x-trajectory: each time the athlete crosses past
a cone in the run direction, that's a turn / shuttle boundary.

Metrics scored: total_completion_time_s, sprint_best_s,
fatigue_drop_off_pct. Informational: rep_times_s, total_distance_m,
average_speed_ms, max_speed_ms, peak_acceleration_ms2,
peak_deceleration_ms2.
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
from src.core.calibration.camera_calibration import CalibrationError
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
from src.metrics.motion.average_speed_ms import average_speed_ms
from src.metrics.motion.max_speed_ms import max_speed_ms
from src.metrics.motion.peak_acceleration_ms2 import peak_acceleration_ms2
from src.metrics.motion.peak_deceleration_ms2 import peak_deceleration_ms2
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

# Protocol-fixed: 2 cones, 10 m apart, 5 shuttles -> 50 m total.
_CONE_SEPARATION_M = 10.0
_N_SHUTTLES = 5
_TOTAL_DISTANCE_M = _CONE_SEPARATION_M * _N_SHUTTLES

# Sprint-best is on the order of 1.8-3.0 s; the slowest plausible
# completion of all 5 reps is ~25 s. Anything below 8 s implies bad
# turn detection or a non-protocol video.
_MIN_RUN_FRAMES = 240          # 8 s @ 30 fps; below this is implausible

_MIN_TRACK_HISTORY_FRAMES = 60
_POSE_INTERVAL_FRAMES = 3
_ENDCARD_HOLD_S = 2.5

# Cone sampling — same shape as t_test/illinois.
_CONE_SAMPLE_STRIDE = 60
_CONE_CLUSTER_RADIUS_PX = 40.0
_CONE_MIN_DETECTIONS = 3

# Turn detection: a turn fires when the athlete's smoothed x crosses
# past a cone IN THE CURRENT RUN DIRECTION and then reverses. Past =
# x - cone_x has the sign expected for "beyond the cone".
_TURN_BEYOND_CONE_FRAC = 0.10  # athlete must overshoot cone by 10% of cone-pair distance
# Minimum frames between consecutive turns: prevents jitter near the
# cone from registering as multiple turns. Sprint-best 1.8 s -> 54 frames
# at 30 fps; use a generous floor of 1.0 s.
_TURN_DEBOUNCE_S = 1.0

# Single-athlete agility-style video: brief tracker hiccups should not
# break the run. See illinois_agility.py for the same rationale.
_TELEPORT_FRAC = 5.0


# --- State -------------------------------------------------------------


@dataclass(frozen=True)
class _TestRun:
    track_id: int
    start_frame: int
    stop_frame: int
    rep_boundary_frames: tuple[int, ...]   # length 6: start, t1, t2, t3, t4, stop

    @property
    def duration_frames(self) -> int:
        return self.stop_frame - self.start_frame


# --- Pipeline ----------------------------------------------------------


class Sprint5x10CodTest(BaseTest):
    """5 x 10 m Sprint with COD: 2-pass + cone-pair calibration."""

    test_id = "5x10m-sprint-cod"

    def __init__(self) -> None:
        self._tracker = ByteTrackTracker(
            classes=[PERSON_CLASS_ID],
            confidence=0.20,
        )
        self._pose = create_pose_estimator("pose_default")
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
            if frame.idx % _CONE_SAMPLE_STRIDE == 0:
                if self._marker is None:
                    self._marker = MarkerDetector()
                for det in self._marker.detect(frame.image):
                    cone_detections.append(
                        (float(det.center[0]), float(det.center[1]))
                    )

        # === Cone-pair calibration ===
        cone_positions = cluster_object_positions(
            cone_detections,
            radius_px=_CONE_CLUSTER_RADIUS_PX,
            min_count=_CONE_MIN_DETECTIONS,
        )
        print(f"[5x10m-cod] {len(cone_positions)} cone clusters detected")
        cone_a, cone_b = _pick_cone_pair(cone_positions)
        if cone_a is None or cone_b is None:
            raise CalibrationError(
                f"need 2 cones for 10 m calibration; found "
                f"{len(cone_positions)} clusters — check cone visibility "
                "and YOLO-World class prompts"
            )
        cone_dist_px = float(np.hypot(cone_b[0] - cone_a[0], cone_b[1] - cone_a[1]))
        if cone_dist_px <= 0:
            raise CalibrationError("degenerate cone pair: zero separation")
        px_per_m = cone_dist_px / _CONE_SEPARATION_M
        print(f"[5x10m-cod] calibration: {px_per_m:.1f} px/m "
              f"(cone-pair = {cone_dist_px:.0f} px = 10 m)")

        # === Pick THE player track ===
        object_positions = (
            {fi: list(cone_positions) for fi in range(n_frames)}
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
                "could not identify a single player track — "
                "neither pixel-area dominance nor cone-proximity fallback "
                "yielded a winner"
            )

        run_window = find_run_on_track(
            track_history[player_track_id],
            min_run_frames=_MIN_RUN_FRAMES,
            teleport_frac=_TELEPORT_FRAC,
        )
        if run_window is None:
            raise ProtocolError(
                f"player track {player_track_id} found but no sustained "
                f"motion segment >= {_MIN_RUN_FRAMES / fps:.0f} s — "
                "could not time the run"
            )
        start_frame, stop_frame = run_window

        # === Detect 4 turns -> 5 rep boundaries ===
        rep_boundaries = _detect_turns(
            history=track_history[player_track_id],
            start_frame=start_frame,
            stop_frame=stop_frame,
            cone_a=cone_a,
            cone_b=cone_b,
            cone_dist_px=cone_dist_px,
            fps=fps,
        )
        if len(rep_boundaries) < 6:
            raise ProtocolError(
                f"detected only {len(rep_boundaries)-1} reps; expected 5. "
                "Camera-perpendicular view is required so the athlete's "
                "x-axis projects cleanly onto the cone line."
            )
        best = _TestRun(
            track_id=player_track_id,
            start_frame=rep_boundaries[0],
            stop_frame=rep_boundaries[-1],
            rep_boundary_frames=tuple(rep_boundaries),
        )

        # === Compute metrics on the picked track within the run window ===
        metrics = _compute_metrics(
            history=track_history[player_track_id],
            best=best,
            fps=fps,
            px_per_m=px_per_m,
        )
        scores, test_score = score_test(metrics, self.test_id, athlete.gender)

        # === PASS 2: re-iterate, annotate only the chosen player ===
        writer = cv2.VideoWriter(
            str(out_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (info.width, info.height),
        )
        if not writer.isOpened():
            raise ProtocolError(f"could not open VideoWriter at {out_path}")

        player_bboxes_by_frame = track_bboxes.get(best.track_id, {})
        history_by_frame: dict[int, tuple[float, float]] = {
            fi: (cx, cy)
            for (fi, cx, cy, _, _) in track_history[player_track_id]
        }
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

                draw_hud(img, _player_hud_fields(
                    frame.idx, fps, best, history_by_frame, px_per_m
                ))
                writer.write(img)

            endcard_rows = [
                (mid.replace("_", " "), f"{mv.raw:.3f} {mv.unit}",
                 int(round(scores[mid].score)) if mid in scores else 0)
                for mid, mv in metrics.items()
            ]
            endcard = render_endcard(
                title="5 x 10 m Sprint w/ COD",
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


# --- Cone helpers -----------------------------------------------------


def _pick_cone_pair(
    clusters: list[tuple[float, float]],
) -> tuple[tuple[float, float] | None, tuple[float, float] | None]:
    """Return the two clusters farthest apart in image x.

    With exactly 2 clusters, returns them in left-to-right order. With
    > 2, picks the pair with maximum x-separation (most likely the
    end-cones of the lane). With < 2, returns (None, None).
    """
    if len(clusters) < 2:
        return None, None
    if len(clusters) == 2:
        a, b = clusters
        return (a, b) if a[0] <= b[0] else (b, a)
    best = (0.0, None, None)
    for i in range(len(clusters)):
        for j in range(i + 1, len(clusters)):
            dx = abs(clusters[i][0] - clusters[j][0])
            if dx > best[0]:
                best = (dx, clusters[i], clusters[j])
    a, b = best[1], best[2]
    return (a, b) if a[0] <= b[0] else (b, a)


# --- Turn detection ----------------------------------------------------


def _detect_turns(
    *,
    history: list[tuple[int, float, float, float, float]],
    start_frame: int,
    stop_frame: int,
    cone_a: tuple[float, float],
    cone_b: tuple[float, float],
    cone_dist_px: float,
    fps: float,
) -> list[int]:
    """Find rep boundaries on the picked track's x-trajectory.

    Returns the 6-element list [start, t1, t2, t3, t4, stop] where each
    t_i is the frame index of the i-th turn. Returns fewer if turns
    couldn't be reliably detected.

    Algorithm: scan the smoothed x signal between start and stop. The
    first crossing past cone B (going right) ends rep 1. Direction
    flips, the next crossing past cone A (going left) ends rep 2. And
    so on, alternating. A "crossing past" requires the athlete's x to
    overshoot the cone by `_TURN_BEYOND_CONE_FRAC * cone_dist_px`.
    """
    in_window = [
        (fi, cx) for (fi, cx, _, _, _) in history
        if start_frame <= fi <= stop_frame
    ]
    if len(in_window) < 10:
        return []
    fis = np.array([w[0] for w in in_window])
    xs = np.array([w[1] for w in in_window], dtype=float)

    # Light smoothing on x to suppress per-frame jitter without delaying
    # turn detection materially.
    win = max(5, int(round(0.2 * fps)) | 1)  # ~0.2 s, odd
    if win < len(xs):
        kernel = np.ones(win) / win
        xs = np.convolve(xs, kernel, mode="same")

    overshoot = _TURN_BEYOND_CONE_FRAC * cone_dist_px
    debounce = int(round(_TURN_DEBOUNCE_S * fps))

    # Establish start direction from the first significant motion.
    cone_left_x, cone_right_x = cone_a[0], cone_b[0]
    boundaries: list[int] = [int(fis[0])]
    expected_target = cone_right_x  # rep 1 target
    target_overshoot_sign = 1       # going right -> overshoot is x > target

    last_turn_idx = -10**9
    for i, x in enumerate(xs):
        if (fis[i] - last_turn_idx) < debounce:
            continue
        delta = (x - expected_target) * target_overshoot_sign
        if delta >= overshoot:
            boundaries.append(int(fis[i]))
            last_turn_idx = fis[i]
            # Flip target for next rep.
            if expected_target == cone_right_x:
                expected_target = cone_left_x
                target_overshoot_sign = -1   # going left -> x < target
            else:
                expected_target = cone_right_x
                target_overshoot_sign = 1
            if len(boundaries) == _N_SHUTTLES + 1:
                break
    return boundaries


# --- Metrics -----------------------------------------------------------


def _compute_metrics(
    *,
    history: list[tuple[int, float, float, float, float]],
    best: _TestRun,
    fps: float,
    px_per_m: float,
) -> dict[str, MetricValue]:
    """Convert pixel trajectory + rep boundaries to scored metrics."""
    in_window = [
        (fi, cx, cy) for (fi, cx, cy, _, _) in history
        if best.start_frame <= fi <= best.stop_frame
    ]
    fis = np.array([w[0] for w in in_window])
    centers_px = np.array([(w[1], w[2]) for w in in_window], dtype=float)
    centers_m = centers_px / px_per_m

    # Per-frame instantaneous speed (m/s) from successive position deltas.
    diffs_m = np.diff(centers_m, axis=0)
    inst_speed = np.linalg.norm(diffs_m, axis=1) * fps  # m/frame * fps = m/s
    if len(inst_speed) < 11:
        inst_speed = np.pad(inst_speed, (0, 11 - len(inst_speed)))

    duration_s = best.duration_frames / fps
    rep_times = tuple(
        (best.rep_boundary_frames[i + 1] - best.rep_boundary_frames[i]) / fps
        for i in range(_N_SHUTTLES)
    )
    sprint_best_s = float(min(rep_times))
    fatigue = (rep_times[-1] - rep_times[0]) / rep_times[0] * 100.0

    return {
        "total_completion_time_s": MetricValue(raw=duration_s, unit="s"),
        "sprint_best_s": MetricValue(raw=sprint_best_s, unit="s"),
        "fatigue_drop_off_pct": MetricValue(raw=float(fatigue), unit="pct"),
        "total_distance_m": MetricValue(raw=_TOTAL_DISTANCE_M, unit="m"),
        "average_speed_ms": MetricValue(
            raw=average_speed_ms(_TOTAL_DISTANCE_M, duration_s), unit="m_per_s",
        ),
        "max_speed_ms": MetricValue(
            raw=max_speed_ms(inst_speed), unit="m_per_s",
        ),
        "peak_acceleration_ms2": MetricValue(
            raw=peak_acceleration_ms2(inst_speed, fps=fps), unit="m_per_s2",
        ),
        "peak_deceleration_ms2": MetricValue(
            raw=peak_deceleration_ms2(inst_speed, fps=fps), unit="m_per_s2",
        ),
        # Per-rep times as informational dict-like metric (unit "s_list").
        # Each rep is also surfaced individually so the endcard can show them.
        **{
            f"rep{i+1}_time_s": MetricValue(raw=float(rep_times[i]), unit="s")
            for i in range(_N_SHUTTLES)
        },
    }


# --- HUD --------------------------------------------------------------


def _player_hud_fields(
    frame_idx: int,
    fps: float,
    run: _TestRun,
    history_by_frame: dict[int, tuple[float, float]],
    px_per_m: float,
) -> dict[str, str]:
    """HUD: phase, current shuttle 1-5, elapsed, current speed (m/s)."""
    if frame_idx < run.start_frame:
        return {"phase": "ready", "shuttle": "-", "time": "-", "speed": "-"}
    if frame_idx > run.stop_frame:
        total_s = run.duration_frames / fps
        return {
            "phase": "finished",
            "shuttle": f"{_N_SHUTTLES}/{_N_SHUTTLES}",
            "time": f"{total_s:.3f} s",
            "speed": "-",
        }
    # Which shuttle?
    shuttle = 1
    for i, b in enumerate(run.rep_boundary_frames[1:], start=1):
        if frame_idx <= b:
            shuttle = i
            break
    elapsed_s = (frame_idx - run.start_frame) / fps
    # Instantaneous speed: 5-frame finite diff in metres -> m/s.
    speed_str = "-"
    here = history_by_frame.get(frame_idx)
    earlier = history_by_frame.get(frame_idx - 5)
    if here is not None and earlier is not None:
        dx_m = (here[0] - earlier[0]) / px_per_m
        dy_m = (here[1] - earlier[1]) / px_per_m
        speed = float(np.hypot(dx_m, dy_m) * fps / 5.0)
        speed_str = f"{speed:.1f} m/s"
    return {
        "phase": "running",
        "shuttle": f"{shuttle}/{_N_SHUTTLES}",
        "time": f"{elapsed_s:.2f} s",
        "speed": speed_str,
    }
