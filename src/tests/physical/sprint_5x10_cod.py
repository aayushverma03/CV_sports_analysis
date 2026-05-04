"""5 x 10 m Sprint with Change of Direction pipeline.

Course: two cones 10 m apart. Athlete sprints A->B (1), turns,
B->A (2), turns, A->B (3), turns, B->A (4), turns, A->B (5). Total
50 m of running with four 180-degree turns.

Three-pass design with shared player_picker:

- Pass 1: detect + track every person, sample cone detections, run
  Lucas-Kanade + ORB camera-motion estimation. Stabilize all positions
  into frame-0 coordinates.
- Pick player + cluster cones into start/finish ends + calibrate.
- Pass 2: pose on the picked player's bbox; collect ankle positions
  in stabilized space. Rep boundaries come from the ankle x-trajectory
  (the athlete's feet land precisely at the cones during turns, while
  the bbox center lags / swings during fast direction changes).
- Pass 3: render annotated video using cached pose results.

Metrics scored: total_completion_time_s, sprint_best_s,
fatigue_drop_off_pct. Informational: rep_times_s, total_distance_m,
average_speed_ms, max_speed_ms, peak_acceleration_ms2,
peak_deceleration_ms2.

Camera-pan compensation: the operator may pan the camera to follow
the athlete. Pass 1 estimates a per-frame affine transform from
background features (everything outside the athlete bbox) and
remaps both athlete bbox centers and cone detections into frame 0's
pixel space. All downstream math (cone clustering, calibration, turn
detection) runs on stabilized coordinates, so the pipeline produces
correct metrics whether the camera is static or follows.
"""
from __future__ import annotations

import json
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
from src.core.calibration.camera_motion import CameraMotion
from src.core.detection.marker_detector import MarkerDetector
from src.core.detection.player_detector import PERSON_CLASS_ID
from src.core.pose.estimator import create_pose_estimator
from src.core.tracking.bytetrack_tracker import ByteTrackTracker
from src.core.tracking.player_picker import pick_player
from src.core.tracking.run_window import cluster_object_positions
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

# A picked track shorter than this can't contain 5 reps.
_MIN_TRACK_HISTORY_FRAMES = 240    # 8 s @ 30 fps

_POSE_INTERVAL_FRAMES = 3
_POSE_CONF_MIN = 0.30
_ENDCARD_HOLD_S = 2.5

# Motion-onset detection: the first frame where the athlete's x has
# moved this fraction of the cone separation toward cone B is the
# real start of rep 1 (the picker's track usually includes a pre-start
# standing period that would otherwise inflate rep 1's time).
_MOTION_ONSET_FRAC = 0.10

# Marker sampling — sport-hall yellow slalom poles need a richer prompt
# vocabulary and lower confidence than the registry default. Stride is
# also tighter than the agility tests because 5x10m videos are short
# (15-25 s) and we need enough samples for the cluster threshold.
_MARKER_PROMPTS = (
    "yellow slalom pole",
    "agility pole",
    "yellow vertical pole",
    "orange traffic cone",
    "training cone",
)
_MARKER_CONFIDENCE = 0.05
_CONE_SAMPLE_STRIDE = 30
_CONE_CLUSTER_RADIUS_PX = 60.0
_CONE_MIN_DETECTIONS = 3

# Turn detection: a turn fires when the athlete's smoothed x crosses
# past a cone IN THE CURRENT RUN DIRECTION and then reverses. Past =
# x - cone_x has the sign expected for "beyond the cone".
_TURN_BEYOND_CONE_FRAC = 0.10  # athlete must overshoot cone by 10% of cone-pair distance
# Minimum frames between consecutive turns: prevents jitter near the
# cone from registering as multiple turns. Sprint-best 1.8 s -> 54 frames
# at 30 fps; use a generous floor of 1.0 s.
_TURN_DEBOUNCE_S = 1.0

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

        # === PASS 1: detect + track + camera-motion estimation ===
        # Track histories store STABILIZED coords (frame-0 pixel space)
        # so cone clustering, calibration, and turn detection all work
        # whether the camera is static or pans to follow the athlete.
        track_history_raw: dict[
            int, list[tuple[int, float, float, float, float]]
        ] = {}
        track_bboxes: dict[int, dict[int, np.ndarray]] = {}
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
                # Raw pixel-space record; stabilized later in one pass.
                track_history_raw.setdefault(p.track_id, []).append(
                    (frame.idx, float(p.center[0]), float(p.center[1]),
                     p.height, p.width)
                )
                track_bboxes.setdefault(p.track_id, {})[frame.idx] = (
                    p.bbox_xyxy.copy()
                )
            if frame.idx % _CONE_SAMPLE_STRIDE == 0:
                if self._marker is None:
                    self._marker = MarkerDetector(
                        prompts=list(_MARKER_PROMPTS),
                        confidence=_MARKER_CONFIDENCE,
                    )
                for det in self._marker.detect(frame.image):
                    cone_detections_per_frame.append((
                        frame.idx,
                        float(det.center[0]),
                        float(det.center[1]),
                    ))
            # Camera motion: mask out all persons in frame so we only
            # track static background features.
            motion.update(
                frame.idx, frame.image,
                exclude_bboxes_xyxy=person_bboxes_this_frame,
            )

        # === Stabilize track history + cone detections to frame-0 coords ===
        track_history: dict[
            int, list[tuple[int, float, float, float, float]]
        ] = {}
        for tid, hist in track_history_raw.items():
            stabilized: list[tuple[int, float, float, float, float]] = []
            for fi, cx, cy, h, w in hist:
                sx, sy = motion.transform_point(fi, (cx, cy))
                stabilized.append((fi, sx, sy, h, w))
            track_history[tid] = stabilized

        cone_detections: list[tuple[float, float]] = [
            motion.transform_point(fi, (cx, cy))
            for (fi, cx, cy) in cone_detections_per_frame
        ]

        # === Calibration ===
        # Preferred: cone-pair geometry (two well-separated detected
        # clusters). Fallback: athlete x-trajectory extrema (the player
        # provably reaches each cone during the test; the trajectory's
        # x-extent in stabilized space spans 10 m by protocol).
        cone_positions = cluster_object_positions(
            cone_detections,
            radius_px=_CONE_CLUSTER_RADIUS_PX,
            min_count=_CONE_MIN_DETECTIONS,
        )
        print(f"[5x10m-cod] {len(cone_positions)} cone clusters detected")
        cone_a, cone_b = _pick_cone_pair(cone_positions)

        eligible_pair = (
            cone_a is not None and cone_b is not None
            and _is_horizontal_pair(cone_a, cone_b)
        )
        if eligible_pair:
            cone_dist_px = float(np.hypot(
                cone_b[0] - cone_a[0], cone_b[1] - cone_a[1]
            ))
            px_per_m = cone_dist_px / _CONE_SEPARATION_M
            print(f"[5x10m-cod] calibration: {px_per_m:.1f} px/m "
                  f"(cone-pair = {cone_dist_px:.0f} px = 10 m)")
        else:
            # Trajectory-extrema fallback. Pick the longest-history
            # track as a calibration proxy (the picker hasn't run yet).
            longest = max(track_history.values(), key=len, default=[])
            xs = [cx for (_, cx, _, _, _) in longest]
            if len(xs) < _MIN_TRACK_HISTORY_FRAMES:
                raise CalibrationError(
                    "no horizontally-aligned cone pair AND no track "
                    "long enough for trajectory-extrema fallback"
                )
            x_min, x_max = min(xs), max(xs)
            cone_dist_px = x_max - x_min
            px_per_m = cone_dist_px / _CONE_SEPARATION_M
            cy_avg = float(np.mean([cy for (_, _, cy, _, _) in longest]))
            cone_a = (x_min, cy_avg)
            cone_b = (x_max, cy_avg)
            print(f"[5x10m-cod] trajectory-extrema calibration: "
                  f"{px_per_m:.1f} px/m (lane span = {cone_dist_px:.0f} px)")

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

        # Forensic dump of picked-track + cones for offline analysis.
        dump_path = output_dir / f"{self.test_id}.dump.json"
        dump_path.write_text(json.dumps({
            "fps": float(fps),
            "cone_a": list(cone_a),
            "cone_b": list(cone_b),
            "cone_dist_px": cone_dist_px,
            "px_per_m": px_per_m,
            "picked_track_id": int(player_track_id),
            "history": [
                [int(fi), float(cx), float(cy), float(h), float(w)]
                for (fi, cx, cy, h, w) in track_history[player_track_id]
            ],
        }))

        # === PASS 2: pose on picked player; collect ankle positions ===
        # Ankle x-trajectory is more accurate than bbox center for rep
        # turn detection — feet land at the cones, while the bbox
        # center lags or swings. Run pose only on the picked player's
        # bbox at _POSE_INTERVAL_FRAMES stride; transform ankle pixel
        # coords into stabilized (frame-0) space using the same camera
        # motion that stabilized the track history.
        player_bboxes_by_frame = track_bboxes.get(player_track_id, {})
        pose_results: dict[int, object] = {}
        ankle_trajectory: list[tuple[int, float, float]] = []   # (fi, ax_stab, ay_stab)
        for frame in frame_iter(video_path):
            if frame.idx % _POSE_INTERVAL_FRAMES != 0:
                continue
            bbox = player_bboxes_by_frame.get(frame.idx)
            if bbox is None:
                continue
            pose = self._pose.estimate_bbox(frame.image, bbox)
            pose_results[frame.idx] = pose
            ankle_xy = _ankle_midpoint_pixel(pose)
            if ankle_xy is None:
                continue
            sx, sy = motion.transform_point(frame.idx, ankle_xy)
            ankle_trajectory.append((frame.idx, sx, sy))

        # Forensic dump for offline rep-detection analysis.
        ankle_dump = output_dir / f"{self.test_id}.ankles.json"
        ankle_dump.write_text(json.dumps({
            "fps": float(fps),
            "ankles": [
                [int(fi), float(ax), float(ay)]
                for (fi, ax, ay) in ankle_trajectory
            ],
        }))

        if len(ankle_trajectory) < 10:
            raise ProtocolError(
                "insufficient pose / ankle data on the picked player "
                f"({len(ankle_trajectory)} samples) — could not run "
                "rep detection"
            )

        ankle_history = [
            (fi, ax, ay, 0.0, 0.0)
            for (fi, ax, ay) in ankle_trajectory
        ]
        rep_boundaries = _detect_turns(
            history=ankle_history,
            start_frame=_motion_onset_frame(
                history=ankle_history,
                onset_px=_MOTION_ONSET_FRAC * cone_dist_px,
            ),
            stop_frame=ankle_history[-1][0],
            cone_a=cone_a,
            cone_b=cone_b,
            cone_dist_px=cone_dist_px,
            fps=fps,
        )
        if len(rep_boundaries) < 6:
            raise ProtocolError(
                f"detected only {len(rep_boundaries)-1} reps; expected 5. "
                "Check that the athlete completes 5 sprints between cones "
                "and that pose tracking is stable on the feet."
            )
        best = _TestRun(
            track_id=player_track_id,
            start_frame=rep_boundaries[0],
            stop_frame=rep_boundaries[-1],
            rep_boundary_frames=tuple(rep_boundaries),
        )

        metrics = _compute_metrics(
            history=track_history[player_track_id],
            best=best,
            fps=fps,
            px_per_m=px_per_m,
        )
        scores, test_score = score_test(metrics, self.test_id, athlete.gender)

        # === PASS 3: render annotated video ===
        writer = cv2.VideoWriter(
            str(out_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (info.width, info.height),
        )
        if not writer.isOpened():
            raise ProtocolError(f"could not open VideoWriter at {out_path}")

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
                    pose = pose_results.get(frame.idx)
                    if pose is not None:
                        last_pose = pose
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


# --- Pose helper -------------------------------------------------------


def _ankle_midpoint_pixel(pose) -> tuple[float, float] | None:
    """Mean of left+right ankle keypoints (pixel coords). Returns None
    if neither ankle has confidence above the threshold.

    Per the test docstring, ankle position is a better proxy than bbox
    center for cone-touch detection: feet land at the cones, while the
    torso bbox lags or swings during turns.
    """
    if pose is None:
        return None
    pts: list[tuple[float, float]] = []
    for kp_name in ("left_ankle", "right_ankle"):
        try:
            conf = pose.confidence_of(kp_name)
        except Exception:
            return None
        if conf < _POSE_CONF_MIN:
            continue
        pos = pose.position(kp_name)
        pts.append((float(pos[0]), float(pos[1])))
    if not pts:
        return None
    cx = sum(p[0] for p in pts) / len(pts)
    cy = sum(p[1] for p in pts) / len(pts)
    return cx, cy


# --- Cone helpers -----------------------------------------------------


def _is_horizontal_pair(
    a: tuple[float, float], b: tuple[float, float],
    *,
    max_y_to_x_ratio: float = 0.4,
) -> bool:
    """A cone pair on a horizontal lane should differ much more in x
    than in y. Reject pairs that are mostly diagonal — these are usually
    pole-top vs disk-base detections of the same physical pole, or
    other false matches.
    """
    dx = abs(b[0] - a[0])
    dy = abs(b[1] - a[1])
    if dx == 0:
        return False
    return (dy / dx) <= max_y_to_x_ratio


def _pick_cone_pair(
    clusters: list[tuple[float, float]],
) -> tuple[tuple[float, float] | None, tuple[float, float] | None]:
    """Return the two LANE-END midpoints, left-to-right.

    Real-world setup is two cones at the start line + two at the finish
    line. After spatial clustering each cone gets its own cluster, so
    the typical input is 4 clusters in 2 spatially-separated groups.
    A 1-D gap split on the x-axis identifies the start group and the
    finish group; centroids of each give the actual lane endpoints
    (where the athlete touches the line, NOT the outer cones).

    Falls back to max-x-separation if there's no clear gap (e.g. only
    2 clusters, or very evenly-spaced cones).
    """
    if len(clusters) < 2:
        return None, None
    if len(clusters) == 2:
        a, b = clusters
        return (a, b) if a[0] <= b[0] else (b, a)

    # Sort by x and find the largest gap.
    s = sorted(clusters, key=lambda c: c[0])
    xs = [c[0] for c in s]
    gaps = [(xs[i + 1] - xs[i], i) for i in range(len(xs) - 1)]
    biggest_gap, split_idx = max(gaps, key=lambda g: g[0])
    other_gaps = [g for g, i in gaps if i != split_idx]

    # Require the lane gap to be at least 2x the average inter-cone
    # spacing within an end. If there's no clean split, the geometry
    # isn't 4-cone-style — fall back to max-separation.
    if other_gaps and biggest_gap < 2.0 * (sum(other_gaps) / len(other_gaps)):
        best = (0.0, None, None)
        for i in range(len(s)):
            for j in range(i + 1, len(s)):
                dx = abs(s[i][0] - s[j][0])
                if dx > best[0]:
                    best = (dx, s[i], s[j])
        a, b = best[1], best[2]
        return (a, b) if a[0] <= b[0] else (b, a)

    left_group = s[: split_idx + 1]
    right_group = s[split_idx + 1:]
    left_centroid = (
        sum(c[0] for c in left_group) / len(left_group),
        sum(c[1] for c in left_group) / len(left_group),
    )
    right_centroid = (
        sum(c[0] for c in right_group) / len(right_group),
        sum(c[1] for c in right_group) / len(right_group),
    )
    return left_centroid, right_centroid


# --- Start / turn detection -------------------------------------------


def _motion_onset_frame(
    *,
    history: list[tuple[int, float, float, float, float]],
    onset_px: float,
) -> int:
    """First frame whose x has moved away from the initial x by `onset_px`.

    Picker-selected tracks usually contain a pre-start standing period
    while the athlete is in starting position. Trimming to the first
    frame of meaningful motion keeps rep 1's timing honest. Works
    regardless of which cone the athlete starts near (or starts in
    between).
    """
    if not history:
        return 0
    first_fi, first_cx, _, _, _ = history[0]
    for fi, cx, _, _, _ in history:
        if abs(cx - first_cx) >= onset_px:
            return int(fi)
    return int(first_fi)


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

    Algorithm: smooth the x signal, find prominent local extrema with
    `scipy.signal.find_peaks` (min distance and min prominence both
    relative to the lane width). Each extremum = one rep boundary
    (athlete reaching a cone and turning). Robust whether the cones
    were detected directly or inferred from trajectory extrema.
    """
    from scipy.signal import find_peaks

    in_window = [
        (fi, cx) for (fi, cx, _, _, _) in history
        if start_frame <= fi <= stop_frame
    ]
    if len(in_window) < 10:
        return []
    fis = np.array([w[0] for w in in_window])
    xs = np.array([w[1] for w in in_window], dtype=float)

    # Smooth strongly enough that small wobbles don't register as
    # extrema; ~0.5 s window is short relative to a sprint rep
    # (1.5-3 s) but long enough to suppress single-step bbox jitter.
    win = max(7, int(round(0.5 * fps)) | 1)
    if win < len(xs):
        kernel = np.ones(win) / win
        xs = np.convolve(xs, kernel, mode="same")

    # Distance: at least half a sprint-rep apart, in frames.
    min_distance = max(int(round(0.8 * fps)), 5)
    # Prominence: at least 15% of the lane width. Ankle trajectory has
    # more high-frequency wobble than bbox center but reaches peaks at
    # the actual cones; a looser prominence avoids missing partial reps
    # where the athlete cuts the turn slightly short.
    min_prominence = 0.15 * cone_dist_px

    maxima, _ = find_peaks(xs, distance=min_distance, prominence=min_prominence)
    minima, _ = find_peaks(-xs, distance=min_distance, prominence=min_prominence)
    extrema_idxs = sorted(set(maxima.tolist() + minima.tolist()))

    if not extrema_idxs:
        return []

    boundaries: list[int] = [int(fis[0])]
    for idx in extrema_idxs:
        boundaries.append(int(fis[idx]))
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

    # Smooth bbox-center positions before differentiating. Bbox detection
    # has frame-to-frame jitter (the box snaps tighter or looser to the
    # body silhouette); without smoothing, those jitters appear as
    # impossible peak speeds (50+ m/s) and accelerations (>500 m/s^2).
    # Window must be short enough to preserve real direction changes
    # (~0.4 s rep transitions) — 7 frames at 30 fps = 0.23 s.
    pos_window = max(7, int(round(0.25 * fps)) | 1)
    if len(centers_px) > pos_window:
        from src.core.utils.smoothing import savgol_smooth
        centers_px = np.column_stack([
            savgol_smooth(centers_px[:, 0], window=pos_window, polyorder=3),
            savgol_smooth(centers_px[:, 1], window=pos_window, polyorder=3),
        ])
    centers_m = centers_px / px_per_m

    # Per-frame instantaneous speed (m/s) from successive position deltas.
    diffs_m = np.diff(centers_m, axis=0)
    inst_speed = np.linalg.norm(diffs_m, axis=1) * fps  # m/frame * fps = m/s
    # Bbox-center jitter (panning cameras, pose / detection wobble) can
    # push a single frame's "speed" past 50 m/s — physically impossible
    # for humans (Bolt's peak: ~12 m/s). Cap at 15 m/s before peak /
    # accel calculations so jitter doesn't dominate the result.
    inst_speed = np.clip(inst_speed, 0.0, 15.0)
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
