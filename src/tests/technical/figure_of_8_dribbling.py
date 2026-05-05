"""Figure of 8 Dribbling pipeline.

Athlete dribbles a ball in a figure-of-8 pattern around two cones
spaced 3 m apart. The protocol calls for 2 complete loops; each loop
threads the athlete around both cones, crossing the midpoint between
them twice (once in each direction).

Three-pass shape mirrors Zig-Zag Dribbling:

- Pass 1: ByteTrack person + ball, sample cones, estimate camera
  motion. Stabilize all positions to frame-0 coordinates.
- Pick player (cone-proximity fallback). Find run window.
- Pass 2: pose on picked player; collect ankle midpoints in
  stabilized space; touch detection (ankle-near-ball, debounced).
- Pass 3: render with HUD showing phase, elapsed, loops_completed,
  running touch count.

Loop detection: count perpendicular crossings of the line between the
two cones. Each figure-8 loop crosses that line twice — once between
the two halves of the 8 — so loops_completed = midpoint_crossings // 2.

v1 ships:
  - total_completion_time_s (scored)
  - total_ball_touches, touches_per_metre, left_leg_utilisation_pct
  - average_speed_ms, total_distance_m
  - loops_completed (informational HUD counter)

Same caveats as Zig-Zag for max_speed / peak_accel / loop_split_times_s
(deferred — needs pose-mid-hip tracking and per-loop boundary
analysis).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import cv2
import numpy as np

from src.core.annotation.overlays import (
    BALL,
    draw_bbox,
    draw_hud,
    draw_skeleton,
    render_endcard,
)
from src.core.calibration.camera_motion import CameraMotion
from src.core.detection.marker_detector import MarkerDetector
from src.core.detection.player_detector import PERSON_CLASS_ID
from src.core.pose.estimator import create_pose_estimator
from src.core.pose.orientation import ankle_side, body_center_x
from src.core.tracking.bytetrack_tracker import ByteTrackTracker, TrackedDetection
from src.core.tracking.player_picker import pick_player
from src.core.tracking.run_window import (
    cluster_object_positions,
    find_run_on_track,
)
from src.core.utils.video_io import frame_iter, video_info
from src.metrics.ball.touches_per_metre import touches_per_metre
from src.metrics.motion.average_speed_ms import average_speed_ms
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

SPORTS_BALL_CLASS_ID = 32

# --- Tunables ----------------------------------------------------------

# Slowest realistic figure-8 (2 loops): ~12 s. Below 4 s is implausible.
_MIN_RUN_FRAMES = 120          # 4 s @ 30 fps

_MIN_TRACK_HISTORY_FRAMES = 60
_POSE_INTERVAL_FRAMES = 3
_ENDCARD_HOLD_S = 2.5

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

# Two cones, 3 m apart per protocol.
_CONE_SPACING_M = 3.0
_DEFAULT_ATHLETE_HEIGHT_M = 1.70
_TELEPORT_FRAC = 5.0

_TOUCH_PROXIMITY_FRAC = 0.30
_TOUCH_DEBOUNCE_S = 0.20
_POSE_CONF_MIN = 0.30


# --- State -------------------------------------------------------------


@dataclass(frozen=True)
class _RunWindow:
    track_id: int
    start_frame: int
    stop_frame: int

    @property
    def duration_frames(self) -> int:
        return self.stop_frame - self.start_frame


@dataclass
class _Touch:
    frame_idx: int
    side: Literal["L", "R"]


# --- Pipeline ----------------------------------------------------------


class FigureOf8DribblingTest(BaseTest):
    """Figure of 8 Dribbling: 3-pass with camera-motion + 2-cone setup."""

    test_id = "figure-of-8-dribbling"

    def __init__(
        self,
        *,
        n_loops: int = 2,
        assumed_athlete_height_m: float = _DEFAULT_ATHLETE_HEIGHT_M,
    ) -> None:
        if n_loops < 1:
            raise ValueError(f"n_loops must be >= 1, got {n_loops}")
        self._n_loops = n_loops
        self._assumed_height_m = float(assumed_athlete_height_m)
        self._tracker = ByteTrackTracker(
            classes=[PERSON_CLASS_ID, SPORTS_BALL_CLASS_ID],
            confidence=0.10,
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

        # === PASS 1 ===
        track_history_raw: dict[
            int, list[tuple[int, float, float, float, float]]
        ] = {}
        people_per_frame: dict[int, list[TrackedDetection]] = {}
        balls_per_frame: dict[int, list[TrackedDetection]] = {}
        cone_detections_per_frame: list[tuple[int, float, float]] = []
        n_frames = 0
        motion = CameraMotion()

        for frame in frame_iter(video_path):
            n_frames += 1
            tracked = self._tracker.update(frame.image)
            person_bboxes_this_frame: list[np.ndarray] = []
            for p in tracked:
                if p.class_id == PERSON_CLASS_ID:
                    person_bboxes_this_frame.append(p.bbox_xyxy)
                    track_history_raw.setdefault(p.track_id, []).append(
                        (frame.idx, float(p.center[0]), float(p.center[1]),
                         p.height, p.width)
                    )
                    people_per_frame.setdefault(frame.idx, []).append(p)
                elif p.class_id == SPORTS_BALL_CLASS_ID:
                    balls_per_frame.setdefault(frame.idx, []).append(p)
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
            motion.update(
                frame.idx, frame.image,
                exclude_bboxes_xyxy=person_bboxes_this_frame,
            )

        # === Stabilize to frame-0 coords ===
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
        print(f"[fig-of-8] {len(cone_positions)} cone clusters detected")

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
                "could not identify a single player track — "
                "neither pixel-area dominance nor cone-proximity fallback "
                "yielded a winner"
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
            cone_spacing_m=_CONE_SPACING_M,
            baseline_bbox_h_px=baseline_bbox_h,
            assumed_height_m=self._assumed_height_m,
        )
        print(f"[fig-of-8] calibration: {px_per_m:.1f} px/m ({calibration_source})")

        # === PASS 2 ===
        player_bboxes_by_frame = {
            fi: next(
                (p.bbox_xyxy for p in people_per_frame.get(fi, [])
                 if p.track_id == player_track_id),
                None,
            )
            for fi in range(n_frames)
        }
        pose_results: dict[int, object] = {}
        touches: list[_Touch] = []
        last_touch_frame = -10**6
        debounce_frames = int(round(_TOUCH_DEBOUNCE_S * fps))
        ankle_xy_stab_by_frame: dict[int, tuple[float, float]] = {}

        for frame in frame_iter(video_path):
            bbox = player_bboxes_by_frame.get(frame.idx)
            if bbox is None or frame.idx % _POSE_INTERVAL_FRAMES != 0:
                continue
            pose = self._pose.estimate_bbox(frame.image, bbox)
            pose_results[frame.idx] = pose
            ankle_pixel = _ankle_midpoint_pixel(pose)
            if ankle_pixel is not None:
                ankle_xy_stab_by_frame[frame.idx] = motion.transform_point(
                    frame.idx, ankle_pixel
                )
            balls = balls_per_frame.get(frame.idx, [])
            ball = _pick_ball_near(balls, bbox)
            if ball is None or pose is None:
                continue
            if (frame.idx - last_touch_frame) < debounce_frames:
                continue
            if not (run_window.start_frame <= frame.idx <= run_window.stop_frame):
                continue
            side = _detect_touch(ball, pose, bbox)
            if side is not None:
                touches.append(_Touch(frame_idx=frame.idx, side=side))
                last_touch_frame = frame.idx

        # === Loop counting ===
        loops_completed = _count_loops(
            ankle_traj=[
                (fi, x, y) for fi, (x, y) in ankle_xy_stab_by_frame.items()
                if run_window.start_frame <= fi <= run_window.stop_frame
            ],
            cone_pair=cone_pair,
        )

        metrics = _compute_metrics(
            run_window=run_window, fps=fps,
            track_history=track_history[player_track_id],
            px_per_m=px_per_m, touches=touches,
        )
        scores, test_score = score_test(metrics, self.test_id, athlete.gender)

        # === PASS 3 ===
        writer = cv2.VideoWriter(
            str(out_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (info.width, info.height),
        )
        if not writer.isOpened():
            raise ProtocolError(f"could not open VideoWriter at {out_path}")
        last_pose = None
        try:
            for frame in frame_iter(video_path):
                img = frame.image
                bbox = player_bboxes_by_frame.get(frame.idx)
                if bbox is not None:
                    pose = pose_results.get(frame.idx)
                    if pose is not None:
                        last_pose = pose
                    draw_bbox(img, bbox)
                    if last_pose is not None:
                        draw_skeleton(img, last_pose.keypoints)
                ball = _pick_ball_near(
                    balls_per_frame.get(frame.idx, []), bbox,
                )
                if ball is not None:
                    draw_bbox(img, ball.bbox_xyxy, color=BALL)
                draw_hud(img, _hud_fields(
                    frame_idx=frame.idx, fps=fps, run=run_window,
                    loops_completed=loops_completed, n_loops=self._n_loops,
                    n_touches=len([t for t in touches if t.frame_idx <= frame.idx]),
                ))
                writer.write(img)

            endcard_rows = [
                (mid.replace("_", " "), f"{mv.raw:.3f} {mv.unit}",
                 int(round(scores[mid].score)) if mid in scores else 0)
                for mid, mv in metrics.items()
            ]
            endcard_rows.append(
                ("Loops Completed", f"{loops_completed}/{self._n_loops}", 0)
            )
            endcard = render_endcard(
                title="Figure of 8 Dribbling",
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


# --- Cone helpers ------------------------------------------------------


def _pick_two_cones(
    clusters: list[tuple[float, float]],
) -> tuple[tuple[float, float], tuple[float, float]] | None:
    """Pick the pair with maximum x-separation. Returns None when there
    are fewer than 2 clusters."""
    if len(clusters) < 2:
        return None
    if len(clusters) == 2:
        a, b = sorted(clusters, key=lambda c: c[0])
        return a, b
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


# --- Loop counting -----------------------------------------------------


def _count_loops(
    *,
    ankle_traj: list[tuple[int, float, float]],
    cone_pair: tuple[tuple[float, float], tuple[float, float]] | None,
) -> int:
    """Count completed figure-8 loops via per-cone winding number.

    For each cone, integrate the angular position of the ankle relative
    to that cone across the trajectory; the magnitude divided by 2π is
    the number of full revolutions. A figure-8 loop = one full
    revolution around each cone, so loops_completed = the minimum of
    the two windings (the athlete must enclose BOTH cones to complete
    a loop).

    Returns 0 if cones aren't available or trajectory is too short.
    """
    if cone_pair is None or len(ankle_traj) < 5:
        return 0

    def _winding(cone: tuple[float, float]) -> float:
        cx, cy = cone
        total = 0.0
        prev: float | None = None
        for _, x, y in ankle_traj:
            ang = float(np.arctan2(y - cy, x - cx))
            if prev is not None:
                d = ang - prev
                if d > np.pi:
                    d -= 2 * np.pi
                elif d < -np.pi:
                    d += 2 * np.pi
                total += d
            prev = ang
        return total

    w_a = abs(_winding(cone_pair[0])) / (2 * float(np.pi))
    w_b = abs(_winding(cone_pair[1])) / (2 * float(np.pi))
    # Discrete revolutions floored from the smaller of the two windings
    # (the athlete must enclose BOTH cones to count a loop). The 0.05
    # tolerance is for floating-point accumulation across the trajectory:
    # a real 2-loop athlete shouldn't lose a count to numerical noise.
    return int(min(w_a, w_b) + 0.05)


# --- Pose / ball helpers ----------------------------------------------


def _ankle_midpoint_pixel(pose) -> tuple[float, float] | None:
    if pose is None:
        return None
    pts: list[tuple[float, float]] = []
    for kp in ("left_ankle", "right_ankle"):
        if pose.confidence_of(kp) >= _POSE_CONF_MIN:
            pos = pose.position(kp)
            pts.append((float(pos[0]), float(pos[1])))
    if not pts:
        return None
    return (
        sum(p[0] for p in pts) / len(pts),
        sum(p[1] for p in pts) / len(pts),
    )


def _pick_ball_near(
    balls: list[TrackedDetection],
    runner_bbox: np.ndarray | None,
) -> TrackedDetection | None:
    if not balls:
        return None
    if runner_bbox is None:
        return balls[0]
    ax = float((runner_bbox[0] + runner_bbox[2]) / 2.0)
    ay = float((runner_bbox[1] + runner_bbox[3]) / 2.0)
    return min(
        balls,
        key=lambda b: (b.center[0] - ax) ** 2 + (b.center[1] - ay) ** 2,
    )


def _detect_touch(
    ball: TrackedDetection, pose, runner_bbox: np.ndarray,
) -> Literal["L", "R"] | None:
    bx, by = ball.center
    bbox_h = float(runner_bbox[3] - runner_bbox[1])
    threshold = _TOUCH_PROXIMITY_FRAC * bbox_h
    best_x: float | None = None
    best_d = float("inf")
    for kp_name in ("left_ankle", "right_ankle"):
        if pose.confidence_of(kp_name) < _POSE_CONF_MIN:
            continue
        kp = pose.position(kp_name)
        d = float(np.hypot(bx - kp[0], by - kp[1]))
        if d < threshold and d < best_d:
            best_d = d
            best_x = float(kp[0])
    if best_x is None:
        return None
    bcx = body_center_x(pose)
    if bcx is None:
        return "R" if best_x < bx else "L"
    return ankle_side(best_x, bcx)


# --- Metrics + HUD ----------------------------------------------------


def _compute_metrics(
    *,
    run_window: _RunWindow,
    fps: float,
    track_history: list[tuple[int, float, float, float, float]],
    px_per_m: float,
    touches: list[_Touch],
) -> dict[str, MetricValue]:
    duration_s = run_window.duration_frames / fps
    in_window = [
        (cx, cy) for (fi, cx, cy, _, _) in track_history
        if run_window.start_frame <= fi <= run_window.stop_frame
    ]
    centers_m = np.asarray(in_window, dtype=float) / px_per_m
    if len(centers_m) >= 2:
        diffs = np.diff(centers_m, axis=0)
        total_distance_m = float(np.linalg.norm(diffs, axis=1).sum())
    else:
        total_distance_m = 0.0
    avg_speed = (
        average_speed_ms(total_distance_m, duration_s)
        if duration_s > 0 else 0.0
    )

    n_touches = len(touches)
    tpm = (
        touches_per_metre(n_touches, total_distance_m)
        if total_distance_m > 0 else 0.0
    )
    leg_touches = [t for t in touches if t.side in ("L", "R")]
    left_pct = (
        sum(1 for t in leg_touches if t.side == "L") / len(leg_touches) * 100.0
        if leg_touches else 0.0
    )

    return {
        "total_completion_time_s": MetricValue(raw=duration_s, unit="s"),
        "total_distance_m": MetricValue(raw=total_distance_m, unit="m"),
        "average_speed_ms": MetricValue(raw=avg_speed, unit="m_per_s"),
        "total_ball_touches": MetricValue(raw=float(n_touches), unit="count"),
        "touches_per_metre": MetricValue(raw=tpm, unit="count_per_m"),
        "left_leg_utilisation_pct": MetricValue(raw=left_pct, unit="percent"),
    }


def _hud_fields(
    *,
    frame_idx: int, fps: float, run: _RunWindow,
    loops_completed: int, n_loops: int, n_touches: int,
) -> dict[str, str]:
    if frame_idx < run.start_frame:
        return {
            "phase": "ready", "elapsed": "-",
            "loops": f"0/{n_loops}", "touches": "0",
        }
    if frame_idx <= run.stop_frame:
        elapsed_s = (frame_idx - run.start_frame) / fps
        return {
            "phase": "running",
            "elapsed": f"{elapsed_s:.2f} s",
            "loops": f"{loops_completed}/{n_loops}",
            "touches": str(n_touches),
        }
    total_s = run.duration_frames / fps
    return {
        "phase": "finished",
        "elapsed": f"{total_s:.3f} s",
        "loops": f"{loops_completed}/{n_loops}",
        "touches": str(n_touches),
    }
