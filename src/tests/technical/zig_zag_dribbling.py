"""Zig-Zag Dribbling pipeline.

Athlete dribbles a ball through a slalom of cones (5-7 markers spaced
~2 m apart), weaves both directions, returns through the finish gate.
Single scored metric: `total_completion_time_s`.

Three-pass design with camera-motion compensation:

- Pass 1: ByteTrack person + ball; sample cone detections at stride;
  estimate frame-to-frame camera transform via shared CameraMotion
  (Lucas-Kanade + ORB anchor). Stabilize all positions into frame-0
  coordinates so per-rep / cone-passage analysis is panning-camera-
  agnostic.
- Pass 2: pose on the picked player only; collect ankle midpoints in
  stabilized coordinates for touch detection and trajectory analysis.
- Pass 3: render annotated video using cached pose results.

Calibration: cone-pair distance / 2 m if at least two horizontally-
aligned cones cluster cleanly; otherwise body-height proxy
(baseline bbox-h / 1.70 m).

v1 ships:
  - total_completion_time_s (scored)
  - total_ball_touches, touches_per_metre, left_leg_utilisation_pct
  - average_speed_ms, total_distance_m
  - cones_passed (informational HUD counter)

Deferred to a follow-up phase (need slalom-side analysis):
  - cone_miss_events
  - avg_cod_angle_deg
  - max_speed_ms, peak_acceleration_ms2, peak_deceleration_ms2
    (same bbox-jitter caveat as 5x10m — needs pose-mid-hip tracking)
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

# Run window — slowest realistic zig-zag at 5-cone slalom: ~15 s. Below
# 5 s = implausible.
_MIN_RUN_FRAMES = 150          # 5 s @ 30 fps

_MIN_TRACK_HISTORY_FRAMES = 60
_POSE_INTERVAL_FRAMES = 3
_ENDCARD_HOLD_S = 2.5

# Marker prompts — same setup as 5x10m: yellow poles + disks.
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

# Cone spacing per protocol (m). 5-cone slalom at 2 m apart -> 8 m
# end-to-end; total dribbled distance ~16 m (out + back).
_CONE_SPACING_M = 2.0
_DEFAULT_ATHLETE_HEIGHT_M = 1.70
_TELEPORT_FRAC = 5.0  # single-athlete clean tracking; disable break

# Touch detection — same shape as juggling/dribbling.
_TOUCH_PROXIMITY_FRAC = 0.30
_TOUCH_DEBOUNCE_S = 0.20
_POSE_CONF_MIN = 0.30

# Cone passage: athlete is "at" a cone when ankle is within this
# fraction of bbox-h from it (in stabilized coords).
_CONE_PASS_PROXIMITY_FRAC = 0.50


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


class ZigZagDribblingTest(BaseTest):
    """Zig-Zag Dribbling: 3-pass pipeline with camera-motion + cones."""

    test_id = "zig-zag-dribbling"

    def __init__(
        self,
        *,
        n_cones: int = 5,
        assumed_athlete_height_m: float = _DEFAULT_ATHLETE_HEIGHT_M,
    ) -> None:
        if n_cones < 2:
            raise ValueError(f"n_cones must be >= 2, got {n_cones}")
        self._n_cones = n_cones
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

        # === PASS 1: detect+track + cones + camera motion ===
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
        print(f"[zig-zag] {len(cone_positions)} cone clusters detected")

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

        # === Run window from picked player's stabilized trajectory ===
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

        # === Calibration: cone-pair if available, else body-height proxy ===
        baseline_bbox_h = float(np.median(
            [h for (_, _, _, h, _) in track_history[player_track_id]]
        ))
        px_per_m, calibration_source = _calibrate(
            cones=cone_positions,
            cone_spacing_m=_CONE_SPACING_M,
            baseline_bbox_h_px=baseline_bbox_h,
            assumed_height_m=self._assumed_height_m,
        )
        print(f"[zig-zag] calibration: {px_per_m:.1f} px/m ({calibration_source})")

        # === PASS 2: pose + touches on picked player only ===
        player_bboxes_by_frame: dict[int, np.ndarray] = {
            t.track_id: None for t in []  # placeholder; populate below
        }
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
            if bbox is None:
                continue
            if frame.idx % _POSE_INTERVAL_FRAMES != 0:
                continue
            pose = self._pose.estimate_bbox(frame.image, bbox)
            pose_results[frame.idx] = pose
            ankle_pixel = _ankle_midpoint_pixel(pose)
            if ankle_pixel is not None:
                ankle_xy_stab_by_frame[frame.idx] = motion.transform_point(
                    frame.idx, ankle_pixel
                )
            balls = balls_per_frame.get(frame.idx, [])
            ball = _pick_ball(balls, runner=_runner_from_bbox(bbox))
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

        # === Compute metrics ===
        cones_passed = _count_cone_passages(
            ankle_traj=[
                (fi, x, y) for fi, (x, y) in ankle_xy_stab_by_frame.items()
                if run_window.start_frame <= fi <= run_window.stop_frame
            ],
            cones=cone_positions,
            baseline_bbox_h_px=baseline_bbox_h,
        )
        metrics = _compute_metrics(
            run_window=run_window,
            fps=fps,
            track_history=track_history[player_track_id],
            px_per_m=px_per_m,
            touches=touches,
        )
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
                # Ball highlight if detected this frame
                ball = _pick_ball(
                    balls_per_frame.get(frame.idx, []),
                    runner=_runner_from_bbox(bbox) if bbox is not None else None,
                )
                if ball is not None:
                    draw_bbox(img, ball.bbox_xyxy, color=BALL)
                draw_hud(img, _hud_fields(
                    frame_idx=frame.idx,
                    fps=fps,
                    run=run_window,
                    cones_passed=cones_passed,
                    n_cones=self._n_cones,
                    n_touches=len([t for t in touches if t.frame_idx <= frame.idx]),
                ))
                writer.write(img)

            endcard_rows = [
                (mid.replace("_", " "), f"{mv.raw:.3f} {mv.unit}",
                 int(round(scores[mid].score)) if mid in scores else 0)
                for mid, mv in metrics.items()
            ]
            endcard_rows.append(("Cones Passed", f"{cones_passed}/{self._n_cones}", 0))
            endcard = render_endcard(
                title="Zig-Zag Dribbling",
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


# --- Calibration -------------------------------------------------------


def _calibrate(
    *,
    cones: list[tuple[float, float]],
    cone_spacing_m: float,
    baseline_bbox_h_px: float,
    assumed_height_m: float,
) -> tuple[float, str]:
    """Cone-pair calibration when 2+ horizontally-aligned cones, else
    body-height proxy. Returns (px_per_m, source)."""
    if len(cones) >= 2:
        sorted_x = sorted(c[0] for c in cones)
        # Use median of consecutive-pair gaps (robust to outliers).
        gaps = [sorted_x[i + 1] - sorted_x[i] for i in range(len(sorted_x) - 1)]
        gaps = [g for g in gaps if g > 0]
        if gaps:
            median_gap_px = float(np.median(gaps))
            return median_gap_px / cone_spacing_m, "cone-pair"
    if baseline_bbox_h_px > 0:
        return baseline_bbox_h_px / assumed_height_m, "body-height-proxy"
    return 1.0, "no-calibration"


# --- Helpers -----------------------------------------------------------


def _ankle_midpoint_pixel(pose) -> tuple[float, float] | None:
    if pose is None:
        return None
    pts: list[tuple[float, float]] = []
    for kp_name in ("left_ankle", "right_ankle"):
        if pose.confidence_of(kp_name) >= _POSE_CONF_MIN:
            pos = pose.position(kp_name)
            pts.append((float(pos[0]), float(pos[1])))
    if not pts:
        return None
    return (
        sum(p[0] for p in pts) / len(pts),
        sum(p[1] for p in pts) / len(pts),
    )


def _pick_ball(
    balls: list[TrackedDetection],
    runner: TrackedDetection | None,
) -> TrackedDetection | None:
    if not balls:
        return None
    if runner is None:
        return balls[0]
    ax, ay = runner.center
    return min(
        balls,
        key=lambda b: (b.center[0] - ax) ** 2 + (b.center[1] - ay) ** 2,
    )


def _runner_from_bbox(bbox: np.ndarray | None) -> TrackedDetection | None:
    """Build a minimal TrackedDetection-like wrapper around a bbox so
    `_pick_ball` (which keys on `.center`) works without re-querying.
    """
    if bbox is None:
        return None
    cx = float((bbox[0] + bbox[2]) / 2.0)
    cy = float((bbox[1] + bbox[3]) / 2.0)

    @dataclass
    class _Runner:
        bbox_xyxy: np.ndarray
        @property
        def center(self) -> tuple[float, float]:
            return (cx, cy)
        @property
        def height(self) -> float:
            return float(bbox[3] - bbox[1])
    return _Runner(bbox_xyxy=bbox)  # type: ignore[return-value]


def _detect_touch(
    ball: TrackedDetection,
    pose,
    runner_bbox: np.ndarray,
) -> Literal["L", "R"] | None:
    """Closest in-zone ankle to the ball; side from image-x relative to
    body center (robust to pose-model L/R label flips)."""
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


def _count_cone_passages(
    *,
    ankle_traj: list[tuple[int, float, float]],
    cones: list[tuple[float, float]],
    baseline_bbox_h_px: float,
) -> int:
    """Count cones the athlete passed near at least once during the run.

    A pass = ankle came within `_CONE_PASS_PROXIMITY_FRAC * bbox_h` of
    the cone in stabilized coords. Each cone is counted at most once.
    """
    if not ankle_traj or not cones:
        return 0
    threshold = _CONE_PASS_PROXIMITY_FRAC * baseline_bbox_h_px
    passed = 0
    for cx_c, cy_c in cones:
        for _, ax, ay in ankle_traj:
            if (ax - cx_c) ** 2 + (ay - cy_c) ** 2 < threshold * threshold:
                passed += 1
                break
    return passed


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
    avg_speed = average_speed_ms(total_distance_m, duration_s) if duration_s > 0 else 0.0

    n_touches = len(touches)
    tpm = touches_per_metre(n_touches, total_distance_m) if total_distance_m > 0 else 0.0
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
    frame_idx: int,
    fps: float,
    run: _RunWindow,
    cones_passed: int,
    n_cones: int,
    n_touches: int,
) -> dict[str, str]:
    if frame_idx < run.start_frame:
        return {
            "phase": "ready",
            "elapsed": "-",
            "cones": f"0/{n_cones}",
            "touches": "0",
        }
    if frame_idx <= run.stop_frame:
        elapsed_s = (frame_idx - run.start_frame) / fps
        return {
            "phase": "running",
            "elapsed": f"{elapsed_s:.2f} s",
            "cones": f"{cones_passed}/{n_cones}",
            "touches": str(n_touches),
        }
    total_s = run.duration_frames / fps
    return {
        "phase": "finished",
        "elapsed": f"{total_s:.3f} s",
        "cones": f"{cones_passed}/{n_cones}",
        "touches": str(n_touches),
    }
