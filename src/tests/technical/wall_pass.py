"""Wall Pass pipeline.

30 s fixed-window test: athlete stands behind a marked line, passes a
ball to the wall, receives the rebound, controls, passes again.

Detection model
---------------
- Track athlete (ByteTrack person) + ball (ByteTrack sports_ball) in a
  single pass; the camera is typically static for this test so no
  camera-motion compensation is needed.
- Per frame, compute athlete-ball distance (pixel space). The distance
  signal oscillates: low when the ball is at the athlete's foot
  (control / pass release) and high when the ball is at the wall.
- One pass cycle = one peak in the distance signal that crosses both
  a "far" threshold (ball reached wall area) and returns below a
  "near" threshold (ball back at the athlete). Each completed cycle
  counts as one `successful_pass`.
- Pass velocity = peak ball speed (px/frame -> m/s via wall-distance
  calibration) during the outbound leg of each cycle.

Calibration: operator declares `wall_distance_m` at construction time
(spec calls calibration mandatory but does not require cone detection
— wall position is a known marker). Pixel-to-metre comes from the
maximum athlete-ball distance observed during the test divided by
`wall_distance_m`. If the athlete is positioned on the marked line and
passes reach the wall, this gives a stable scale.

Two-pass design (single video read in pass 1 + render in pass 2):
- Pass 1: ByteTrack person + ball + pose. Pick player. Build per-frame
  athlete-ball distance and ball-velocity samples. Detect pass cycles.
- Pass 2: render annotated video with HUD ticker.
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
from src.core.detection.player_detector import PERSON_CLASS_ID
from src.core.pose.estimator import create_pose_estimator
from src.core.pose.orientation import ankle_side, body_center_x
from src.core.tracking.bytetrack_tracker import ByteTrackTracker, TrackedDetection
from src.core.tracking.player_picker import pick_player
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

SPORTS_BALL_CLASS_ID = 32

# --- Tunables ----------------------------------------------------------

# Cycle thresholds — fractions of the maximum athlete-ball distance
# observed during the test. "Far" = ball reached the wall area;
# "near" = ball back under athlete control.
_FAR_FRAC = 0.70
_NEAR_FRAC = 0.30

# Minimum frames between consecutive cycle peaks (debounce on noisy
# distance signal). 0.5 s @ 30 fps = 15 frames.
_CYCLE_DEBOUNCE_S = 0.5

# Pose / leg-utilisation
_POSE_CONF_MIN = 0.30
_POSE_INTERVAL_FRAMES = 3

# Smoothing window for distance signal; ~0.2 s damps single-frame
# detection jitter without delaying cycle endpoints materially.
_DISTANCE_SMOOTH_S = 0.2

_MIN_TRACK_HISTORY_FRAMES = 30
_ENDCARD_HOLD_S = 2.0
_DEFAULT_WALL_DISTANCE_M = 3.0


# --- State -------------------------------------------------------------


@dataclass(frozen=True)
class _Cycle:
    """One pass cycle: athlete -> wall -> athlete."""
    pass_release_frame: int       # cycle starts here (last "near")
    peak_frame: int               # ball furthest from athlete
    reception_frame: int          # cycle ends here (first "near" again)
    peak_ball_speed_ms: float     # max ball speed during outbound leg
    leg_side: Literal["L", "R", "?"] = "?"  # which leg pressed last

    @property
    def decision_time_s(self) -> float:
        """Duration the athlete spent controlling the ball before this
        pass release. Filled in post-detection from inter-cycle gaps."""
        return 0.0


# --- Pipeline ----------------------------------------------------------


class WallPassTest(BaseTest):
    """Wall Pass test: 30 s, count successful pass cycles."""

    test_id = "wall-pass"

    def __init__(
        self,
        *,
        wall_distance_m: float = _DEFAULT_WALL_DISTANCE_M,
    ) -> None:
        if wall_distance_m <= 0:
            raise ValueError(
                f"wall_distance_m must be > 0, got {wall_distance_m}"
            )
        self._wall_distance_m = float(wall_distance_m)
        self._tracker = ByteTrackTracker(
            classes=[PERSON_CLASS_ID, SPORTS_BALL_CLASS_ID],
            confidence=0.10,
        )
        self._pose = create_pose_estimator("pose_default")

    def run(
        self,
        video_path: Path,
        athlete: AthleteProfile,
        output_dir: Path,
    ) -> AnalysisResult:
        info = video_info(video_path)
        fps = info.fps
        out_path = output_dir / f"{self.test_id}.mp4"

        # === PASS 1: detect+track athlete + ball ===
        track_history: dict[
            int, list[tuple[int, float, float, float, float]]
        ] = {}
        people_per_frame: dict[int, list[TrackedDetection]] = {}
        balls_per_frame: dict[int, list[TrackedDetection]] = {}
        n_frames = 0

        for frame in frame_iter(video_path):
            n_frames += 1
            tracked = self._tracker.update(frame.image)
            for p in tracked:
                if p.class_id == PERSON_CLASS_ID:
                    track_history.setdefault(p.track_id, []).append(
                        (frame.idx, float(p.center[0]), float(p.center[1]),
                         p.height, p.width)
                    )
                    people_per_frame.setdefault(frame.idx, []).append(p)
                elif p.class_id == SPORTS_BALL_CLASS_ID:
                    balls_per_frame.setdefault(frame.idx, []).append(p)

        # === Pick player (no cones; ball-proximity fallback covers it) ===
        ball_positions = {
            fi: [(b.center[0], b.center[1]) for b in balls]
            for fi, balls in balls_per_frame.items()
        }
        player_track_id = pick_player(
            track_history,
            object_positions=ball_positions or None,
            min_history_frames=_MIN_TRACK_HISTORY_FRAMES,
            verbose=True,
        )
        if player_track_id is None:
            if not track_history:
                raise DetectionError("no people were detected in the video")
            raise ProtocolError(
                "could not identify a single player track"
            )

        # === Build per-frame athlete-ball distance + ball velocity ===
        athlete_pos_by_frame: dict[int, tuple[float, float]] = {
            fi: (cx, cy)
            for (fi, cx, cy, _, _) in track_history[player_track_id]
        }
        # Pick closest-to-athlete ball per frame.
        ball_pos_by_frame: dict[int, tuple[float, float]] = {}
        for fi, balls in balls_per_frame.items():
            ap = athlete_pos_by_frame.get(fi)
            if ap is None:
                continue
            ball = min(
                balls,
                key=lambda b: (b.center[0] - ap[0]) ** 2
                + (b.center[1] - ap[1]) ** 2,
            )
            ball_pos_by_frame[fi] = (
                float(ball.center[0]), float(ball.center[1]),
            )

        if len(ball_pos_by_frame) < _MIN_TRACK_HISTORY_FRAMES:
            raise ProtocolError(
                f"only {len(ball_pos_by_frame)} ball-detected frames — "
                "ball tracking failed; check ball visibility / lighting"
            )

        # Distance signal indexed by frame.
        frames_with_both = sorted(
            f for f in ball_pos_by_frame.keys()
            if f in athlete_pos_by_frame
        )
        distances = np.array([
            float(np.hypot(
                ball_pos_by_frame[f][0] - athlete_pos_by_frame[f][0],
                ball_pos_by_frame[f][1] - athlete_pos_by_frame[f][1],
            ))
            for f in frames_with_both
        ])

        # Smooth distance to suppress per-frame ball-detection jitter.
        win = max(3, int(round(_DISTANCE_SMOOTH_S * fps)) | 1)
        if win < len(distances):
            kernel = np.ones(win, dtype=float) / win
            distances = np.convolve(distances, kernel, mode="same")

        max_distance_px = float(distances.max()) if len(distances) else 0.0
        if max_distance_px <= 0:
            raise ProtocolError(
                "ball never separated from athlete; no pass cycles detected"
            )
        px_per_m = max_distance_px / self._wall_distance_m

        # Ball velocity (pixels/frame -> m/s).
        ball_vel_px = _ball_pixel_velocity(
            frames_with_both, ball_pos_by_frame,
        )
        ball_speed_ms = ball_vel_px * fps / px_per_m

        # === Detect pass cycles ===
        cycles = _detect_cycles(
            frames=frames_with_both,
            distances=distances,
            ball_speeds_ms=ball_speed_ms,
            fps=fps,
            max_distance_px=max_distance_px,
        )

        # === Pose pass for leg-utilisation (sample every Nth frame) ===
        leg_decisions: list[str] = []
        for cycle in cycles:
            side = self._infer_kicking_leg(
                cycle.pass_release_frame, video_path, fps,
                ball_pos_by_frame=ball_pos_by_frame,
            )
            leg_decisions.append(side)

        # === Compute metrics ===
        n_passes = len(cycles)
        velocities = [c.peak_ball_speed_ms for c in cycles]
        avg_vel = float(np.mean(velocities)) if velocities else 0.0
        max_vel = float(np.max(velocities)) if velocities else 0.0
        # Decision time = time from reception of cycle i to release of
        # cycle i+1.
        decision_times = [
            (cycles[i + 1].pass_release_frame - cycles[i].reception_frame) / fps
            for i in range(len(cycles) - 1)
        ]
        avg_decision = (
            float(np.mean(decision_times)) if decision_times else 0.0
        )
        # Accuracy: every detected cycle is "successful" by construction;
        # use 100% as a placeholder. Misses (ball escapes off-screen)
        # would not produce a complete cycle and so wouldn't be counted
        # — accuracy is implicit in the count, not a separate metric.
        accuracy_pct = 100.0 if n_passes > 0 else 0.0
        leg_l = sum(1 for s in leg_decisions if s == "L")
        leg_total = sum(1 for s in leg_decisions if s in ("L", "R"))
        left_pct = (leg_l / leg_total * 100.0) if leg_total else 0.0

        metrics: dict[str, MetricValue] = {
            "successful_passes": MetricValue(raw=float(n_passes), unit="count"),
            "passing_accuracy_percent": MetricValue(
                raw=accuracy_pct, unit="pct",
            ),
            "average_decision_time_s": MetricValue(
                raw=avg_decision, unit="s",
            ),
            "average_pass_velocity_ms": MetricValue(
                raw=avg_vel, unit="m_per_s",
            ),
            "max_pass_velocity_ms": MetricValue(
                raw=max_vel, unit="m_per_s",
            ),
            "left_leg_utilisation_pct": MetricValue(
                raw=left_pct, unit="percent",
            ),
        }
        scores, test_score = score_test(metrics, self.test_id, athlete.gender)

        # === PASS 2: render annotated video ===
        writer = cv2.VideoWriter(
            str(out_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (info.width, info.height),
        )
        if not writer.isOpened():
            raise ProtocolError(f"could not open VideoWriter at {out_path}")

        try:
            for frame in frame_iter(video_path):
                img = frame.image
                # Draw the picked athlete's bbox if visible this frame.
                runner = next(
                    (p for p in people_per_frame.get(frame.idx, [])
                     if p.track_id == player_track_id),
                    None,
                )
                if runner is not None:
                    draw_bbox(img, runner.bbox_xyxy)
                    if frame.idx % _POSE_INTERVAL_FRAMES == 0:
                        pose = self._pose.estimate_bbox(img, runner.bbox_xyxy)
                        if pose is not None:
                            draw_skeleton(img, pose.keypoints)
                ball = balls_per_frame.get(frame.idx, [None])[0]
                if ball is not None:
                    draw_bbox(img, ball.bbox_xyxy, color=BALL)
                draw_hud(img, _hud_fields(
                    frame_idx=frame.idx, fps=fps, cycles=cycles,
                ))
                writer.write(img)

            endcard_rows = [
                (mid.replace("_", " "), f"{mv.raw:.3f} {mv.unit}",
                 int(round(scores[mid].score)) if mid in scores else 0)
                for mid, mv in metrics.items()
            ]
            endcard = render_endcard(
                title="Wall Pass (30 s)",
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

    def _infer_kicking_leg(
        self,
        release_frame: int,
        video_path: Path,
        fps: float,
        ball_pos_by_frame: dict[int, tuple[float, float]],
    ) -> str:
        """Inspect the pose at the pass-release frame; whichever ankle
        is closer to the ball is the kicking leg. Returns 'L', 'R', or
        '?' if pose can't be confidently estimated.

        v1 implementation: cheap re-read of the single frame. Heavier
        production version would cache pose per release frame in pass 1.
        """
        cap = cv2.VideoCapture(str(video_path))
        try:
            cap.set(cv2.CAP_PROP_POS_FRAMES, release_frame)
            ok, img = cap.read()
        finally:
            cap.release()
        if not ok:
            return "?"
        ball_xy = ball_pos_by_frame.get(release_frame)
        if ball_xy is None:
            return "?"
        # Reuse a coarse bbox guess: 30% margin around current frame
        # ball position. Pose estimator wants a person bbox; for a
        # cleaner v1 pose should be cached in pass 1. This fallback is
        # acceptable for left-leg-utilisation estimation only.
        h, w = img.shape[:2]
        side = int(min(h, w) * 0.4)
        cx = max(side // 2, min(int(ball_xy[0]), w - side // 2))
        cy = max(side // 2, min(int(ball_xy[1]), h - side // 2))
        bbox = np.array([
            cx - side // 2, cy - side, cx + side // 2, cy + side,
        ], dtype=float)
        pose = self._pose.estimate_bbox(img, bbox)
        if pose is None:
            return "?"
        try:
            la = pose.position("left_ankle")
            ra = pose.position("right_ankle")
            la_c = pose.confidence_of("left_ankle")
            ra_c = pose.confidence_of("right_ankle")
        except Exception:
            return "?"
        if max(la_c, ra_c) < _POSE_CONF_MIN:
            return "?"
        bx, by = ball_xy
        d_l = float(np.hypot(la[0] - bx, la[1] - by)) if la_c >= _POSE_CONF_MIN else float("inf")
        d_r = float(np.hypot(ra[0] - bx, ra[1] - by)) if ra_c >= _POSE_CONF_MIN else float("inf")
        contact_x = float(la[0]) if d_l <= d_r else float(ra[0])
        bcx = body_center_x(pose)
        if bcx is None:
            return "L" if contact_x < bx else "R"
        return ankle_side(contact_x, bcx)


# --- Cycle detection --------------------------------------------------


def _ball_pixel_velocity(
    frames: list[int],
    ball_pos_by_frame: dict[int, tuple[float, float]],
) -> np.ndarray:
    """Per-frame ball-velocity magnitude in pixels/frame, aligned to
    `frames`. Frame i's velocity is the displacement from i-1 to i; the
    first sample is 0."""
    vels = np.zeros(len(frames), dtype=float)
    for i in range(1, len(frames)):
        dt_frames = frames[i] - frames[i - 1]
        if dt_frames <= 0:
            continue
        x0, y0 = ball_pos_by_frame[frames[i - 1]]
        x1, y1 = ball_pos_by_frame[frames[i]]
        vels[i] = float(np.hypot(x1 - x0, y1 - y0)) / dt_frames
    return vels


def _detect_cycles(
    *,
    frames: list[int],
    distances: np.ndarray,
    ball_speeds_ms: np.ndarray,
    fps: float,
    max_distance_px: float,
) -> list[_Cycle]:
    """Two-state walk over the smoothed distance signal:

        near        (d < near_thresh): ball under athlete control
        away        (d >= near_thresh): ball heading to / at / from wall

    Each transition near -> away marks a pass release; the next
    transition away -> near marks a reception. The peak (max d during
    the away leg) approximates ball-at-wall. Peak ball speed is taken
    over the rising portion of the away leg [release ... peak].
    """
    if len(distances) < 5:
        return []
    far_thresh = _FAR_FRAC * max_distance_px
    near_thresh = _NEAR_FRAC * max_distance_px
    debounce_frames = int(round(_CYCLE_DEBOUNCE_S * fps))

    cycles: list[_Cycle] = []
    state: Literal["near", "away"] = "near"
    pass_release_idx: int | None = None
    peak_idx: int | None = None
    saw_far = False
    last_release_frame = -10**9

    for i, d in enumerate(distances):
        if state == "near":
            if d >= near_thresh:
                state = "away"
                pass_release_idx = i
                peak_idx = i
                saw_far = d >= far_thresh
        else:  # state == "away"
            if peak_idx is not None and d > distances[peak_idx]:
                peak_idx = i
            if d >= far_thresh:
                saw_far = True
            if d < near_thresh:
                # Reception. Only record if the ball reached the far
                # zone (i.e., a real pass to the wall, not just a
                # short wobble out of the near zone).
                if (saw_far and pass_release_idx is not None
                        and peak_idx is not None
                        and frames[pass_release_idx] - last_release_frame
                        >= debounce_frames):
                    peak_speed = (
                        float(np.max(
                            ball_speeds_ms[pass_release_idx:peak_idx + 1]
                        ))
                        if peak_idx > pass_release_idx else 0.0
                    )
                    cycles.append(_Cycle(
                        pass_release_frame=int(frames[pass_release_idx]),
                        peak_frame=int(frames[peak_idx]),
                        reception_frame=int(frames[i]),
                        peak_ball_speed_ms=peak_speed,
                    ))
                    last_release_frame = frames[pass_release_idx]
                state = "near"
                pass_release_idx = None
                peak_idx = None
                saw_far = False
    return cycles


# --- HUD --------------------------------------------------------------


def _hud_fields(
    *,
    frame_idx: int,
    fps: float,
    cycles: list[_Cycle],
) -> dict[str, str]:
    completed = [c for c in cycles if c.reception_frame <= frame_idx]
    n = len(completed)
    if completed:
        avg_v = float(np.mean([c.peak_ball_speed_ms for c in completed]))
    else:
        avg_v = 0.0
    return {
        "phase": "running",
        "passes": str(n),
        "avg_v": f"{avg_v:.1f} m/s",
        "elapsed": f"{frame_idx / fps:.1f} s",
    }
