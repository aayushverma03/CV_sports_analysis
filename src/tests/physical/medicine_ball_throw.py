"""Medicine Ball Throw pipeline.

Side-on camera, athlete throws a medicine ball forward from the chest;
distance from the start line to the first landing point is the scored
metric.

Detection model
---------------
- Track athlete + ball with ByteTrack in a single pass.
- Per frame, compute ball position and athlete position.
- **Release frame**: the ball moves more than half a bbox-h away from
  the athlete and continues moving away — last "ball-near-chest"
  frame before that crossover.
- **Flight**: ball follows a parabolic trajectory; we track it until
  it falls back to (and stays at) the athlete's foot baseline y.
- **Landing frame**: ball y plateaus at the foot baseline (no further
  downward motion).
- Throw distance = horizontal pixel displacement from release point
  to landing point, scaled to metres via the body-height proxy
  (athlete bbox-h / assumed 1.70 m).

Single-pass design — athlete is largely stationary during the throw,
so no camera-motion compensation is needed; pose runs only at the
release frame for trunk-rotation read.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

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

# Release detection — ball is "released" when its distance from the
# athlete crosses this fraction of bbox-h going up.
_RELEASE_DISTANCE_FRAC = 0.5
# After release, ball must keep moving away for a few frames to confirm
# (rejects brief detection wobble).
_RELEASE_CONFIRM_FRAMES = 3

# Landing detection — ball y has stopped descending (within tolerance)
# AND has been below the athlete's hip y for at least this long.
_LANDING_TOLERANCE_PX = 5.0
_LANDING_HOLD_FRAMES = 5

_POSE_CONF_MIN = 0.30
_POSE_INTERVAL_FRAMES = 3
_ENDCARD_HOLD_S = 2.5
_DEFAULT_ATHLETE_HEIGHT_M = 1.70

_MIN_TRACK_HISTORY_FRAMES = 30


# --- State -------------------------------------------------------------


@dataclass(frozen=True)
class _Throw:
    release_frame: int
    landing_frame: int
    release_xy: tuple[float, float]            # ball position at release
    athlete_release_xy: tuple[float, float]    # athlete center at release
    landing_xy: tuple[float, float]
    max_height_xy: tuple[float, float]         # ball position at peak

    @property
    def flight_frames(self) -> int:
        return self.landing_frame - self.release_frame

    @property
    def horizontal_px(self) -> float:
        """Throw distance is measured from the athlete's start line
        (= athlete's center x at release) to the ball's first landing
        x, per the protocol's "from the start line to the first
        landing point" definition.
        """
        return abs(self.landing_xy[0] - self.athlete_release_xy[0])

    @property
    def peak_height_px(self) -> float:
        # Smaller pixel-y = higher in image. Height = release_y - peak_y.
        return max(0.0, self.release_xy[1] - self.max_height_xy[1])


# --- Pipeline ----------------------------------------------------------


class MedicineBallThrowTest(BaseTest):
    """Medicine Ball Throw: detect release + landing, measure distance."""

    test_id = "medicine-ball-throw"

    def __init__(
        self,
        *,
        assumed_athlete_height_m: float = _DEFAULT_ATHLETE_HEIGHT_M,
    ) -> None:
        if assumed_athlete_height_m <= 0:
            raise ValueError(
                f"assumed_athlete_height_m must be > 0, "
                f"got {assumed_athlete_height_m}"
            )
        self._assumed_height_m = float(assumed_athlete_height_m)
        self._tracker = ByteTrackTracker(
            classes=[PERSON_CLASS_ID, SPORTS_BALL_CLASS_ID],
            confidence=0.10,
        )
        self._pose = create_pose_estimator("pose_biomech")

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

        # === Pick athlete ===
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

        # === Build per-frame ball + athlete signals ===
        athlete_pos_by_frame: dict[int, tuple[float, float]] = {
            fi: (cx, cy)
            for (fi, cx, cy, _, _) in track_history[player_track_id]
        }
        bbox_h_by_frame: dict[int, float] = {
            fi: float(h)
            for (fi, _, _, h, _) in track_history[player_track_id]
        }
        # Ball nearest the athlete each frame.
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

        if len(ball_pos_by_frame) < _MIN_TRACK_HISTORY_FRAMES // 2:
            raise ProtocolError(
                f"only {len(ball_pos_by_frame)} ball-detected frames — "
                "ball tracking failed; check ball visibility"
            )

        # === Detect release + landing ===
        throw = _detect_throw(
            athlete_pos_by_frame=athlete_pos_by_frame,
            bbox_h_by_frame=bbox_h_by_frame,
            ball_pos_by_frame=ball_pos_by_frame,
        )
        if throw is None:
            raise ProtocolError(
                "could not locate a release-throw-landing sequence in the "
                "video; check that the ball is visible at the chest before "
                "release and through landing"
            )

        # === Calibration: body-height proxy ===
        baseline_h = float(np.median(list(bbox_h_by_frame.values())))
        if baseline_h <= 0:
            raise ProtocolError(
                "no usable bbox-h baseline for body-proxy calibration"
            )
        px_per_m = baseline_h / self._assumed_height_m

        # === Compute metrics ===
        throw_distance_m = throw.horizontal_px / px_per_m
        max_height_m = throw.peak_height_px / px_per_m
        flight_time_s = throw.flight_frames / fps
        release_angle_deg, release_velocity_ms = _release_kinematics(
            throw=throw, fps=fps, px_per_m=px_per_m,
            ball_pos_by_frame=ball_pos_by_frame,
        )

        metrics: dict[str, MetricValue] = {
            "throw_distance_m": MetricValue(raw=throw_distance_m, unit="m"),
            "release_velocity_ms": MetricValue(
                raw=release_velocity_ms, unit="m_per_s",
            ),
            "release_angle_deg": MetricValue(
                raw=release_angle_deg, unit="deg",
            ),
            "flight_time_s": MetricValue(raw=flight_time_s, unit="s"),
            "max_height_m": MetricValue(raw=max_height_m, unit="m"),
        }
        scores, test_score = score_test(metrics, self.test_id, athlete.gender)

        # === PASS 2: render ===
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
                ball = (
                    balls_per_frame.get(frame.idx, [None])[0]
                    if balls_per_frame.get(frame.idx) else None
                )
                if ball is not None:
                    draw_bbox(img, ball.bbox_xyxy, color=BALL)
                draw_hud(img, _hud_fields(
                    frame_idx=frame.idx, fps=fps, throw=throw,
                    throw_distance_m=throw_distance_m,
                    release_velocity_ms=release_velocity_ms,
                ))
                writer.write(img)

            endcard_rows = [
                (mid.replace("_", " "), f"{mv.raw:.3f} {mv.unit}",
                 int(round(scores[mid].score)) if mid in scores else 0)
                for mid, mv in metrics.items()
            ]
            endcard = render_endcard(
                title="Medicine Ball Throw",
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


# --- Throw detection --------------------------------------------------


def _detect_throw(
    *,
    athlete_pos_by_frame: dict[int, tuple[float, float]],
    bbox_h_by_frame: dict[int, float],
    ball_pos_by_frame: dict[int, tuple[float, float]],
) -> _Throw | None:
    """Walk through frames in order; identify the release frame as the
    last frame the ball was near the athlete before it moves away
    persistently. Track ball through flight, find peak height; landing
    is when ball y stops descending (plateau)."""
    common_frames = sorted(
        f for f in ball_pos_by_frame.keys()
        if f in athlete_pos_by_frame and f in bbox_h_by_frame
    )
    if len(common_frames) < 10:
        return None

    # Find release: the longest "ball moving away" run after a "near"
    # state. Concretely, scan for a frame where ball-athlete distance
    # exceeds 0.5x bbox-h AND the previous N frames show monotonically
    # increasing distance.
    release_frame: int | None = None
    last_near_frame: int | None = None
    confirm_count = 0
    prev_distance: float | None = None
    for f in common_frames:
        ax, ay = athlete_pos_by_frame[f]
        bx, by = ball_pos_by_frame[f]
        d = float(np.hypot(bx - ax, by - ay))
        threshold = _RELEASE_DISTANCE_FRAC * bbox_h_by_frame[f]
        if d <= threshold:
            last_near_frame = f
            confirm_count = 0
            prev_distance = d
            continue
        # d > threshold
        if prev_distance is not None and d > prev_distance:
            confirm_count += 1
            if (confirm_count >= _RELEASE_CONFIRM_FRAMES
                    and last_near_frame is not None):
                release_frame = last_near_frame
                break
        else:
            confirm_count = 0
        prev_distance = d
    if release_frame is None:
        return None

    release_xy = ball_pos_by_frame[release_frame]

    # Find landing: scan forward from release; track peak (min y) and
    # then find a frame where y stops descending (plateau).
    post_release = [f for f in common_frames if f > release_frame]
    if len(post_release) < 5:
        return None

    peak_y = release_xy[1]
    peak_xy = release_xy
    peak_frame = release_frame
    for f in post_release:
        bx, by = ball_pos_by_frame[f]
        if by < peak_y:    # higher in image = smaller y
            peak_y = by
            peak_xy = (bx, by)
            peak_frame = f

    # Landing: ball y plateaus at or near the release-y baseline,
    # scanned only AFTER the peak. Two extra constraints keep us from
    # mis-firing near the apex (where vertical motion is slow):
    #   1. only count plateau on frames where y has descended at
    #      least halfway from the peak back to the release baseline;
    #   2. only count plateau when |Δy| is small AND the previous
    #      frames showed real downward motion (so a slow apex pass
    #      doesn't trigger).
    after_peak = [f for f in post_release if f > peak_frame]
    midpoint_y = (peak_y + release_xy[1]) / 2.0
    landing_frame: int | None = None
    last_y: float | None = None
    plateau_count = 0
    for f in after_peak:
        bx, by = ball_pos_by_frame[f]
        if last_y is None:
            last_y = by
            continue
        descended_enough = by >= midpoint_y
        small_step = abs(by - last_y) <= _LANDING_TOLERANCE_PX
        if descended_enough and small_step:
            plateau_count += 1
            if plateau_count >= _LANDING_HOLD_FRAMES:
                landing_frame = f
                break
        else:
            plateau_count = 0
        last_y = by
    if landing_frame is None:
        # Fall back to the last frame the ball was tracked.
        landing_frame = post_release[-1]
    landing_xy = ball_pos_by_frame[landing_frame]

    athlete_release_xy = athlete_pos_by_frame[release_frame]
    return _Throw(
        release_frame=release_frame,
        landing_frame=landing_frame,
        release_xy=release_xy,
        athlete_release_xy=athlete_release_xy,
        landing_xy=landing_xy,
        max_height_xy=peak_xy,
    )


def _release_kinematics(
    *,
    throw: _Throw,
    fps: float,
    px_per_m: float,
    ball_pos_by_frame: dict[int, tuple[float, float]],
) -> tuple[float, float]:
    """Return (release_angle_deg, release_velocity_ms).

    Velocity uses ball positions at release_frame and 3 frames after to
    estimate initial velocity. Angle is measured above the horizontal
    using the same two points (positive = upward, negative = downward).
    """
    fr = throw.release_frame
    fnext = fr + 3
    if fnext not in ball_pos_by_frame:
        # Fall back to whatever's closest after release.
        candidates = sorted(
            f for f in ball_pos_by_frame if f > fr
        )
        if not candidates:
            return 0.0, 0.0
        fnext = candidates[min(2, len(candidates) - 1)]
    x0, y0 = throw.release_xy
    x1, y1 = ball_pos_by_frame[fnext]
    dt_frames = fnext - fr
    if dt_frames <= 0:
        return 0.0, 0.0
    dx_px = x1 - x0
    dy_px = y0 - y1     # image y inverted: positive = upward
    velocity_px_per_frame = float(np.hypot(dx_px, dy_px) / dt_frames)
    velocity_ms = velocity_px_per_frame * fps / px_per_m
    angle_deg = float(np.degrees(np.arctan2(dy_px, dx_px)))
    return angle_deg, velocity_ms


# --- HUD --------------------------------------------------------------


def _hud_fields(
    *,
    frame_idx: int,
    fps: float,
    throw: _Throw,
    throw_distance_m: float,
    release_velocity_ms: float,
) -> dict[str, str]:
    if frame_idx < throw.release_frame:
        return {"phase": "ready", "distance": "-", "velocity": "-"}
    if frame_idx <= throw.landing_frame:
        return {
            "phase": "in_flight",
            "distance": "-",
            "velocity": f"{release_velocity_ms:.1f} m/s",
        }
    return {
        "phase": "landed",
        "distance": f"{throw_distance_m:.2f} m",
        "velocity": f"{release_velocity_ms:.1f} m/s",
    }
