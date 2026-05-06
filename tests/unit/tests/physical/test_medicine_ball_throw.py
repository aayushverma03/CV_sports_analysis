"""Unit tests for Medicine Ball Throw pipeline-specific helpers."""
from __future__ import annotations

import numpy as np

from src.tests.physical.medicine_ball_throw import (
    _Throw,
    _detect_throw,
    _release_kinematics,
)


def _build_synthetic_throw(
    *,
    release_frame: int = 30,
    landing_frame: int = 90,
    n_frames_total: int = 120,
    bbox_h: float = 200.0,
    chest_x: float = 200.0,
    chest_y: float = 400.0,
    landing_x: float = 1000.0,
    peak_y_above_chest: float = 200.0,
):
    """Build athlete + ball + bbox-h dicts modelling a single throw."""
    athlete = {fi: (chest_x, chest_y) for fi in range(n_frames_total)}
    bbox_h_d = {fi: bbox_h for fi in range(n_frames_total)}
    ball: dict[int, tuple[float, float]] = {}
    # Pre-release: ball at chest.
    for fi in range(release_frame):
        ball[fi] = (chest_x, chest_y)
    # Flight: simple parabola from (chest_x, chest_y) to (landing_x,
    # chest_y), peaking at the midpoint with peak_y_above_chest below
    # chest_y in image space (smaller y = higher).
    flight_n = landing_frame - release_frame
    for k in range(flight_n + 1):
        t = k / flight_n
        x = chest_x + (landing_x - chest_x) * t
        # Parabola: peak at t=0.5
        y_offset = -peak_y_above_chest * (4 * t * (1 - t))
        ball[release_frame + k] = (x, chest_y + y_offset)
    # Post-landing: ball stays at landing position.
    for fi in range(landing_frame + 1, n_frames_total):
        ball[fi] = (landing_x, chest_y)
    return athlete, bbox_h_d, ball


# --- _detect_throw ----------------------------------------------------


def test_detect_throw_finds_release_and_landing():
    athlete, bbox_h, ball = _build_synthetic_throw(
        release_frame=30, landing_frame=90,
    )
    throw = _detect_throw(
        athlete_pos_by_frame=athlete,
        bbox_h_by_frame=bbox_h,
        ball_pos_by_frame=ball,
    )
    assert throw is not None
    # Release lands somewhere between the synthetic release frame
    # (30) and a few frames into flight, depending on how quickly the
    # ball clears the 0.5x bbox-h "near" zone. The synthetic uses a
    # smooth parabola; real throws accelerate sharply at release.
    assert 28 <= throw.release_frame <= 40
    # Landing detection plateaus a few frames after the actual landing.
    assert throw.landing_frame >= 90
    # Horizontal distance close to (1000 - 200) = 800 px (measured from
    # athlete's center x at release, not the ball's release x).
    assert abs(throw.horizontal_px - 800.0) < 5.0
    # Peak height ~ 100-200 px above release_xy.y. The release frame is
    # detected mid-flight (after the ball clears 0.5x bbox-h), so the
    # release-to-peak rise is less than the full peak_y_above_chest.
    assert 100.0 < throw.peak_height_px < 210.0


def test_detect_throw_none_when_ball_never_leaves_chest():
    """Ball stays at chest the whole video — no throw."""
    n = 120
    athlete = {fi: (200.0, 400.0) for fi in range(n)}
    bbox_h = {fi: 200.0 for fi in range(n)}
    ball = {fi: (200.0, 400.0) for fi in range(n)}
    throw = _detect_throw(
        athlete_pos_by_frame=athlete,
        bbox_h_by_frame=bbox_h,
        ball_pos_by_frame=ball,
    )
    assert throw is None


def test_detect_throw_none_when_too_few_frames():
    athlete = {fi: (200.0, 400.0) for fi in range(5)}
    bbox_h = {fi: 200.0 for fi in range(5)}
    ball = {fi: (200.0, 400.0) for fi in range(5)}
    throw = _detect_throw(
        athlete_pos_by_frame=athlete,
        bbox_h_by_frame=bbox_h,
        ball_pos_by_frame=ball,
    )
    assert throw is None


# --- _release_kinematics ----------------------------------------------


def test_release_kinematics_horizontal_throw():
    """Ball moves 30 px right + 0 px up over 3 frames at 30 fps,
    px_per_m = 100. Velocity = 30 px / 3 frames = 10 px/frame ->
    10 * 30 / 100 = 3 m/s; angle = 0 deg."""
    throw = _Throw(
        release_frame=10, landing_frame=20,
        release_xy=(200.0, 400.0),
        athlete_release_xy=(200.0, 400.0),
        landing_xy=(500.0, 400.0),
        max_height_xy=(350.0, 380.0),
    )
    ball_pos = {
        10: (200.0, 400.0),
        13: (230.0, 400.0),
    }
    angle, vel = _release_kinematics(
        throw=throw, fps=30.0, px_per_m=100.0,
        ball_pos_by_frame=ball_pos,
    )
    assert abs(angle - 0.0) < 1e-6
    assert abs(vel - 3.0) < 1e-6


def test_release_kinematics_45_degree_upward():
    """30 px right + 30 px up (lower image-y) -> 45 deg upward."""
    throw = _Throw(
        release_frame=10, landing_frame=20,
        release_xy=(200.0, 400.0),
        athlete_release_xy=(200.0, 400.0),
        landing_xy=(500.0, 400.0),
        max_height_xy=(350.0, 350.0),
    )
    ball_pos = {
        10: (200.0, 400.0),
        13: (230.0, 370.0),
    }
    angle, _ = _release_kinematics(
        throw=throw, fps=30.0, px_per_m=100.0,
        ball_pos_by_frame=ball_pos,
    )
    assert abs(angle - 45.0) < 0.5
