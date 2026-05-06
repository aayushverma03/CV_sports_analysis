"""Unit tests for Multistage Fitness pipeline-specific helpers."""
from __future__ import annotations

from src.tests.physical.multistage_fitness import (
    _count_shuttles,
    _level_and_max_speed_for_shuttle_count,
    _vo2max_leger,
    _RunWindow,
)


# --- _level_and_max_speed_for_shuttle_count ---------------------------


def test_level_zero_at_zero_shuttles():
    level, speed = _level_and_max_speed_for_shuttle_count(0)
    assert level == 0.0
    assert speed == 0.0


def test_level_1_after_seven_shuttles():
    """Level 1 = 7 shuttles cumulative."""
    level, speed = _level_and_max_speed_for_shuttle_count(7)
    assert level == 1.0
    assert speed == 8.5


def test_level_progresses_through_ladder():
    """Cumulative landmarks: 7=L1, 15=L2, 23=L3, 32=L4, 41=L5."""
    assert _level_and_max_speed_for_shuttle_count(15)[0] == 2.0
    assert _level_and_max_speed_for_shuttle_count(23)[0] == 3.0
    assert _level_and_max_speed_for_shuttle_count(32)[0] == 4.0
    assert _level_and_max_speed_for_shuttle_count(41)[0] == 5.0


def test_level_caps_at_top_of_ladder():
    level, speed = _level_and_max_speed_for_shuttle_count(10000)
    assert level == 21.0
    assert speed == 18.5


# --- _vo2max_leger ----------------------------------------------------


def test_vo2max_zero_below_msft_floor():
    assert _vo2max_leger(0.0) == 0.0


def test_vo2max_at_level_1_start():
    """Level 1 starts at 8.5 km/h: 6 * 8.5 - 27.4 = 23.6."""
    assert abs(_vo2max_leger(8.5) - 23.6) < 1e-6


def test_vo2max_grows_linearly_with_speed():
    a = _vo2max_leger(10.0)
    b = _vo2max_leger(11.0)
    assert abs((b - a) - 6.0) < 1e-6


# --- _count_shuttles --------------------------------------------------


def _shuttle_traj(
    n_shuttles: int,
    x_left: float = 0.0,
    x_right: float = 1000.0,
    samples_per_shuttle: int = 90,
) -> list[tuple[int, float, float, float, float]]:
    """Synthetic stabilized trajectory: athlete alternates one-way
    20 m trips, `n_shuttles` of them. Each shuttle goes from one end
    to the other (alternating direction)."""
    pts: list[tuple[int, float, float, float, float]] = []
    fi = 0
    cur, target = x_left, x_right
    for _ in range(n_shuttles):
        for k in range(samples_per_shuttle):
            t = (k + 1) / samples_per_shuttle
            x = cur + (target - cur) * t
            pts.append((fi, x, 400.0, 200.0, 100.0))
            fi += 1
        cur, target = target, cur
    return pts


def test_count_shuttles_three_one_way_trips():
    history = _shuttle_traj(n_shuttles=3, samples_per_shuttle=90)
    run = _RunWindow(track_id=1, start_frame=0, stop_frame=history[-1][0])
    assert _count_shuttles(history=history, run_window=run, fps=30.0) == 3


def test_count_shuttles_zero_when_stationary():
    history = [(i, 500.0, 400.0, 200.0, 100.0) for i in range(120)]
    run = _RunWindow(track_id=1, start_frame=0, stop_frame=119)
    assert _count_shuttles(history=history, run_window=run, fps=30.0) == 0


def test_count_shuttles_window_filter():
    """Shuttles before / after the run window should not count."""
    pre = [(i, 0.0, 400.0, 200.0, 100.0) for i in range(50)]
    main = _shuttle_traj(n_shuttles=4, samples_per_shuttle=90)
    main_shifted = [
        (50 + fi, x, y, h, w) for (fi, x, y, h, w) in main
    ]
    post_offset = main_shifted[-1][0] + 1
    post = [(post_offset + i, 0.0, 400.0, 200.0, 100.0) for i in range(50)]
    history = pre + main_shifted + post
    run = _RunWindow(
        track_id=1, start_frame=50, stop_frame=main_shifted[-1][0],
    )
    assert _count_shuttles(history=history, run_window=run, fps=30.0) == 4
