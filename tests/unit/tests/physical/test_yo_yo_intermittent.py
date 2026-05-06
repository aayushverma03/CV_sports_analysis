"""Unit tests for Yo-Yo IR2 pipeline-specific helpers."""
from __future__ import annotations

from src.tests.physical.yo_yo_intermittent import (
    _count_shuttles,
    _level_for_shuttle_count,
    _vo2max_from_distance_ir2,
    _RunWindow,
)


# --- _level_for_shuttle_count -----------------------------------------


def test_level_zero_when_no_shuttles():
    assert _level_for_shuttle_count(0) == 0.0


def test_level_19_at_first_shuttle():
    """1 shuttle = level 19 just completed."""
    assert _level_for_shuttle_count(1) == 19.0


def test_level_progresses_through_ladder():
    # cumulative shuttles: 1, 2, 4, 6, 9, 13, 21, 29
    # corresponding levels: 19, 20, 21, 22, 23, 24, 25, 26
    assert _level_for_shuttle_count(2) == 20.0
    assert _level_for_shuttle_count(3) == 20.0   # mid-21, not yet completed
    assert _level_for_shuttle_count(4) == 21.0
    assert _level_for_shuttle_count(9) == 23.0
    assert _level_for_shuttle_count(13) == 24.0


def test_level_caps_at_top_of_ladder():
    """100 shuttles is way past level 26; cap there."""
    assert _level_for_shuttle_count(100) == 26.0


# --- _vo2max_from_distance_ir2 ----------------------------------------


def test_vo2max_at_zero_distance_is_baseline():
    """Bangsbo's intercept: 45.3 at 0 m."""
    assert abs(_vo2max_from_distance_ir2(0.0) - 45.3) < 1e-6


def test_vo2max_linear_with_distance():
    """Slope 0.0136 * distance, in ml/kg/min."""
    assert abs(_vo2max_from_distance_ir2(1000.0) - (45.3 + 13.6)) < 1e-6


# --- _count_shuttles --------------------------------------------------


def _shuttle_traj(
    n_shuttles: int,
    x_left: float = 0.0,
    x_right: float = 1000.0,
    samples_per_half: int = 30,
) -> list[tuple[int, float, float, float, float]]:
    """Synthetic stabilized trajectory: athlete alternates from x_left
    -> x_right -> x_left -> ... for `n_shuttles` complete out-and-back
    cycles."""
    pts: list[tuple[int, float, float, float, float]] = []
    fi = 0
    for s in range(n_shuttles):
        # Out leg: x_left -> x_right
        for k in range(samples_per_half):
            t = (k + 1) / samples_per_half
            pts.append((fi, x_left + (x_right - x_left) * t, 400.0, 200.0, 100.0))
            fi += 1
        # Back leg: x_right -> x_left
        for k in range(samples_per_half):
            t = (k + 1) / samples_per_half
            pts.append((fi, x_right + (x_left - x_right) * t, 400.0, 200.0, 100.0))
            fi += 1
    return pts


def test_count_shuttles_three_clean_cycles():
    history = _shuttle_traj(n_shuttles=3, samples_per_half=30)
    run = _RunWindow(track_id=1, start_frame=0, stop_frame=history[-1][0])
    assert _count_shuttles(history=history, run_window=run, fps=30.0) == 3


def test_count_shuttles_zero_when_stationary():
    history = [(i, 500.0, 400.0, 200.0, 100.0) for i in range(120)]
    run = _RunWindow(track_id=1, start_frame=0, stop_frame=119)
    assert _count_shuttles(history=history, run_window=run, fps=30.0) == 0


def test_count_shuttles_ignores_frames_outside_window():
    """Athlete shuttles before AND after the run window — only in-window
    cycles count."""
    pre = [(i, 0.0, 400.0, 200.0, 100.0) for i in range(50)]
    main = _shuttle_traj(n_shuttles=2, samples_per_half=30)
    main_shifted = [
        (50 + fi, x, y, h, w) for (fi, x, y, h, w) in main
    ]
    post_offset = main_shifted[-1][0] + 1
    post = [(post_offset + i, 0.0, 400.0, 200.0, 100.0) for i in range(50)]
    history = pre + main_shifted + post
    run = _RunWindow(
        track_id=1,
        start_frame=50,
        stop_frame=main_shifted[-1][0],
    )
    assert _count_shuttles(history=history, run_window=run, fps=30.0) == 2
