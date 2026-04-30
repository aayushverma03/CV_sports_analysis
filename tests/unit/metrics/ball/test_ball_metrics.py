"""Tests for the Ball metric batch (Phase 2.12 - 2.16)."""
from __future__ import annotations

import numpy as np
import pytest

from src.metrics.ball.average_pass_velocity_ms import average_pass_velocity_ms
from src.metrics.ball.ball_foot_distance_m import ball_foot_distance_m
from src.metrics.ball.max_consecutive_touches import max_consecutive_touches
from src.metrics.ball.max_pass_velocity_ms import max_pass_velocity_ms
from src.metrics.ball.passing_accuracy_percent import passing_accuracy_percent
from src.metrics.ball.touches_per_metre import touches_per_metre


# --- touches_per_metre --------------------------------------------------


def test_touches_per_metre_basic():
    assert touches_per_metre(total_ball_touches=12, total_distance_m=24.0) == 0.5


# --- ball_foot_distance_m -----------------------------------------------


def test_ball_foot_distance_picks_nearest_foot():
    ball = np.array([[0, 0], [1, 0], [2, 0]])
    left = np.array([[5, 0], [5, 0], [5, 0]])    # always far
    right = np.array([[0.3, 0], [1.2, 0], [1.8, 0]])  # always near
    out = ball_foot_distance_m(ball, left, right)
    # nearest = right side: 0.3, 0.2, 0.2 -> mean 0.2333, median 0.2
    assert out["mean_m"] == pytest.approx(0.2333, abs=1e-3)
    assert out["median_m"] == pytest.approx(0.2)
    assert len(out["series_m"]) == 3


def test_ball_foot_distance_handles_nan_frames():
    """NaN ball positions exclude their frame from mean / median."""
    ball = np.array([[0, 0], [np.nan, np.nan], [2, 0]])
    left = np.array([[5, 0], [5, 0], [5, 0]])
    right = np.array([[0.5, 0], [0.5, 0], [1.0, 0]])
    out = ball_foot_distance_m(ball, left, right)
    # frame 0 dist = 0.5, frame 1 = NaN, frame 2 = 1.0
    # mean over non-NaN = 0.75, median = 0.75
    assert out["mean_m"] == pytest.approx(0.75)
    assert out["median_m"] == pytest.approx(0.75)


# --- max_consecutive_touches --------------------------------------------


def test_max_consecutive_touches_basic():
    assert max_consecutive_touches([5, 12, 3, 8, 1]) == 12


def test_max_consecutive_touches_empty():
    assert max_consecutive_touches([]) == 0


def test_max_consecutive_touches_single_streak():
    assert max_consecutive_touches([42]) == 42


# --- pass_velocity ------------------------------------------------------


def test_average_pass_velocity_constant_motion():
    # Ball moves 0.5 m per frame at 30 fps -> 15 m/s
    pts = np.column_stack([np.arange(0, 5, 0.5), np.zeros(10)])
    assert average_pass_velocity_ms(pts, fps=30.0) == pytest.approx(15.0)


def test_max_pass_velocity_picks_peak():
    # 4 frames: distances 0.1, 0.5, 0.2 m at 30 fps -> 3, 15, 6 m/s
    pts = np.array([[0.0, 0], [0.1, 0], [0.6, 0], [0.8, 0]])
    assert max_pass_velocity_ms(pts, fps=30.0) == pytest.approx(15.0)


def test_average_pass_velocity_lower_than_max():
    pts = np.array([[0.0, 0], [0.1, 0], [0.6, 0], [0.8, 0]])
    assert average_pass_velocity_ms(pts, fps=30.0) < max_pass_velocity_ms(pts, fps=30.0)


# --- passing_accuracy_percent -------------------------------------------


def test_passing_accuracy_basic():
    assert passing_accuracy_percent(successful_passes=15, total_attempts=20) == 75.0


def test_passing_accuracy_zero_attempts():
    assert passing_accuracy_percent(0, 0) == 0.0


def test_passing_accuracy_perfect():
    assert passing_accuracy_percent(10, 10) == 100.0
