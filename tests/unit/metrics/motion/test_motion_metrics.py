"""Tests for the Motion metric batch (Phase 2.1 - 2.7)."""
from __future__ import annotations

import numpy as np
import pytest

from src.metrics.motion.average_speed_ms import average_speed_ms
from src.metrics.motion.max_speed_ms import max_speed_ms
from src.metrics.motion.peak_acceleration_ms2 import peak_acceleration_ms2
from src.metrics.motion.peak_deceleration_ms2 import peak_deceleration_ms2
from src.metrics.motion.split_times_s import split_times_s
from src.metrics.motion.total_completion_time_s import total_completion_time_s
from src.metrics.motion.total_distance_m import total_distance_m


# --- total_completion_time_s --------------------------------------------


def test_total_completion_time_s_basic():
    assert total_completion_time_s(start_frame=10, end_frame=130, fps=30.0) == 4.0


def test_total_completion_time_s_zero_window():
    assert total_completion_time_s(start_frame=42, end_frame=42, fps=30.0) == 0.0


# --- split_times_s ------------------------------------------------------


def test_split_times_s_first_is_zero():
    out = split_times_s(crossing_frames=[100, 130, 160, 190], fps=30.0)
    assert out[0] == 0.0
    assert out == pytest.approx([0.0, 1.0, 2.0, 3.0])


def test_split_times_s_single_crossing():
    assert split_times_s([100], fps=30.0) == [0.0]


# --- total_distance_m ---------------------------------------------------


def test_total_distance_m_straight_line():
    pts = np.array([[0, 0], [3, 0], [3, 4]])  # 3 m + 4 m = 7 m
    assert total_distance_m(pts) == pytest.approx(7.0)


def test_total_distance_m_single_point():
    assert total_distance_m(np.array([[1.0, 2.0]])) == 0.0


def test_total_distance_m_empty():
    assert total_distance_m(np.zeros((0, 2))) == 0.0


# --- average_speed_ms ---------------------------------------------------


def test_average_speed_ms_basic():
    assert average_speed_ms(total_distance_m=100.0, total_completion_time_s=20.0) == 5.0


# --- max_speed_ms -------------------------------------------------------


def test_max_speed_ms_constant_series():
    s = np.full(50, 7.5)
    assert max_speed_ms(s) == pytest.approx(7.5, abs=1e-6)


def test_max_speed_ms_linear_ramp():
    s = np.linspace(0, 10, 50)
    # smoothed peak should be near 10
    assert max_speed_ms(s) == pytest.approx(10.0, abs=0.1)


def test_max_speed_ms_suppresses_single_frame_spike():
    s = np.full(50, 5.0)
    s[25] = 100.0
    assert max_speed_ms(s) < 50.0  # spike attenuated, not 100


# --- peak_acceleration_ms2 ----------------------------------------------


def test_peak_acceleration_ms2_constant_velocity_is_zero():
    s = np.full(50, 7.0)
    accel = peak_acceleration_ms2(s, fps=30.0)
    assert accel == pytest.approx(0.0, abs=1e-6)


def test_peak_acceleration_ms2_constant_acceleration():
    # speed ramps from 0 to 10 m/s linearly over 50 frames @ 30 fps -> dt=49/30 s
    # accel = 10 / (49/30) = 6.122 m/s²
    fps = 30.0
    s = np.linspace(0, 10, 50)
    expected = 10.0 / (49 / fps)
    out = peak_acceleration_ms2(s, fps=fps)
    assert out == pytest.approx(expected, rel=0.01)  # within 1% interior


# --- peak_deceleration_ms2 ----------------------------------------------


def test_peak_deceleration_ms2_returned_positive():
    fps = 30.0
    s = np.linspace(10, 0, 50)  # decelerating
    out = peak_deceleration_ms2(s, fps=fps)
    expected = 10.0 / (49 / fps)
    assert out > 0
    assert out == pytest.approx(expected, rel=0.01)


def test_peak_deceleration_ms2_zero_on_only_accelerating():
    s = np.linspace(0, 10, 50)
    assert peak_deceleration_ms2(s, fps=30.0) == 0.0
