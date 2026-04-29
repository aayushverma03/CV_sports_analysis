"""Tests for smoothing helpers."""
from __future__ import annotations

import numpy as np

from src.core.utils.smoothing import kalman_smooth_2d, savgol_smooth


def test_savgol_preserves_polynomial():
    # SG with polyorder p exactly preserves polynomials of order <= p.
    x = np.arange(50)
    y = 2.0 + 3.0 * x - 0.1 * x**2  # quadratic
    smoothed = savgol_smooth(y, window=11, polyorder=3)
    assert np.allclose(smoothed, y, atol=1e-9)


def test_savgol_suppresses_single_frame_spike():
    # SG cubic ringing is expected; we just want the spike attenuated.
    y = np.zeros(50)
    y[25] = 100.0
    smoothed = savgol_smooth(y, window=11, polyorder=3)
    assert smoothed[25] < 30.0  # spike peak attenuated to <30% of original


def test_savgol_first_derivative_constant_velocity():
    # y = 2*t with delta=1/fps gives derivative = 2*fps converted properly
    fps = 30.0
    t = np.arange(60)
    y = 2.0 * t  # in pixels per frame, slope 2
    deriv = savgol_smooth(y, window=11, polyorder=2, deriv=1, delta=1.0 / fps)
    # dy/dt = 2 px/frame * fps frames/s = 60 px/s
    # check interior points (edges of SG can deviate)
    assert np.allclose(deriv[20:40], 60.0, atol=1e-6)


def test_savgol_clamps_oversized_window():
    y = np.arange(7, dtype=float)  # short series
    out = savgol_smooth(y, window=51, polyorder=3)
    assert out.shape == y.shape
    assert np.all(np.isfinite(out))


def test_savgol_handles_even_window():
    y = np.linspace(0, 10, 30)
    out = savgol_smooth(y, window=10, polyorder=3)  # even -> 9
    assert out.shape == y.shape


def test_kalman_smooth_2d_recovers_linear_motion():
    rng = np.random.default_rng(0)
    fps = 30.0
    n = 100
    t = np.arange(n) / fps
    truth = np.column_stack([2.0 * t, 1.0 * t])  # 2 m/s in x, 1 m/s in y
    noise = rng.normal(0.0, 0.1, (n, 2))
    measured = truth + noise
    smoothed = kalman_smooth_2d(measured, fps=fps,
                                 process_noise=0.01, measurement_noise=0.1)
    # Smoothed result should be substantially closer to truth than raw
    raw_err = np.linalg.norm(measured - truth, axis=1).mean()
    smooth_err = np.linalg.norm(smoothed - truth, axis=1).mean()
    assert smooth_err < raw_err * 0.6


def test_kalman_shape_preserved():
    pos = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
    out = kalman_smooth_2d(pos, fps=30.0)
    assert out.shape == pos.shape
