"""`peak_deceleration_ms2` — most-negative d(speed)/dt of a smoothed series, returned positive."""
from __future__ import annotations

import numpy as np

from src.core.utils.smoothing import savgol_smooth


def peak_deceleration_ms2(
    speed_series_ms: np.ndarray,
    fps: float,
    window: int = 11,
    polyorder: int = 3,
) -> float:
    """Maximum magnitude of negative acceleration, returned as a positive m/s².

    Mirror of `peak_acceleration_ms2`. Convention: deceleration is reported
    as a positive number (e.g. -8 m/s² → 8.0). Returns 0.0 if the athlete
    only accelerated across the window.

    Parameters
    ----------
    speed_series_ms : np.ndarray, shape (N,)
        Per-frame instantaneous speed in m/s.
    fps : float
        Frame rate.
    window : int
        SG window. Default 11.
    polyorder : int
        SG polynomial order. Default 3.

    Returns
    -------
    float
        Peak deceleration magnitude (m/s²), >= 0.
    """
    accel = savgol_smooth(
        speed_series_ms, window=window, polyorder=polyorder,
        deriv=1, delta=1.0 / fps,
    )
    return float(max(0.0, -np.min(accel)))
