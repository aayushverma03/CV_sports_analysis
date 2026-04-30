"""`peak_acceleration_ms2` — peak positive d(speed)/dt of a smoothed series."""
from __future__ import annotations

import numpy as np

from src.core.utils.smoothing import savgol_smooth


def peak_acceleration_ms2(
    speed_series_ms: np.ndarray,
    fps: float,
    window: int = 11,
    polyorder: int = 3,
) -> float:
    """Maximum acceleration (positive derivative of speed), in m/s².

    Computed as the SG first derivative of the speed series with the same
    smoothing as `max_speed_ms` so that the two metrics are consistent.

    Parameters
    ----------
    speed_series_ms : np.ndarray, shape (N,)
        Per-frame instantaneous speed in m/s.
    fps : float
        Frame rate; sets the SG `delta = 1/fps` so the derivative is in
        per-second units.
    window : int
        SG window. Default 11.
    polyorder : int
        SG polynomial order. Default 3.

    Returns
    -------
    float
        Peak positive acceleration (m/s²). Negative if the athlete is only
        decelerating across the whole window.
    """
    accel = savgol_smooth(
        speed_series_ms, window=window, polyorder=polyorder,
        deriv=1, delta=1.0 / fps,
    )
    return float(np.max(accel))
