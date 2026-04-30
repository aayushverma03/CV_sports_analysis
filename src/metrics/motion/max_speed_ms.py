"""`max_speed_ms` — peak instantaneous speed of a smoothed series."""
from __future__ import annotations

import numpy as np

from src.core.utils.smoothing import savgol_smooth


def max_speed_ms(
    speed_series_ms: np.ndarray,
    window: int = 11,
    polyorder: int = 3,
) -> float:
    """Peak speed after Savitzky-Golay smoothing.

    Smoothing suppresses single-frame tracking spikes that would otherwise
    inflate the peak. Default window/polyorder match `core.utils.smoothing`
    defaults — short enough to preserve real acceleration peaks, long enough
    to absorb single-frame noise.

    Parameters
    ----------
    speed_series_ms : np.ndarray, shape (N,)
        Per-frame instantaneous speed in m/s.
    window : int
        Savitzky-Golay window length (odd). Default 11.
    polyorder : int
        Polynomial order. Default 3.

    Returns
    -------
    float
        Maximum smoothed speed (m/s).
    """
    smoothed = savgol_smooth(speed_series_ms, window=window, polyorder=polyorder)
    return float(np.max(smoothed))
