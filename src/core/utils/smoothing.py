"""Signal smoothing — Savitzky-Golay and Kalman wrappers.

Used by metrics to suppress single-frame noise in speed, acceleration, and
trajectory series before peak detection (per METRICS_CATALOG.md).
"""
from __future__ import annotations

import numpy as np
from scipy.signal import savgol_filter


def savgol_smooth(
    y: np.ndarray,
    window: int = 11,
    polyorder: int = 3,
    deriv: int = 0,
    delta: float = 1.0,
) -> np.ndarray:
    """Savitzky-Golay smoothing or numerical derivative of a 1D series.

    Parameters
    ----------
    y : np.ndarray, shape (N,)
        Input series.
    window : int
        Filter window length. If even, decremented to odd. If larger than N,
        clamped to the largest valid odd window.
    polyorder : int
        Polynomial order. Reduced if it would exceed the effective window.
    deriv : int
        0 = smoothed series, 1 = first derivative, 2 = second derivative.
    delta : float
        Sample spacing. Set to `1 / fps` so derivatives come out in
        per-second units (e.g. m/s when y is metres).

    Returns
    -------
    np.ndarray, shape (N,)
        Smoothed series or its derivative.
    """
    y = np.asarray(y, dtype=float)
    n = len(y)
    w = window if window % 2 == 1 else window - 1
    w = min(w, n if n % 2 == 1 else n - 1)
    p = min(polyorder, max(0, w - 1))
    return savgol_filter(y, w, p, deriv=deriv, delta=delta)


def kalman_smooth_2d(
    positions: np.ndarray,
    fps: float,
    process_noise: float = 1.0,
    measurement_noise: float = 1.0,
) -> np.ndarray:
    """Smooth a 2D position series with a constant-velocity Kalman + RTS smoother.

    Parameters
    ----------
    positions : np.ndarray, shape (N, 2)
        Noisy (x, y) measurements, e.g. athlete COM in pixels or metres.
    fps : float
        Frame rate.
    process_noise : float
        Process noise magnitude (Q diagonal). Larger = more responsive.
    measurement_noise : float
        Measurement noise magnitude (R diagonal). Larger = more smoothing.

    Returns
    -------
    np.ndarray, shape (N, 2)
        Smoothed position estimates.
    """
    from filterpy.kalman import KalmanFilter

    pos = np.asarray(positions, dtype=float)
    dt = 1.0 / fps

    kf = KalmanFilter(dim_x=4, dim_z=2)
    kf.F = np.array(
        [
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ],
        dtype=float,
    )
    kf.H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=float)
    kf.R = np.eye(2) * measurement_noise
    kf.Q = np.eye(4) * process_noise
    kf.x = np.array([pos[0, 0], pos[0, 1], 0.0, 0.0])
    kf.P = np.eye(4) * 1000.0

    means, covs, _, _ = kf.batch_filter(pos)
    smoothed, _, _, _ = kf.rts_smoother(means, covs)
    return smoothed[:, :2]
