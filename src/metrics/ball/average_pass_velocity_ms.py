"""`average_pass_velocity_ms` — mean ball speed across a pass-flight window."""
from __future__ import annotations

import numpy as np


def average_pass_velocity_ms(
    ball_positions_m: np.ndarray, fps: float
) -> float:
    """Mean ball speed over a sequence of in-flight ball positions.

    Caller passes only the post-release / pre-impact frames (the test
    pipeline is responsible for windowing).

    Parameters
    ----------
    ball_positions_m : np.ndarray, shape (N, 2)
        Ball centres in world coordinates (metres). N >= 2.
    fps : float
        Frame rate.

    Returns
    -------
    float
        Mean instantaneous speed in m/s.
    """
    pts = np.asarray(ball_positions_m, dtype=float)
    diffs = np.diff(pts, axis=0)
    speeds = np.linalg.norm(diffs, axis=1) * fps
    return float(np.mean(speeds))
