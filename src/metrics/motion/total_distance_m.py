"""`total_distance_m` — cumulative path length of an N x 2 trajectory."""
from __future__ import annotations

import numpy as np


def total_distance_m(positions_world_m: np.ndarray) -> float:
    """Sum of step-to-step Euclidean distances along a trajectory.

    Parameters
    ----------
    positions_world_m : np.ndarray, shape (N, 2)
        Athlete COM positions in world coordinates (metres). Caller is
        responsible for converting from pixels via the calibration layer.

    Returns
    -------
    float
        Total path length in metres. Returns 0.0 for N < 2.
    """
    pts = np.asarray(positions_world_m, dtype=float)
    if pts.shape[0] < 2:
        return 0.0
    diffs = np.diff(pts, axis=0)
    return float(np.linalg.norm(diffs, axis=1).sum())
