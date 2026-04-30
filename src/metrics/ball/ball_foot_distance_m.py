"""`ball_foot_distance_m` — distance from ball centre to nearest foot keypoint per frame."""
from __future__ import annotations

from typing import TypedDict

import numpy as np


class BallFootDistance(TypedDict):
    mean_m: float
    median_m: float
    series_m: list[float]


def ball_foot_distance_m(
    ball_positions_m: np.ndarray,
    left_ankle_m: np.ndarray,
    right_ankle_m: np.ndarray,
) -> BallFootDistance:
    """Per-frame distance from the ball to the nearer of the two ankle keypoints.

    NaN entries (frames where the ball or either ankle was not localized)
    contribute NaN to `series_m` but are excluded from `mean_m` and
    `median_m` via `np.nanmean` / `np.nanmedian`.

    Parameters
    ----------
    ball_positions_m : np.ndarray, shape (N, 2)
        Ball centre per frame, world coordinates (metres).
    left_ankle_m, right_ankle_m : np.ndarray, shape (N, 2)
        Ankle keypoints per frame, world coordinates (metres).

    Returns
    -------
    BallFootDistance
        Dict with `mean_m`, `median_m`, and `series_m` (list).
    """
    ball = np.asarray(ball_positions_m, dtype=float)
    left = np.asarray(left_ankle_m, dtype=float)
    right = np.asarray(right_ankle_m, dtype=float)

    d_left = np.linalg.norm(ball - left, axis=1)
    d_right = np.linalg.norm(ball - right, axis=1)
    series = np.minimum(d_left, d_right)
    return {
        "mean_m": float(np.nanmean(series)),
        "median_m": float(np.nanmedian(series)),
        "series_m": series.tolist(),
    }
