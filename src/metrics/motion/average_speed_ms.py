"""`average_speed_ms` — mean speed over a test window."""
from __future__ import annotations


def average_speed_ms(total_distance_m: float, total_completion_time_s: float) -> float:
    """Mean speed = distance / time.

    Parameters
    ----------
    total_distance_m : float
        Cumulative path length in metres.
    total_completion_time_s : float
        Test elapsed time in seconds. Must be > 0.

    Returns
    -------
    float
        Average speed in m/s.
    """
    return total_distance_m / total_completion_time_s
