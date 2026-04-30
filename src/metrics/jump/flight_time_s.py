"""`flight_time_s` — duration the athlete is airborne during a jump."""
from __future__ import annotations


def flight_time_s(takeoff_frame: int, landing_frame: int, fps: float) -> float:
    """Time between toe-off and touch-down.

    Parameters
    ----------
    takeoff_frame : int
        Frame index where the athlete's feet leave the ground.
    landing_frame : int
        Frame index of first foot contact after the jump.
    fps : float
        Frame rate.

    Returns
    -------
    float
        Flight time in seconds. Always >= 0.
    """
    return max(0.0, (landing_frame - takeoff_frame) / fps)
