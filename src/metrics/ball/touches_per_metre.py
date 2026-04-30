"""`touches_per_metre` — ball touches normalized by distance covered."""
from __future__ import annotations


def touches_per_metre(total_ball_touches: int, total_distance_m: float) -> float:
    """Touches divided by distance.

    Lower values typically indicate better ball control at speed (fewer
    steering nudges per metre travelled).

    Parameters
    ----------
    total_ball_touches : int
        Count of ball–foot contact events through the test.
    total_distance_m : float
        Athlete path length in metres. Must be > 0.

    Returns
    -------
    float
        Touches per metre.
    """
    return total_ball_touches / total_distance_m
