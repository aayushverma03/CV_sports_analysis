"""`ground_contact_time_s` — time on ground between landing and rebound takeoff."""
from __future__ import annotations


def ground_contact_time_s(
    landing_frame: int, rebound_takeoff_frame: int, fps: float
) -> float:
    """Time the foot is in contact with the ground during a drop-jump rebound.

    Parameters
    ----------
    landing_frame : int
        Frame of first foot contact after the drop.
    rebound_takeoff_frame : int
        Frame at which the athlete leaves the ground for the rebound jump.
    fps : float
        Frame rate.

    Returns
    -------
    float
        Ground contact time in seconds. Always >= 0.
    """
    return max(0.0, (rebound_takeoff_frame - landing_frame) / fps)
