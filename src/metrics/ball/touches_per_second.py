"""`touches_per_second` — sustained juggling cadence."""
from __future__ import annotations


def touches_per_second(total_touches: int, active_duration_s: float) -> float:
    """Touches per second over the active juggling window.

    Parameters
    ----------
    total_touches : int
        Sum of touches across all streaks. >= 0.
    active_duration_s : float
        Time from first touch to last touch in the run, in seconds.
        Must be > 0; if 0 (single touch or no touches), returns 0.

    Returns
    -------
    float
        Touches per second (Hz). Higher = faster sustained cadence.
    """
    if active_duration_s <= 0:
        return 0.0
    return total_touches / active_duration_s
