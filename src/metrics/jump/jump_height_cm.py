"""`jump_height_cm` — vertical jump height from flight time."""
from __future__ import annotations

GRAVITY_MS2 = 9.81


def jump_height_cm(flight_time_s: float) -> float:
    """Vertical jump height via the projectile / flight-time method.

    Formula: ``h = g * t² / 8``, where t is total flight time and the
    athlete lands at the same height they took off from. Output converted
    to centimetres.

    Parameters
    ----------
    flight_time_s : float
        Time between takeoff and landing, in seconds. Must be >= 0.

    Returns
    -------
    float
        Jump height in centimetres.
    """
    return GRAVITY_MS2 * flight_time_s ** 2 / 8.0 * 100.0
