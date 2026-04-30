"""`rsi` — Reactive Strength Index from rebound height and ground contact time."""
from __future__ import annotations


def rsi(rebound_height_cm: float, ground_contact_time_s: float) -> float:
    """Reactive Strength Index = rebound_height (m) / ground contact time (s).

    Parameters
    ----------
    rebound_height_cm : float
        Drop-jump rebound height in centimetres (typically computed via
        `jump_height_cm` from rebound flight time).
    ground_contact_time_s : float
        Ground contact time in seconds. Must be > 0.

    Returns
    -------
    float
        RSI in m/s (units are conventional even though the ratio is
        dimensionally m/s). Higher = more explosive reactive ability.
    """
    return (rebound_height_cm / 100.0) / ground_contact_time_s
