"""`passing_accuracy_percent` — percent of passes that ended in successful recovery."""
from __future__ import annotations


def passing_accuracy_percent(successful_passes: int, total_attempts: int) -> float:
    """Successful-pass rate as a percentage.

    For Wall Pass: a "successful pass" = athlete struck ball toward wall +
    ball returned + athlete recontrolled the rebound. `total_attempts` is
    every pass release.

    Parameters
    ----------
    successful_passes : int
    total_attempts : int

    Returns
    -------
    float
        Percentage in [0, 100]. Returns 0.0 when no attempts were made.
    """
    if total_attempts == 0:
        return 0.0
    return successful_passes / total_attempts * 100.0
