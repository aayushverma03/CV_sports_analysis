"""`max_consecutive_touches` — longest streak of touches before a drop."""
from __future__ import annotations

from collections.abc import Sequence


def max_consecutive_touches(streaks: Sequence[int]) -> int:
    """Return the longest streak from a list of streak lengths.

    The metric is intentionally streak-aware, not event-aware: the caller
    (test pipeline) is responsible for partitioning a touch event sequence
    into streaks separated by drops, then passing the lengths here. This
    keeps the metric pure (no test-specific drop-detection logic).

    Parameters
    ----------
    streaks : Sequence[int]
        Per-streak touch counts. Empty sequence -> 0.

    Returns
    -------
    int
        Maximum streak length.
    """
    return max(streaks) if streaks else 0
