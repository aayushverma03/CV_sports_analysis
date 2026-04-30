"""`split_times_s` — cumulative times at gate crossings."""
from __future__ import annotations

from collections.abc import Sequence


def split_times_s(crossing_frames: Sequence[int], fps: float) -> list[float]:
    """Convert a list of gate-crossing frame indices to cumulative seconds.

    Times are measured from the FIRST crossing (treated as t=0, the start
    gate). Subsequent splits are absolute times relative to that origin.

    Parameters
    ----------
    crossing_frames : Sequence[int]
        Frame indices at which gates were crossed, in order. Must contain
        at least one entry (the start). Caller is responsible for matching
        gate ordering.
    fps : float
        Frame rate.

    Returns
    -------
    list[float]
        Cumulative split times, same length as `crossing_frames`. First
        entry is always 0.0.
    """
    start = crossing_frames[0]
    return [(f - start) / fps for f in crossing_frames]
