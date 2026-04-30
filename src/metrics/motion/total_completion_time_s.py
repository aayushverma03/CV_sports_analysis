"""`total_completion_time_s` — time from test-start event to test-end event."""
from __future__ import annotations


def total_completion_time_s(start_frame: int, end_frame: int, fps: float) -> float:
    """Compute test elapsed time from start and end frame indices.

    Parameters
    ----------
    start_frame : int
        Index of the test-start event (per the per-test §3 spec — e.g. CMJ
        countermovement onset, Linear Sprint torso crossing the start gate).
        NOT video frame 0.
    end_frame : int
        Index of the test-end event (last frame of the test window).
    fps : float
        Frame rate, must be > 0.

    Returns
    -------
    float
        Elapsed time in seconds.
    """
    return (end_frame - start_frame) / fps
