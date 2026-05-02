"""Unit tests for T-Test state machine."""
from __future__ import annotations

import numpy as np

from src.core.tracking.bytetrack_tracker import TrackedDetection
from src.tests.physical.t_test import (
    TTestTest,
    _RunState,
)


def _tracked(cx: float, cy: float, h: float = 200.0) -> TrackedDetection:
    return TrackedDetection(
        bbox_xyxy=np.array([cx - 50, cy - h / 2, cx + 50, cy + h / 2]),
        confidence=0.8,
        class_id=0,
        track_id=1,
    )


def _feed(test, state, frames):
    for fi, cx, cy in frames:
        det = _tracked(cx, cy)
        state.center_history.append((fi, cx, cy, det.height))
        test._update_state(state, det, fi)


def test_pre_start_to_running_to_stopped():
    """Min run is 180 frames (6 s); feed enough running frames to satisfy."""
    test = TTestTest.__new__(TTestTest)  # bypass __init__ (loads model)
    state = _RunState()
    pre = [(i, 200.0, 400.0) for i in range(15)]
    moving = [(i, 200.0 + (i - 14) * 5.0, 400.0) for i in range(15, 215)]
    last_x = 200.0 + (214 - 14) * 5.0
    stopping = [(i, last_x, 400.0) for i in range(215, 280)]
    _feed(test, state, pre + moving + stopping)
    assert state.state == "stopped"
    assert state.stationary_confirmed
    assert state.start_frame is not None
    assert state.stop_frame is not None
    assert state.stop_frame > state.start_frame


def test_no_stationary_period_blocks_start():
    test = TTestTest.__new__(TTestTest)
    state = _RunState()
    # Athlete moving from frame 0
    _feed(test, state, [(i, 200.0 + i * 25.0, 400.0) for i in range(60)])
    assert state.state == "pre_start"
    assert not state.stationary_confirmed


def test_handles_2d_motion():
    """Athlete moves in both x and y (T-Test has lateral + longitudinal motion)."""
    test = TTestTest.__new__(TTestTest)
    state = _RunState()
    pre = [(i, 320.0, 400.0) for i in range(15)]
    # Mixed x/y motion over 200 frames (~6.7 s) to clear MIN_RUN_FRAMES
    moving = [(i, 320.0 + (i - 14) * 2.0, 400.0 + (i - 14) * 1.0)
              for i in range(15, 215)]
    last_x = 320.0 + 200 * 2.0
    last_y = 400.0 + 200 * 1.0
    stopping = [(i, last_x, last_y) for i in range(215, 280)]
    _feed(test, state, pre + moving + stopping)
    assert state.state == "stopped"
