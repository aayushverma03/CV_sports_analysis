"""Unit tests for Straight Line Dribbling state machine + view classifier.

Full pipeline test runs in `scripts/smoke_straight_line_dribbling.py`.
"""
from __future__ import annotations

import numpy as np
import pytest

from src.core.tracking.bytetrack_tracker import TrackedDetection
from src.tests.technical.straight_line_dribbling import (
    StraightLineDribblingTest,
    _RunState,
    _classify_view,
)


def _tracked(cx: float, cy: float, h: float = 200.0, track_id: int = 1) -> TrackedDetection:
    return TrackedDetection(
        bbox_xyxy=np.array([cx - 50, cy - h / 2, cx + 50, cy + h / 2]),
        confidence=0.8,
        class_id=0,
        track_id=track_id,
    )


def _feed(test, state, frames):
    for fi, cx, cy in frames:
        det = _tracked(cx, cy)
        # Mirror the buffer-update step that run() does before _update_state
        state.center_history.append((fi, cx, cy, det.height))
        test._update_state(state, det, fi, fps=30.0)


# --- distance validation ---------------------------------------------


def test_distance_must_be_positive():
    with pytest.raises(ValueError):
        StraightLineDribblingTest(distance_m=0)
    with pytest.raises(ValueError):
        StraightLineDribblingTest(distance_m=-5)


# --- state machine ---------------------------------------------------


def test_state_transitions_pre_start_to_running_to_stopped():
    test = StraightLineDribblingTest(distance_m=30.0)
    state = _RunState()
    # 6 frames at (cx=200, cy=400) then athlete starts moving
    pre = [(i, 200.0, 400.0) for i in range(6)]
    moving = [(i, 200.0 + (i - 5) * 20.0, 400.0) for i in range(6, 36)]
    last_x = 200.0 + (35 - 5) * 20.0
    stopping = [(i, last_x, 400.0) for i in range(36, 70)]
    _feed(test, state, pre + moving + stopping)
    assert state.state == "stopped"
    assert state.start_frame is not None
    assert state.stop_frame is not None
    assert state.stop_frame > state.start_frame


def test_state_fires_start_for_athlete_already_moving():
    """Athlete moving from frame 0 — first-motion fires start (no stationary gate)."""
    test = StraightLineDribblingTest(distance_m=30.0)
    state = _RunState()
    _feed(test, state, [(i, 200.0 + i * 30.0, 400.0) for i in range(30)])
    assert state.state == "running"
    assert state.start_frame is not None


def test_state_handles_rear_view_motion():
    """Rear-view: athlete cy changes (moving away), cx stable -> still fires start."""
    test = StraightLineDribblingTest(distance_m=30.0)
    state = _RunState()
    pre = [(i, 320.0, 480.0) for i in range(6)]
    moving_y = [(i, 320.0, 480.0 - (i - 5) * 15.0) for i in range(6, 36)]
    last_y = 480.0 - (35 - 5) * 15.0
    stopping = [(i, 320.0, last_y) for i in range(36, 70)]
    _feed(test, state, pre + moving_y + stopping)
    assert state.state == "stopped"


# --- view classifier --------------------------------------------------


def test_classify_view_side_on():
    """Athlete pixel-x range covers > 30% of frame width -> side_on."""
    state = _RunState(start_frame=15, stop_frame=44)
    # Run-window xs span 100..900 = 800 px (62% of 1280)
    state.center_history = [
        (i, 100.0 + (i - 15) * 30.0, 400.0, 200.0) for i in range(15, 45)
    ]
    assert _classify_view(state, frame_width=1280) == "side_on"


def test_classify_view_rear_view():
    """Athlete pixel-x range tiny relative to frame width -> rear_view."""
    state = _RunState(start_frame=15, stop_frame=44)
    state.center_history = [
        (i, 320.0 + np.sin(i) * 5.0, 480.0 - (i - 15) * 5.0, 200.0)
        for i in range(15, 45)
    ]
    assert _classify_view(state, frame_width=1280) == "rear_view"


def test_classify_view_unknown_when_no_run():
    state = _RunState()  # no start/stop frames
    assert _classify_view(state, frame_width=1280) == "unknown"
