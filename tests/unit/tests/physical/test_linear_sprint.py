"""Tests for the Linear Sprint helpers + state machine.

The full `LinearSprintTest.run()` needs YOLO-World loaded; that's covered
by the smoke script.
"""
from __future__ import annotations

import numpy as np
import pytest

from src.core.detection.player_detector import Detection
from src.core.tracking.bytetrack_tracker import TrackedDetection
from src.tests.physical.linear_sprint import (
    SUPPORTED_DISTANCES_M,
    LinearSprintTest,
    _RunState,
    _consensus_clusters,
)


def _det(cx: float, cy: float) -> Detection:
    return Detection(
        bbox_xyxy=np.array([cx - 10, cy - 10, cx + 10, cy + 10]),
        confidence=0.5,
        class_id=0,
    )


def _tracked(cx: float, cy: float, h: float = 200.0, track_id: int = 1) -> TrackedDetection:
    return TrackedDetection(
        bbox_xyxy=np.array([cx - 50, cy - h / 2, cx + 50, cy + h / 2]),
        confidence=0.8,
        class_id=0,
        track_id=track_id,
    )


# --- consensus clustering --------------------------------------------


def test_consensus_returns_count_sorted_descending():
    detections = [_det(100, 200)] * 5 + [_det(500, 200)] * 8
    out = _consensus_clusters(detections, radius_px=30, min_count=5)
    assert [count for _, count in out] == [8, 5]
    assert out[0][0][0] == 500.0  # highest count first


def test_consensus_filters_below_min_count():
    detections = [_det(100, 200)] * 4 + [_det(500, 200)] * 6
    out = _consensus_clusters(detections, radius_px=30, min_count=5)
    assert len(out) == 1
    assert out[0][0][0] == 500.0


# --- state machine ---------------------------------------------------


def _new_state_with_cone(cone_x: float = 1000.0) -> _RunState:
    state = _RunState(finish_cone_px=np.array([cone_x, 400.0]))
    return state


def _feed_stationary(test, state, frames: int, x: float = 200.0):
    """Feed the state machine `frames` of athlete stationary at `x`."""
    for i in range(frames):
        test._update_run_state(state, _tracked(x, 400.0), i, fps=30.0)


def test_run_state_detects_start_after_stationary_then_motion():
    """Athlete stationary for 15 frames, then moves -> start_frame fires."""
    test = LinearSprintTest(distance_m=10.0)
    state = _new_state_with_cone(1000.0)
    _feed_stationary(test, state, 15, x=200.0)
    assert state.state == "pre_start"
    assert state.stationary_confirmed
    # Athlete moves to x=300 (well above motion threshold)
    test._update_run_state(state, _tracked(300.0, 400.0), 15, fps=30.0)
    assert state.state == "running"
    assert state.direction_sign == 1
    assert state.start_frame == 10  # 5 frames before frame 15


def test_run_state_does_not_fire_start_without_stationary_period():
    """Athlete moving from frame 0 — never enters running state."""
    test = LinearSprintTest(distance_m=10.0)
    state = _new_state_with_cone(1000.0)
    # Athlete already running at frame 0, x increases steadily
    for i in range(20):
        test._update_run_state(state, _tracked(200.0 + i * 30, 400.0), i, fps=30.0)
    assert state.state == "pre_start"
    assert not state.stationary_confirmed


def test_run_state_finish_when_athlete_crosses_cone():
    test = LinearSprintTest(distance_m=10.0)
    state = _new_state_with_cone(1000.0)
    _feed_stationary(test, state, 15, x=200.0)
    test._update_run_state(state, _tracked(300.0, 400.0), 15, fps=30.0)
    assert state.state == "running"
    # Continue accelerating across cone
    for i in range(16, 30):
        test._update_run_state(state, _tracked(200 + (i - 14) * 60, 400.0), i, fps=30.0)
    assert state.state == "finished"
    assert state.finish_frame is not None


def test_run_state_picks_left_direction_when_athlete_moves_left():
    test = LinearSprintTest(distance_m=10.0)
    state = _new_state_with_cone(200.0)
    _feed_stationary(test, state, 15, x=1000.0)
    test._update_run_state(state, _tracked(900.0, 400.0), 15, fps=30.0)
    assert state.state == "running"
    assert state.direction_sign == -1


# --- LinearSprintTest config -----------------------------------------


def test_distance_m_validated():
    with pytest.raises(ValueError):
        LinearSprintTest(distance_m=15.0)


def test_supports_all_four_distances():
    for d in SUPPORTED_DISTANCES_M:
        # Constructor loads MarkerDetector and tracker — skip if model missing
        try:
            test = LinearSprintTest(distance_m=d)
        except FileNotFoundError:
            pytest.skip("YOLO-World weights missing")
        assert test._distance_m == d
