"""Unit tests for Foot Tapping helpers + state.

Full pipeline run is exercised by `scripts/smoke_foot_tapping.py`.
"""
from __future__ import annotations

from src.tests.physical.foot_tapping import (
    FootTappingTest,
    _RunState,
    _Tap,
)


def test_runstate_starts_empty():
    state = _RunState()
    assert state.taps == []
    assert state.last_tap_frame < 0


def test_compute_metrics_basic():
    test = FootTappingTest.__new__(FootTappingTest)  # bypass __init__ (loads models)
    state = _RunState()
    state.taps = [
        _Tap(frame_idx=10, side="L"),
        _Tap(frame_idx=20, side="R"),
        _Tap(frame_idx=30, side="L"),
        _Tap(frame_idx=40, side="R"),
    ]
    metrics = test._compute_metrics(state, duration_s=2.0)
    assert metrics["total_taps"].raw == 4
    assert metrics["taps_per_second"].raw == 2.0
    assert metrics["left_taps"].raw == 2
    assert metrics["right_taps"].raw == 2


def test_compute_metrics_zero_duration():
    test = FootTappingTest.__new__(FootTappingTest)
    state = _RunState()
    state.taps = [_Tap(frame_idx=0, side="L")]
    metrics = test._compute_metrics(state, duration_s=0.0)
    assert metrics["total_taps"].raw == 1
    assert metrics["taps_per_second"].raw == 0.0


def test_compute_metrics_all_left():
    test = FootTappingTest.__new__(FootTappingTest)
    state = _RunState()
    state.taps = [_Tap(frame_idx=i, side="L") for i in range(5)]
    metrics = test._compute_metrics(state, duration_s=5.0)
    assert metrics["left_taps"].raw == 5
    assert metrics["right_taps"].raw == 0
    assert metrics["taps_per_second"].raw == 1.0
