"""Tests for the CMJ streaming event detector.

The full `CounterMovementJumpTest.run()` needs a real video and a loaded
RTMPose model — exercised via the smoke script, not here.
"""
from __future__ import annotations

from src.tests.physical.counter_movement_jump import _StreamingDetector


def _feed(detector, ankle_y_series, bbox_h: float = 200.0):
    for i, y in enumerate(ankle_y_series):
        detector.update(i, y, bbox_h)


def test_streaming_detector_no_jump_yields_no_candidates():
    det = _StreamingDetector()
    _feed(det, [500.0] * 60)
    assert det.candidates == []
    assert det.best_jump() is None


def test_streaming_detector_locates_single_jump():
    det = _StreamingDetector()
    # 30 frames standing -> baseline 500, threshold 500 - 0.05*200 = 490
    standing = [500.0] * 30
    airborne = [400.0] * 10
    landed = [500.0] * 10
    _feed(det, standing + airborne + landed)
    assert det.candidates == [(30, 40)]
    assert det.best_jump() == (30, 40)


def test_streaming_detector_picks_longest_of_multiple_candidates():
    """Walking step (3 frames) + actual jump (15 frames) -> picks the jump."""
    det = _StreamingDetector()
    series = (
        [500.0] * 30                          # standing baseline
        + [400.0] * 3 + [500.0] * 5           # short walking step
        + [400.0] * 15 + [500.0] * 5          # the real jump
    )
    _feed(det, series)
    # Two candidates: (30, 33) length 3, (38, 53) length 15
    assert len(det.candidates) == 2
    assert det.best_jump() == (38, 53)


def test_streaming_detector_rejects_single_frame_spike():
    det = _StreamingDetector()
    series = [500.0] * 30 + [400.0] + [500.0] * 10
    _feed(det, series)
    assert det.candidates == []
    assert det.best_jump() is None


def test_streaming_detector_ignores_missing_frames():
    det = _StreamingDetector()
    for i in range(30):
        det.update(i, 500.0, 200.0)
    for i in range(30, 35):
        det.update(i, None, None)  # gap
    for i in range(35, 50):
        det.update(i, 400.0, 200.0)
    for i in range(50, 60):
        det.update(i, 500.0, 200.0)
    assert det.candidates == [(35, 50)]
    assert det.best_jump() == (35, 50)
