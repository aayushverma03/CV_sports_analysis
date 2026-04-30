"""Tests for the Drop Jump streaming event detector.

The full `DropJumpTest.run()` needs a real video and pose model loaded —
exercised via the smoke script.
"""
from __future__ import annotations

from src.tests.physical.drop_jump import _DropJumpDetector


def _feed(detector, ankle_y_series, bbox_h: float = 200.0):
    for i, y in enumerate(ankle_y_series):
        detector.update(i, y, bbox_h)


def test_detector_initial_state_is_on_box():
    det = _DropJumpDetector()
    assert det.state == "on_box"


def test_detector_locks_box_level_after_warmup():
    det = _DropJumpDetector()
    _feed(det, [400.0] * 15)  # 15 frames stationary on box
    assert det.box_y == 400.0


def test_detector_full_drop_jump_sequence():
    """on_box -> dropping -> contact_1 -> rebound -> contact_2 (done)."""
    det = _DropJumpDetector()
    bbox_h = 200.0
    # 15 frames on box at y=400
    on_box = [400.0] * 15
    # 5 frames dropping (ankle_y rising rapidly)
    dropping = [420.0, 440.0, 470.0, 500.0, 530.0]
    # 5 frames first ground contact (stable)
    contact_1 = [550.0, 550.0, 550.0, 550.0, 550.0]
    # 6 frames rebound airborne (ankle_y drops then comes back)
    rebound = [530.0, 510.0, 500.0, 510.0, 530.0, 545.0]
    # back near ground
    final = [550.0, 550.0]
    _feed(det, on_box + dropping + contact_1 + rebound + final, bbox_h=bbox_h)

    assert det.state == "done"
    assert det.step_off_frame is not None
    assert det.first_landing_frame is not None
    assert det.rebound_takeoff_frame is not None
    assert det.rebound_landing_frame is not None
    # First landing = first frame the ankle is at ground level (delta from
    # prior frame is small). Frame 19 = y=530 (still falling); frame 20 =
    # y=550 (first frame on ground). Provisional landing locks at 20.
    assert det.first_landing_frame == 20
    # Rebound takeoff: ground_y=550, lift threshold = 550 - 0.05*200 = 540.
    # rebound[0] frame 25 has y=530 < 540 -> first liftoff frame.
    assert det.rebound_takeoff_frame == 25
    # Rebound landing: ankle returns to within 0.03*200 = 6 of ground 550
    # (>= 544). Frame 30 has y=545 -> landed.
    assert det.rebound_landing_frame == 30


def test_detector_rejects_no_drop():
    """Athlete stays on box, never drops -> state machine stays at on_box."""
    det = _DropJumpDetector()
    _feed(det, [400.0] * 60)
    assert det.state == "on_box"
    assert det.first_landing_frame is None


def test_detector_handles_missing_frames():
    """update(None, None) does not crash or shift state."""
    det = _DropJumpDetector()
    _feed(det, [400.0] * 15)  # warmup
    assert det.box_y == 400.0
    # Insert a gap of None frames
    for i in range(15, 20):
        det.update(i, None, None)
    # Drop continues
    _feed_offset = lambda series, off: [(off + i, y) for i, y in enumerate(series)]
    for i, y in _feed_offset([420.0, 440.0, 470.0, 500.0, 530.0], 20):
        det.update(i, y, 200.0)
    assert det.state == "dropping"
    assert det.step_off_frame is not None
