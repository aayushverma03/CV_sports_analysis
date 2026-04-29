"""Tests for ByteTrack tracker wrapper."""
from __future__ import annotations

import numpy as np

from src.core.detection.player_detector import Detection
from src.core.tracking.bytetrack_tracker import ByteTrackTracker, TrackedDetection


def test_tracked_detection_inherits_detection():
    td = TrackedDetection(
        bbox_xyxy=np.array([10.0, 20.0, 30.0, 60.0]),
        confidence=0.9,
        class_id=0,
        track_id=7,
    )
    assert isinstance(td, Detection)
    assert td.track_id == 7
    assert np.allclose(td.center, [20.0, 40.0])
    assert td.width == 20.0
    assert td.height == 40.0


def test_tracker_constructs_with_defaults():
    t = ByteTrackTracker()
    assert t._classes == [0]  # default = person


def test_tracker_constructs_with_overrides():
    t = ByteTrackTracker(classes=[0, 32], confidence=0.5, iou=0.6)
    assert t._classes == [0, 32]
    assert t._conf == 0.5
    assert t._iou == 0.6


def test_blank_frame_returns_list():
    t = ByteTrackTracker()
    out = t.update(np.full((480, 640, 3), 128, dtype=np.uint8))
    assert isinstance(out, list)
    assert all(isinstance(d, TrackedDetection) for d in out)


def test_persist_flag_flips_after_first_update():
    t = ByteTrackTracker()
    assert t._initialized is False
    t.update(np.full((480, 640, 3), 128, dtype=np.uint8))
    assert t._initialized is True
