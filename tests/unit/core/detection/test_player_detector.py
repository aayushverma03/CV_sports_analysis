"""Tests for the player detector."""
from __future__ import annotations

import numpy as np
import pytest

from src.core.detection.player_detector import (
    PERSON_CLASS_ID,
    Detection,
    detect_players,
)


# --- Detection dataclass --------------------------------------------------


def test_detection_center():
    d = Detection(bbox_xyxy=np.array([10.0, 20.0, 30.0, 60.0]),
                   confidence=0.9, class_id=0)
    assert np.allclose(d.center, [20.0, 40.0])


def test_detection_width_height():
    d = Detection(bbox_xyxy=np.array([10.0, 20.0, 30.0, 60.0]),
                   confidence=0.9, class_id=0)
    assert d.width == 20.0
    assert d.height == 40.0


def test_person_class_id():
    assert PERSON_CLASS_ID == 0


# --- detect_players (real model) -----------------------------------------


def test_detect_players_returns_list_on_blank_frame():
    """A solid-grey frame should detect nothing — exercises the call path."""
    frame = np.full((480, 640, 3), 128, dtype=np.uint8)
    detections = detect_players(frame)
    assert isinstance(detections, list)
    assert all(isinstance(d, Detection) for d in detections)


def test_detect_players_filters_to_person_class():
    """All returned detections must be class 0 (person)."""
    frame = np.full((480, 640, 3), 128, dtype=np.uint8)
    detections = detect_players(frame)
    assert all(d.class_id == PERSON_CLASS_ID for d in detections)


def test_detect_players_sorted_by_confidence():
    frame = np.full((480, 640, 3), 128, dtype=np.uint8)
    detections = detect_players(frame)
    confs = [d.confidence for d in detections]
    assert confs == sorted(confs, reverse=True)


def test_confidence_override(monkeypatch):
    """Higher confidence threshold yields fewer-or-equal detections."""
    frame = np.full((480, 640, 3), 128, dtype=np.uint8)
    a = detect_players(frame, confidence=0.1)
    b = detect_players(frame, confidence=0.99)
    assert len(b) <= len(a)
