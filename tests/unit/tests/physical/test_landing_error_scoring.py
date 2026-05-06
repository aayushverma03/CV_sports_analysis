"""Unit tests for the LESS subset scoring + detector helpers."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.tests.physical.landing_error_scoring import (
    _LessDetector,
    _ankle_asymmetry_frac,
    _hip_flexion_deg,
    _knee_flexion_deg,
    _score_landing_errors,
    _trunk_flexion_deg,
)


# --- _score_landing_errors --------------------------------------------


def test_score_landing_errors_perfect_landing():
    """All angles healthy + symmetric: zero errors."""
    items = _score_landing_errors(
        knee_flex_ic=35.0,
        hip_flex_ic=35.0,
        trunk_flex_ic=25.0,
        knee_flex_peak=90.0,    # +55 deg displacement
        hip_flex_peak=70.0,     # +35 deg
        trunk_flex_peak=40.0,   # +15 deg
        ankle_asymmetry_frac=0.02,
    )
    assert items.total == 0


def test_score_landing_errors_stiff_landing_all_flagged():
    """Stiff at IC + insufficient absorption + asymmetric: 7 items + overall."""
    items = _score_landing_errors(
        knee_flex_ic=20.0,                # stiff knee
        hip_flex_ic=20.0,                 # stiff hip
        trunk_flex_ic=10.0,               # upright trunk
        knee_flex_peak=30.0,              # +10 (insufficient)
        hip_flex_peak=25.0,               # +5  (insufficient)
        trunk_flex_peak=12.0,             # +2  (insufficient)
        ankle_asymmetry_frac=0.10,        # asymmetric
    )
    assert items.stiff_knee_at_ic
    assert items.stiff_hip_at_ic
    assert items.upright_trunk_at_ic
    assert items.insufficient_knee_disp
    assert items.insufficient_hip_disp
    assert items.insufficient_trunk_disp
    assert items.asymmetric_initial_contact
    assert items.overall_impression
    assert items.total == 8


def test_score_landing_errors_overall_threshold_at_4():
    """4 individual flags trigger overall_impression."""
    items = _score_landing_errors(
        knee_flex_ic=20.0,                # flag 1
        hip_flex_ic=20.0,                 # flag 2
        trunk_flex_ic=10.0,               # flag 3
        knee_flex_peak=30.0,              # flag 4 (insuff knee)
        hip_flex_peak=80.0,               # ok
        trunk_flex_peak=40.0,             # ok
        ankle_asymmetry_frac=0.02,        # ok
    )
    assert items.total == 5  # 4 individual + overall_impression


def test_score_landing_errors_three_flags_no_overall():
    """3 individual flags do not trigger overall_impression."""
    items = _score_landing_errors(
        knee_flex_ic=20.0,
        hip_flex_ic=20.0,
        trunk_flex_ic=10.0,
        knee_flex_peak=80.0,              # ok (60 deg)
        hip_flex_peak=80.0,               # ok
        trunk_flex_peak=40.0,             # ok
        ankle_asymmetry_frac=0.02,        # ok
    )
    assert items.total == 3
    assert not items.overall_impression


def test_score_landing_errors_displacement_at_threshold_is_error():
    """Displacement < threshold flags; equal-to threshold does NOT."""
    items_below = _score_landing_errors(
        knee_flex_ic=40.0, hip_flex_ic=40.0, trunk_flex_ic=25.0,
        knee_flex_peak=84.99,             # +44.99 < 45 -> insuff
        hip_flex_peak=80.0,
        trunk_flex_peak=40.0,
        ankle_asymmetry_frac=0.02,
    )
    assert items_below.insufficient_knee_disp

    items_at = _score_landing_errors(
        knee_flex_ic=40.0, hip_flex_ic=40.0, trunk_flex_ic=25.0,
        knee_flex_peak=85.0,              # +45 == threshold -> not flagged
        hip_flex_peak=80.0,
        trunk_flex_peak=40.0,
        ankle_asymmetry_frac=0.02,
    )
    assert not items_at.insufficient_knee_disp


# --- joint angle helpers (synthetic poses) -----------------------------


@dataclass
class _FakePose:
    """Just enough of PoseDetection for the angle helpers."""

    points: dict[str, tuple[float, float]]
    confs: dict[str, float]

    def position(self, name: str) -> np.ndarray:
        return np.asarray(self.points[name], dtype=float)

    def confidence_of(self, name: str) -> float:
        return self.confs.get(name, 0.0)


def _full_conf_pose(points: dict[str, tuple[float, float]]) -> _FakePose:
    return _FakePose(points=points, confs={k: 1.0 for k in points})


def test_knee_flexion_straight_leg_is_zero():
    """Hip directly above knee directly above ankle: 0 deg flexion."""
    pose = _full_conf_pose({
        "left_hip": (100, 100), "left_knee": (100, 200), "left_ankle": (100, 300),
        "right_hip": (200, 100), "right_knee": (200, 200), "right_ankle": (200, 300),
    })
    assert abs(_knee_flexion_deg(pose) - 0.0) < 1e-6


def test_knee_flexion_right_angle_is_90():
    """Hip directly above knee, ankle 100 px to the right of knee:
    interior angle = 90 deg, so flexion = 90 deg."""
    pose = _full_conf_pose({
        "left_hip": (100, 100), "left_knee": (100, 200), "left_ankle": (200, 200),
        "right_hip": (300, 100), "right_knee": (300, 200), "right_ankle": (400, 200),
    })
    assert abs(_knee_flexion_deg(pose) - 90.0) < 1e-6


def test_knee_flexion_returns_none_when_low_confidence():
    pose = _FakePose(
        points={
            "left_hip": (100, 100), "left_knee": (100, 200), "left_ankle": (100, 300),
            "right_hip": (200, 100), "right_knee": (200, 200), "right_ankle": (200, 300),
        },
        confs={
            "left_hip": 0.1, "left_knee": 1.0, "left_ankle": 1.0,
            "right_hip": 0.1, "right_knee": 1.0, "right_ankle": 1.0,
        },
    )
    assert _knee_flexion_deg(pose) is None


def test_hip_flexion_upright_trunk_is_zero():
    """Shoulder directly above hip directly above knee: 0 deg."""
    pose = _full_conf_pose({
        "left_shoulder": (100, 50),
        "left_hip": (100, 150),
        "left_knee": (100, 250),
        "right_shoulder": (200, 50),
        "right_hip": (200, 150),
        "right_knee": (200, 250),
    })
    assert abs(_hip_flexion_deg(pose) - 0.0) < 1e-6


def test_trunk_flexion_upright_is_zero_and_horizontal_is_90():
    """Mid-shoulder directly above mid-hip => 0; shoulder directly to
    the side => 90 deg lean."""
    upright = _full_conf_pose({
        "left_shoulder": (100, 50), "right_shoulder": (200, 50),
        "left_hip": (100, 150), "right_hip": (200, 150),
    })
    assert abs(_trunk_flexion_deg(upright) - 0.0) < 1e-6

    horizontal = _full_conf_pose({
        # mid-shoulder = (200, 100), mid-hip = (100, 100): horizontal trunk.
        "left_shoulder": (150, 100), "right_shoulder": (250, 100),
        "left_hip": (50, 100), "right_hip": (150, 100),
    })
    assert abs(_trunk_flexion_deg(horizontal) - 90.0) < 1e-6


def test_ankle_asymmetry_frac():
    pose = _full_conf_pose({
        "left_ankle": (100, 200),
        "right_ankle": (200, 250),
    })
    # |200 - 250| / bbox_h = 50 / 500 = 0.10
    assert abs(_ankle_asymmetry_frac(pose, bbox_h=500.0) - 0.10) < 1e-6


# --- _LessDetector -----------------------------------------------------


def test_less_detector_finds_initial_contact():
    """Athlete on box for 20 frames at y=100, drops to y=400 over 10
    frames, plateaus there. Detector should locate the IC frame."""
    det = _LessDetector()
    bbox_h = 500.0
    # On box: ankle at y=100 for 20 frames.
    for fi in range(20):
        det.update(fi, ankle_y=100.0, bbox_h=bbox_h)
    assert det.state == "on_box"
    # Drop from 100 -> 400 over 10 frames (linear).
    for k, fi in enumerate(range(20, 30)):
        det.update(fi, ankle_y=100.0 + 30.0 * (k + 1), bbox_h=bbox_h)
    # Plateau at y=400 for 5 frames.
    for fi in range(30, 35):
        det.update(fi, ankle_y=400.0, bbox_h=bbox_h)
    assert det.state == "contact_1"
    assert det.initial_contact_frame is not None
    # IC should be roughly at frame 30 (start of plateau).
    assert 29 <= det.initial_contact_frame <= 31


def test_less_detector_stays_on_box_when_no_drop():
    """No motion: detector never leaves on_box."""
    det = _LessDetector()
    for fi in range(40):
        det.update(fi, ankle_y=100.0, bbox_h=500.0)
    assert det.state == "on_box"
    assert det.initial_contact_frame is None


def test_less_detector_handles_missing_pose_frames():
    """None ankle_y is tolerated mid-stream."""
    det = _LessDetector()
    bbox_h = 500.0
    for fi in range(20):
        det.update(fi, ankle_y=100.0, bbox_h=bbox_h)
    det.update(20, ankle_y=None, bbox_h=None)
    for k, fi in enumerate(range(21, 31)):
        det.update(fi, ankle_y=100.0 + 30.0 * (k + 1), bbox_h=bbox_h)
    for fi in range(31, 36):
        det.update(fi, ankle_y=400.0, bbox_h=bbox_h)
    assert det.state == "contact_1"
    assert det.initial_contact_frame is not None
