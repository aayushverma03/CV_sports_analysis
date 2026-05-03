"""Tests for the athlete-side helper."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.core.pose.orientation import ankle_side, body_center_x


@dataclass
class _FakePose:
    """Stand-in for PoseDetection — only the methods we need."""
    keypoints: dict[str, tuple[float, float, float]]  # name -> (x, y, conf)

    def position(self, name: str) -> np.ndarray:
        return np.array(self.keypoints[name][:2])

    def confidence_of(self, name: str) -> float:
        return self.keypoints[name][2]


# --- body_center_x ----------------------------------------------------


def test_body_center_x_averages_both_hips():
    pose = _FakePose({
        "left_hip": (300.0, 400.0, 0.9),
        "right_hip": (200.0, 400.0, 0.9),
    })
    assert body_center_x(pose) == 250.0


def test_body_center_x_uses_one_hip_when_other_low_conf():
    pose = _FakePose({
        "left_hip": (300.0, 400.0, 0.9),
        "right_hip": (200.0, 400.0, 0.10),  # below threshold
    })
    assert body_center_x(pose) == 300.0


def test_body_center_x_returns_none_when_both_low_conf():
    pose = _FakePose({
        "left_hip": (300.0, 400.0, 0.10),
        "right_hip": (200.0, 400.0, 0.10),
    })
    assert body_center_x(pose) is None


# --- ankle_side -------------------------------------------------------


def test_ankle_side_image_left_facing_camera_is_athlete_right():
    """Athlete faces camera; ankle on image-left = athlete's right leg."""
    assert ankle_side(ankle_image_x=200.0, body_center_x_value=300.0) == "R"


def test_ankle_side_image_right_facing_camera_is_athlete_left():
    assert ankle_side(ankle_image_x=400.0, body_center_x_value=300.0) == "L"


def test_ankle_side_facing_away_flips():
    """Athlete faces away; image-left = athlete's left, image-right = athlete's right."""
    assert ankle_side(200.0, 300.0, facing_camera=False) == "L"
    assert ankle_side(400.0, 300.0, facing_camera=False) == "R"


def test_ankle_side_robust_to_pose_label_flip():
    """The same physical position always returns the same side, no
    matter which keypoint name produced it."""
    # Imagine a foot on image-left at x=200, body center at 300.
    # Whether the pose model labelled it "left_ankle" or "right_ankle",
    # ankle_side returns 'R' (face-camera default). That's the point.
    assert ankle_side(200.0, 300.0) == "R"
    assert ankle_side(200.0, 300.0) == "R"  # called again, same answer
