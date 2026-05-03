"""Athlete-side classification from pose data.

Pose models occasionally swap `left_*` / `right_*` keypoint labels
across frames when body orientation is ambiguous (subtle rotation, low
shoulder confidence, motion blur). Trusting those labels directly causes
foot-side counters to drift even when only one physical foot is moving.

Robust alternative: classify a foot by its IMAGE-X position relative to
the body's central axis (mid-hip), then map image-side to athlete-
anatomical side based on whether the athlete is facing the camera.
This is invariant to pose-model L/R label flips for any single
detection.

Used by foot-tapping, juggling, and straight-line-dribbling for the
left/right side metric.
"""
from __future__ import annotations

from typing import Literal

_POSE_CONF_MIN = 0.30


def body_center_x(pose) -> float | None:
    """Image-x of the athlete's central axis, derived from mid-hip.

    Returns None if both hip keypoints are below confidence threshold.
    """
    lh_c = pose.confidence_of("left_hip")
    rh_c = pose.confidence_of("right_hip")
    xs: list[float] = []
    if lh_c >= _POSE_CONF_MIN:
        xs.append(float(pose.position("left_hip")[0]))
    if rh_c >= _POSE_CONF_MIN:
        xs.append(float(pose.position("right_hip")[0]))
    if not xs:
        return None
    return sum(xs) / len(xs)


def ankle_side(
    ankle_image_x: float,
    body_center_x_value: float,
    *,
    facing_camera: bool = True,
) -> Literal["L", "R"]:
    """Athlete's anatomical L/R for an ankle from its image-x position.

    With the athlete facing the camera (default v1 assumption — most
    user-supplied test videos are front-on), the ankle on the IMAGE-LEFT
    is the athlete's anatomical RIGHT and vice versa. When facing away,
    the mapping flips.

    Per-frame orientation detection from shoulder ordering can be added
    later; for now the face-camera default holds for foot-tapping,
    juggling, and most dribbling-test footage.
    """
    is_image_left = ankle_image_x < body_center_x_value
    if facing_camera:
        return "R" if is_image_left else "L"
    return "L" if is_image_left else "R"
