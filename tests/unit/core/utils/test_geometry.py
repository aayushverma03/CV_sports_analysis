"""Tests for geometry helpers."""
from __future__ import annotations

import numpy as np
import pytest

from src.core.utils.geometry import (
    angle_3pt,
    angle_between,
    apply_homography,
    m_to_px,
    pixel_distance,
    px_to_m,
    signed_angle,
    vector_angle_deg,
)


# --- Pixel <-> world ----------------------------------------------------


def test_px_m_roundtrip():
    px_per_m = 50.0
    d = 3.4
    assert px_to_m(m_to_px(d, px_per_m), px_per_m) == pytest.approx(d)


def test_apply_homography_identity():
    pts = np.array([[10.0, 20.0], [30.0, 40.0]])
    H = np.eye(3)
    out = apply_homography(pts, H)
    assert np.allclose(out, pts)


def test_apply_homography_translation():
    pts = np.array([[0.0, 0.0], [5.0, 5.0]])
    H = np.array([[1, 0, 7], [0, 1, -3], [0, 0, 1]], dtype=float)
    out = apply_homography(pts, H)
    assert np.allclose(out, [[7, -3], [12, 2]])


def test_apply_homography_scale():
    pts = np.array([[1.0, 2.0]])
    H = np.diag([2.0, 3.0, 1.0])
    out = apply_homography(pts, H)
    assert np.allclose(out, [[2.0, 6.0]])


def test_apply_homography_bad_shape():
    with pytest.raises(ValueError):
        apply_homography(np.array([1.0, 2.0]), np.eye(3))


def test_pixel_distance():
    assert pixel_distance([0, 0], [3, 4]) == pytest.approx(5.0)
    assert pixel_distance([1, 1], [1, 1]) == pytest.approx(0.0)


# --- Angle math ---------------------------------------------------------


def test_vector_angle_deg():
    assert vector_angle_deg([1, 0]) == pytest.approx(0.0)
    assert vector_angle_deg([0, 1]) == pytest.approx(90.0)
    assert vector_angle_deg([-1, 0]) == pytest.approx(180.0)
    assert vector_angle_deg([0, -1]) == pytest.approx(-90.0)


@pytest.mark.parametrize(
    "v1,v2,expected",
    [
        ([1, 0], [1, 0], 0.0),
        ([1, 0], [0, 1], 90.0),
        ([1, 0], [-1, 0], 180.0),
        ([1, 1], [-1, -1], 180.0),
        ([1, 0], [1, 1], 45.0),
    ],
)
def test_angle_between(v1, v2, expected):
    assert angle_between(v1, v2) == pytest.approx(expected, abs=1e-3)


def test_angle_between_zero_vector_raises():
    with pytest.raises(ValueError):
        angle_between([0, 0], [1, 0])


def test_angle_3pt_straight_line():
    # a-b-c colinear => 180
    assert angle_3pt([0, 0], [1, 0], [2, 0]) == pytest.approx(180.0)


def test_angle_3pt_right_angle():
    # knee at (0,0), hip up, ankle right => 90
    assert angle_3pt([0, 1], [0, 0], [1, 0]) == pytest.approx(90.0)


def test_angle_3pt_acute():
    # squat-like fold at the joint
    assert angle_3pt([1, 1], [0, 0], [1, -1]) == pytest.approx(90.0)


@pytest.mark.parametrize(
    "v1,v2,expected_sign",
    [
        ([1, 0], [0, 1], 1),    # CCW
        ([1, 0], [0, -1], -1),  # CW
    ],
)
def test_signed_angle_sign(v1, v2, expected_sign):
    angle = signed_angle(v1, v2)
    assert np.sign(angle) == expected_sign
    assert abs(angle) == pytest.approx(90.0)


def test_signed_angle_full_range():
    # full 180 (anti-parallel) returns +180 not -180 by atan2 convention
    assert abs(signed_angle([1, 0], [-1, 0])) == pytest.approx(180.0)
