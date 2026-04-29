"""Tests for camera calibration."""
from __future__ import annotations

import numpy as np
import pytest

from src.core.calibration.camera_calibration import (
    Calibration,
    CalibrationError,
    calibrate_homography,
    calibrate_linear,
)


# --- linear --------------------------------------------------------------


def test_linear_perfect_fit():
    # Linear sprint: cones at 0, 10, 20, 30 m along the x-axis at 50 px/m.
    cones_px = np.array([[0, 100], [500, 100], [1000, 100], [1500, 100]], dtype=float)
    cones_m = np.array([0.0, 10.0, 20.0, 30.0])
    cal = calibrate_linear(cones_px, cones_m)
    assert cal.px_per_m == pytest.approx(50.0)
    assert cal.rms_error_m == pytest.approx(0.0, abs=1e-9)
    assert cal.quality == "good"
    assert cal.n_points == 4


def test_linear_two_cones_minimum():
    cones_px = np.array([[0, 0], [100, 0]], dtype=float)
    cones_m = np.array([0.0, 5.0])
    cal = calibrate_linear(cones_px, cones_m)
    assert cal.px_per_m == pytest.approx(20.0)


def test_linear_to_world_roundtrip():
    cones_px = np.array([[0, 0], [200, 0]], dtype=float)
    cones_m = np.array([0.0, 10.0])
    cal = calibrate_linear(cones_px, cones_m)
    world = cal.to_world(np.array([[100, 0], [400, 0]]))
    assert np.allclose(world, [[5, 0], [20, 0]])


def test_linear_too_few_cones():
    with pytest.raises(CalibrationError, match="need >= 2"):
        calibrate_linear(np.array([[0, 0]], dtype=float), np.array([0.0]))


def test_linear_length_mismatch():
    with pytest.raises(CalibrationError, match="length mismatch"):
        calibrate_linear(
            np.array([[0, 0], [100, 0]], dtype=float),
            np.array([0.0, 5.0, 10.0]),
        )


def test_linear_non_colinear_cones_fails():
    # Cones drifting wildly off a line - RMS will exceed threshold
    cones_px = np.array(
        [[0, 0], [500, 200], [1000, -300], [1500, 800]], dtype=float
    )
    cones_m = np.array([0.0, 10.0, 20.0, 30.0])
    with pytest.raises(CalibrationError, match="exceeds marginal threshold"):
        calibrate_linear(cones_px, cones_m)


# --- homography ----------------------------------------------------------


def test_homography_square_marker_perfect_fit():
    # Four corners of a 5x5 m square in world, projected to a tilted image
    # quadrilateral. cv2.findHomography should recover this exactly.
    world = np.array([[0, 0], [5, 0], [5, 5], [0, 5]], dtype=float)
    image = np.array([[100, 200], [800, 220], [820, 600], [80, 580]], dtype=float)
    cal = calibrate_homography(image, world)
    assert cal.homography is not None
    assert cal.homography.shape == (3, 3)
    assert cal.rms_error_m == pytest.approx(0.0, abs=1e-6)
    assert cal.quality == "good"
    # Round-trip the four image corners back to world space
    reprojected = cal.to_world(image)
    assert np.allclose(reprojected, world, atol=1e-6)


def test_homography_too_few_points():
    with pytest.raises(CalibrationError, match="need >= 4"):
        calibrate_homography(
            np.array([[0, 0], [1, 0], [0, 1]], dtype=float),
            np.array([[0, 0], [1, 0], [0, 1]], dtype=float),
        )


def test_homography_length_mismatch():
    with pytest.raises(CalibrationError, match="length mismatch"):
        calibrate_homography(
            np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=float),
            np.array([[0, 0], [1, 0], [1, 1]], dtype=float),
        )


# --- Calibration class behaviour ----------------------------------------


def test_uninitialized_to_world_raises():
    cal = Calibration()
    with pytest.raises(CalibrationError, match="not initialized"):
        cal.to_world(np.array([[1.0, 2.0]]))
