"""Geometric primitives — pixel/world conversions and angle math.

Pure functions, no module state. Used by detection, tracking, pose,
calibration, and metric layers.
"""
from __future__ import annotations

import numpy as np

# --- Pixel <-> world ----------------------------------------------------


def px_to_m(d_px: float, px_per_m: float) -> float:
    """Convert a pixel distance to metres using a uniform scale.

    Parameters
    ----------
    d_px : float
        Distance in pixels.
    px_per_m : float
        Calibrated pixels-per-metre ratio (positive).

    Returns
    -------
    float
        Distance in metres.
    """
    return d_px / px_per_m


def m_to_px(d_m: float, px_per_m: float) -> float:
    """Convert a metre distance to pixels using a uniform scale."""
    return d_m * px_per_m


def apply_homography(points: np.ndarray, H: np.ndarray) -> np.ndarray:
    """Project N x 2 points through a 3 x 3 homography matrix.

    Parameters
    ----------
    points : np.ndarray, shape (N, 2)
        Source points, e.g. pixel coordinates.
    H : np.ndarray, shape (3, 3)
        Homography mapping source to target frame.

    Returns
    -------
    np.ndarray, shape (N, 2)
        Points in the target frame.
    """
    pts = np.asarray(points, dtype=float)
    if pts.ndim != 2 or pts.shape[1] != 2:
        raise ValueError(f"expected (N, 2), got {pts.shape}")
    homog = np.hstack([pts, np.ones((pts.shape[0], 1))])
    transformed = homog @ np.asarray(H, dtype=float).T
    return transformed[:, :2] / transformed[:, 2:3]


def pixel_distance(p1: np.ndarray, p2: np.ndarray) -> float:
    """Euclidean distance between two 2D points (any units)."""
    return float(np.linalg.norm(np.asarray(p1, dtype=float) - np.asarray(p2, dtype=float)))


# --- Angle math ---------------------------------------------------------


def vector_angle_deg(v: np.ndarray) -> float:
    """Angle of a 2D vector from the +x axis, in degrees, range (-180, 180]."""
    v = np.asarray(v, dtype=float)
    return float(np.degrees(np.arctan2(v[1], v[0])))


def angle_between(v1: np.ndarray, v2: np.ndarray) -> float:
    """Unsigned angle between two 2D vectors, in degrees, range [0, 180].

    Raises
    ------
    ValueError
        If either vector has zero length.
    """
    a = np.asarray(v1, dtype=float)
    b = np.asarray(v2, dtype=float)
    n1 = np.linalg.norm(a)
    n2 = np.linalg.norm(b)
    if n1 == 0.0 or n2 == 0.0:
        raise ValueError("zero-length vector")
    cos_theta = float(np.clip(np.dot(a, b) / (n1 * n2), -1.0, 1.0))
    return float(np.degrees(np.arccos(cos_theta)))


def angle_3pt(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """Interior angle at point b formed by points a, b, c, in degrees, range [0, 180].

    Standard convention for joint angles (knee, hip, elbow): vectors from
    the joint outward to the two adjacent landmarks.

    Raises
    ------
    ValueError
        If a == b or c == b.
    """
    return angle_between(np.asarray(a) - np.asarray(b), np.asarray(c) - np.asarray(b))


def signed_angle(v1: np.ndarray, v2: np.ndarray) -> float:
    """Signed 2D angle from v1 to v2, in degrees, range (-180, 180].

    Positive = counter-clockwise. Useful for change-of-direction analysis.
    """
    a = np.asarray(v1, dtype=float)
    b = np.asarray(v2, dtype=float)
    return float(np.degrees(np.arctan2(a[0] * b[1] - a[1] * b[0], np.dot(a, b))))
