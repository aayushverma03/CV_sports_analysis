"""Camera calibration — pixel <-> world via cones or known markers.

Hard rule #7: when a test requires real-world distances and calibration
fails, raise `CalibrationError`. No silent fallback to pixel space.

Two modes:

- `calibrate_linear` — uniform px-per-m scale from N cones along a line
  (sprint family, single-axis distance tests).
- `calibrate_homography` — full 3x3 perspective mapping from >=4
  image-world point pairs (agility, dribbling, T-Test, etc.).

Both return a `Calibration` whose `to_world()` method maps pixel points to
metres.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import cv2
import numpy as np

from src.core.utils.geometry import apply_homography, pixel_distance

_QUALITY_GOOD_RMS_M = 0.05
_QUALITY_MARGINAL_RMS_M = 0.20


class CalibrationError(RuntimeError):
    """Raised when pixel-to-metre ratio cannot be established."""


@dataclass(frozen=True)
class Calibration:
    """Pixel <-> world mapping. Exactly one of `px_per_m` or `homography` is set."""

    px_per_m: float | None = None
    homography: np.ndarray | None = None
    rms_error_m: float = 0.0
    n_points: int = 0
    quality: Literal["good", "marginal"] = "good"

    def to_world(self, points_px: np.ndarray) -> np.ndarray:
        """Map N x 2 pixel points to N x 2 world coords (metres)."""
        if self.px_per_m is not None:
            return np.asarray(points_px, dtype=float) / self.px_per_m
        if self.homography is not None:
            return apply_homography(points_px, self.homography)
        raise CalibrationError("calibration not initialized")


def calibrate_linear(
    cone_positions_px: np.ndarray,
    cone_distances_m: np.ndarray,
) -> Calibration:
    """Fit a uniform pixel-to-metre scale from N cones along a line.

    Parameters
    ----------
    cone_positions_px : np.ndarray, shape (N, 2)
        Pixel coordinates of each cone, in order. N >= 2.
    cone_distances_m : np.ndarray, shape (N,)
        World distance from cone 0 to each cone, in metres. First entry
        must be 0 (cone 0 is the reference).

    Returns
    -------
    Calibration
        Linear-mode calibration with `px_per_m`.

    Raises
    ------
    CalibrationError
        If fewer than 2 cones, the fit yields a non-positive scale, or RMS
        reprojection error exceeds the marginal threshold (cones not
        approximately colinear).
    """
    pts = np.asarray(cone_positions_px, dtype=float)
    dists = np.asarray(cone_distances_m, dtype=float)
    if pts.shape[0] != dists.shape[0]:
        raise CalibrationError(
            f"length mismatch: {pts.shape[0]} cones vs {dists.shape[0]} distances"
        )
    if pts.shape[0] < 2:
        raise CalibrationError(f"need >= 2 cones, got {pts.shape[0]}")

    pixel_dists = np.array([pixel_distance(pts[0], p) for p in pts])

    # Least-squares slope through origin: px_per_m = sum(d_px * d_m) / sum(d_m^2)
    valid = dists > 0
    if not np.any(valid):
        raise CalibrationError("all cone distances are zero")
    px_per_m = float(
        np.sum(pixel_dists[valid] * dists[valid]) / np.sum(dists[valid] ** 2)
    )
    if px_per_m <= 0:
        raise CalibrationError(f"non-positive scale: {px_per_m}")

    predicted_m = pixel_dists / px_per_m
    rms = float(np.sqrt(np.mean((predicted_m - dists) ** 2)))

    if rms >= _QUALITY_MARGINAL_RMS_M:
        raise CalibrationError(
            f"calibration RMS {rms:.3f} m exceeds marginal threshold "
            f"{_QUALITY_MARGINAL_RMS_M} m — cones likely not colinear"
        )

    quality: Literal["good", "marginal"] = "good" if rms < _QUALITY_GOOD_RMS_M else "marginal"
    return Calibration(
        px_per_m=px_per_m, rms_error_m=rms, n_points=len(pts), quality=quality
    )


def calibrate_homography(
    image_points_px: np.ndarray,
    world_points_m: np.ndarray,
) -> Calibration:
    """Fit a 3 x 3 homography from N >= 4 image <-> world point pairs.

    Parameters
    ----------
    image_points_px : np.ndarray, shape (N, 2)
        Pixel coordinates.
    world_points_m : np.ndarray, shape (N, 2)
        Corresponding world coordinates (typically ground-plane x, y) in metres.

    Returns
    -------
    Calibration
        Homography-mode calibration with `homography` matrix.

    Raises
    ------
    CalibrationError
        If fewer than 4 points, fit fails (degenerate configuration), or RMS
        reprojection error exceeds the marginal threshold.
    """
    img_pts = np.asarray(image_points_px, dtype=np.float32)
    world_pts = np.asarray(world_points_m, dtype=np.float32)
    if img_pts.shape[0] != world_pts.shape[0]:
        raise CalibrationError(
            f"length mismatch: {img_pts.shape[0]} image pts vs {world_pts.shape[0]} world pts"
        )
    if img_pts.shape[0] < 4:
        raise CalibrationError(f"need >= 4 point pairs, got {img_pts.shape[0]}")

    H, _ = cv2.findHomography(img_pts, world_pts, method=0)
    if H is None:
        raise CalibrationError("homography fit failed (degenerate point configuration)")

    projected = apply_homography(img_pts, H)
    errors = np.linalg.norm(projected - world_pts, axis=1)
    rms = float(np.sqrt(np.mean(errors**2)))

    if rms >= _QUALITY_MARGINAL_RMS_M:
        raise CalibrationError(
            f"homography RMS {rms:.3f} m exceeds marginal threshold "
            f"{_QUALITY_MARGINAL_RMS_M} m"
        )

    quality: Literal["good", "marginal"] = "good" if rms < _QUALITY_GOOD_RMS_M else "marginal"
    return Calibration(
        homography=H, rms_error_m=rms, n_points=int(img_pts.shape[0]), quality=quality
    )
