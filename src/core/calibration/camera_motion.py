"""Frame-to-frame camera-pan compensation via optical flow.

When the operator pans the camera to follow the athlete, the athlete's
pixel-x trajectory mixes their real motion with the camera's pan. Per-
rep turn detection and per-frame speed estimates then become unreliable.

This module estimates a frame-to-frame affine transform from background
features (everything outside the athlete bbox) and accumulates it so any
point in any frame can be expressed in the FIRST frame's pixel
coordinates. Athlete bbox centers transformed this way reflect the
athlete's true ground-plane motion, modulo perspective.

Usage:
    motion = CameraMotion()
    for frame in frame_iter(video):
        athlete_bbox = detector(frame)
        motion.update(frame.image, exclude_bbox=athlete_bbox)
    # later: motion.transform(frame_idx, point_xy) -> stabilized point
"""
from __future__ import annotations

import cv2
import numpy as np


class CameraMotion:
    """Online estimator of frame-to-frame camera transform.

    Tracks ~`max_features` background corner features with Lucas-Kanade
    optical flow. Each frame's transform is composed cumulatively so a
    point in any tracked frame can be remapped to frame 0's coordinates.

    The transform is partial-affine (rotation + translation + uniform
    scale) — sufficient for pan / tilt / mild zoom on a roughly planar
    scene. Full perspective would need more anchor features and a
    homography fit, which is overkill for typical sports footage.
    """

    def __init__(
        self,
        *,
        max_features: int = 400,
        quality_level: float = 0.01,
        min_distance_px: int = 12,
        refresh_when_below: int = 100,
    ) -> None:
        self._max_features = max_features
        self._quality = quality_level
        self._min_dist = min_distance_px
        self._refresh_below = refresh_when_below

        # Per-frame cumulative 3x3 transform: stabilized = T_cum @ pixel.
        self._transforms: dict[int, np.ndarray] = {}
        self._cum = np.eye(3, dtype=np.float64)

        self._prev_gray: np.ndarray | None = None
        self._prev_pts: np.ndarray | None = None
        self._last_idx: int | None = None

    def update(
        self,
        frame_idx: int,
        bgr: np.ndarray,
        *,
        exclude_bboxes_xyxy: list[np.ndarray] | None = None,
    ) -> None:
        """Process one frame; record its cumulative transform.

        `exclude_bboxes_xyxy` is a list of (x1,y1,x2,y2) bboxes to mask
        out from feature detection — the athlete and any other moving
        subjects should be excluded so the estimator only tracks static
        background.
        """
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        mask = np.full((h, w), 255, dtype=np.uint8)
        if exclude_bboxes_xyxy:
            for box in exclude_bboxes_xyxy:
                x1, y1, x2, y2 = (int(v) for v in box)
                cv2.rectangle(mask, (x1, y1), (x2, y2), 0, -1)

        if self._prev_gray is None:
            # First frame: identity transform, seed feature points.
            self._transforms[frame_idx] = self._cum.copy()
            self._prev_gray = gray
            self._prev_pts = self._detect(gray, mask)
            self._last_idx = frame_idx
            return

        if self._prev_pts is None or len(self._prev_pts) < 6:
            # Not enough points to track; reuse previous transform.
            self._transforms[frame_idx] = self._cum.copy()
            self._prev_gray = gray
            self._prev_pts = self._detect(gray, mask)
            self._last_idx = frame_idx
            return

        # Track previous points into the current frame.
        next_pts, status, _ = cv2.calcOpticalFlowPyrLK(
            self._prev_gray, gray, self._prev_pts, None,
            winSize=(21, 21), maxLevel=3,
        )
        ok = status.flatten() == 1
        good_old = self._prev_pts[ok]
        good_new = next_pts[ok] if next_pts is not None else None

        delta = np.eye(3, dtype=np.float64)
        if good_new is not None and len(good_old) >= 6:
            tf2x3, _ = cv2.estimateAffinePartial2D(
                good_new, good_old, method=cv2.RANSAC,
                ransacReprojThreshold=3.0,
            )
            if tf2x3 is not None:
                delta[:2, :] = tf2x3
        # delta maps current-frame pixels to previous-frame pixels;
        # cumulative composition expresses current-frame -> frame 0.
        self._cum = self._cum @ delta
        self._transforms[frame_idx] = self._cum.copy()

        # Maintain feature pool.
        if good_new is None or len(good_new) < self._refresh_below:
            self._prev_pts = self._detect(gray, mask)
        else:
            self._prev_pts = good_new.reshape(-1, 1, 2).astype(np.float32)
        self._prev_gray = gray
        self._last_idx = frame_idx

    def _detect(
        self, gray: np.ndarray, mask: np.ndarray
    ) -> np.ndarray | None:
        return cv2.goodFeaturesToTrack(
            gray,
            maxCorners=self._max_features,
            qualityLevel=self._quality,
            minDistance=self._min_dist,
            mask=mask,
        )

    def transform_point(
        self, frame_idx: int, point_xy: tuple[float, float]
    ) -> tuple[float, float]:
        """Map (x, y) from `frame_idx`'s pixel coords into frame 0's
        coords. Frames not seen by `update` map to the most recent
        known transform (or identity)."""
        T = self._transforms.get(frame_idx)
        if T is None:
            T = (
                self._cum if self._last_idx is not None
                else np.eye(3, dtype=np.float64)
            )
        v = np.array([point_xy[0], point_xy[1], 1.0], dtype=np.float64)
        out = T @ v
        if out[2] == 0:
            return float(point_xy[0]), float(point_xy[1])
        return float(out[0] / out[2]), float(out[1] / out[2])

    def transform_points(
        self,
        frame_idx_to_xy: dict[int, tuple[float, float]],
    ) -> dict[int, tuple[float, float]]:
        return {
            fi: self.transform_point(fi, xy)
            for fi, xy in frame_idx_to_xy.items()
        }
