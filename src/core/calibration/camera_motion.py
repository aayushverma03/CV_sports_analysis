"""Frame-to-frame camera-pan compensation via optical flow + ORB anchoring.

When the operator pans the camera to follow the athlete, the athlete's
pixel-x trajectory mixes their real motion with the camera's pan. Per-
rep turn detection and per-frame speed estimates then become unreliable.

This module estimates a per-frame affine transform that maps any later
frame's pixel coordinates back into FRAME 0's pixel space. Two
estimators run together:

- Frame 0 anchor (ORB descriptor matching). Drift-free: every frame is
  matched directly against frame 0's keypoints, so accumulated error
  cannot grow with video length. Used whenever there are enough good
  matches.
- Lucas-Kanade chain (frame N-1 → N). Cheap and resilient when the
  scene drifts off frame 0's view (lighting / occlusion / camera moves
  far). Used as fallback when ORB matching is weak.

Athlete bbox centers and cone detections, transformed by these per-
frame matrices, reflect the athlete's true ground-plane motion, modulo
perspective.

Usage:
    motion = CameraMotion()
    for frame in frame_iter(video):
        athlete_bbox = detector(frame)
        motion.update(frame.idx, frame.image, exclude_bboxes_xyxy=[athlete_bbox])
    # later: motion.transform_point(frame_idx, point_xy) -> stabilized point
"""
from __future__ import annotations

import cv2
import numpy as np


class CameraMotion:
    """Online estimator of frame-to-frame camera transform.

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
        anchor_features: int = 1500,
        anchor_match_min: int = 30,
    ) -> None:
        self._max_features = max_features
        self._quality = quality_level
        self._min_dist = min_distance_px
        self._refresh_below = refresh_when_below
        self._anchor_match_min = anchor_match_min

        # Per-frame transform: stabilized = T @ pixel.
        self._transforms: dict[int, np.ndarray] = {}
        # Cumulative LK fallback transform.
        self._cum_lk = np.eye(3, dtype=np.float64)

        self._prev_gray: np.ndarray | None = None
        self._prev_pts: np.ndarray | None = None
        self._last_idx: int | None = None

        # Frame-0 ORB anchor — drift-free reference.
        self._orb = cv2.ORB_create(nfeatures=anchor_features)
        self._matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        self._anchor_kp: tuple[cv2.KeyPoint, ...] | None = None
        self._anchor_desc: np.ndarray | None = None
        # Telemetry: per-frame source of the transform, for diagnostics.
        self._source: dict[int, str] = {}

    def update(
        self,
        frame_idx: int,
        bgr: np.ndarray,
        *,
        exclude_bboxes_xyxy: list[np.ndarray] | None = None,
    ) -> None:
        """Process one frame; record its transform back to frame 0."""
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        mask = np.full((h, w), 255, dtype=np.uint8)
        if exclude_bboxes_xyxy:
            for box in exclude_bboxes_xyxy:
                x1, y1, x2, y2 = (int(v) for v in box)
                cv2.rectangle(mask, (x1, y1), (x2, y2), 0, -1)

        if self._prev_gray is None:
            # First frame: seed both LK features and the ORB anchor.
            self._transforms[frame_idx] = np.eye(3, dtype=np.float64)
            self._source[frame_idx] = "anchor"
            self._prev_gray = gray
            self._prev_pts = self._detect(gray, mask)
            kp, desc = self._orb.detectAndCompute(gray, mask)
            if desc is not None and len(kp) >= self._anchor_match_min:
                self._anchor_kp = tuple(kp)
                self._anchor_desc = desc
            self._last_idx = frame_idx
            return

        # 1. Try ORB match against frame 0 first (drift-free).
        T_anchor = self._try_anchor(gray, mask)
        if T_anchor is not None:
            self._transforms[frame_idx] = T_anchor
            self._source[frame_idx] = "anchor"
            self._cum_lk = T_anchor.copy()  # resync LK chain to anchor
            # Refresh LK features for next frame's fallback path.
            self._prev_pts = self._detect(gray, mask)
            self._prev_gray = gray
            self._last_idx = frame_idx
            return

        # 2. Fallback: LK chain from previous frame.
        if self._prev_pts is None or len(self._prev_pts) < 6:
            self._transforms[frame_idx] = self._cum_lk.copy()
            self._source[frame_idx] = "carry"
            self._prev_gray = gray
            self._prev_pts = self._detect(gray, mask)
            self._last_idx = frame_idx
            return

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
        self._cum_lk = self._cum_lk @ delta
        self._transforms[frame_idx] = self._cum_lk.copy()
        self._source[frame_idx] = "lk"

        if good_new is None or len(good_new) < self._refresh_below:
            self._prev_pts = self._detect(gray, mask)
        else:
            self._prev_pts = good_new.reshape(-1, 1, 2).astype(np.float32)
        self._prev_gray = gray
        self._last_idx = frame_idx

    def _try_anchor(
        self, gray: np.ndarray, mask: np.ndarray
    ) -> np.ndarray | None:
        """Direct affine fit against frame-0 ORB descriptors.

        Returns a 3x3 transform mapping current-frame pixels to frame-0
        pixels, or None if matching is too weak.
        """
        if self._anchor_desc is None or self._anchor_kp is None:
            return None
        kp, desc = self._orb.detectAndCompute(gray, mask)
        if desc is None or len(kp) < self._anchor_match_min:
            return None
        try:
            matches = self._matcher.match(desc, self._anchor_desc)
        except cv2.error:
            return None
        if len(matches) < self._anchor_match_min:
            return None
        # Top matches by Hamming distance.
        matches = sorted(matches, key=lambda m: m.distance)[:300]
        src = np.array(
            [kp[m.queryIdx].pt for m in matches], dtype=np.float32
        )
        dst = np.array(
            [self._anchor_kp[m.trainIdx].pt for m in matches], dtype=np.float32
        )
        if len(src) < self._anchor_match_min:
            return None
        tf2x3, inliers = cv2.estimateAffinePartial2D(
            src, dst, method=cv2.RANSAC, ransacReprojThreshold=3.0,
        )
        if tf2x3 is None or inliers is None:
            return None
        if int(inliers.sum()) < self._anchor_match_min:
            return None
        T = np.eye(3, dtype=np.float64)
        T[:2, :] = tf2x3
        return T

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
                self._cum_lk if self._last_idx is not None
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

    def source_summary(self) -> dict[str, int]:
        """Count of frames per transform source: anchor / lk / carry."""
        out: dict[str, int] = {}
        for s in self._source.values():
            out[s] = out.get(s, 0) + 1
        return out
