"""Pluggable pose estimator — factory by registry key.

Two backends with a uniform top-down API:

- `pose_default`  -> Ultralytics YOLO26-pose (run on a person crop).
- `pose_biomech`  -> RTMPose-x ONNX (preprocess crop, ONNX inference, SIMCC decode).

Family base classes pick the backend (`pose_default` for sprint/agility/etc.,
`pose_biomech` for jump/throw biomech). Metric code never branches on backend.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import cv2
import numpy as np

from src.core.models.registry import get_model, get_spec

COCO_KEYPOINT_NAMES = [
    "nose",
    "left_eye", "right_eye",
    "left_ear", "right_ear",
    "left_shoulder", "right_shoulder",
    "left_elbow", "right_elbow",
    "left_wrist", "right_wrist",
    "left_hip", "right_hip",
    "left_knee", "right_knee",
    "left_ankle", "right_ankle",
]
COCO_KEYPOINT_INDEX = {name: i for i, name in enumerate(COCO_KEYPOINT_NAMES)}


@dataclass(frozen=True)
class PoseDetection:
    """A single COCO 17-keypoint pose."""

    keypoints: np.ndarray  # (17, 3): (x_px, y_px, confidence)
    bbox_xyxy: np.ndarray  # source bbox in original frame coordinates

    def position(self, name: str) -> np.ndarray:
        """Return (x, y) of a named keypoint, in pixels."""
        return self.keypoints[COCO_KEYPOINT_INDEX[name], :2]

    def confidence_of(self, name: str) -> float:
        """Confidence of a named keypoint, range [0, 1]."""
        return float(self.keypoints[COCO_KEYPOINT_INDEX[name], 2])

    @property
    def mean_confidence(self) -> float:
        return float(self.keypoints[:, 2].mean())


class PoseEstimator(Protocol):
    """Top-down pose-estimation interface."""

    def estimate_bbox(
        self, frame: np.ndarray, bbox_xyxy: np.ndarray
    ) -> PoseDetection | None: ...


def create_pose_estimator(model_key: str) -> PoseEstimator:
    """Factory: return a pose estimator for the given registry key."""
    spec = get_spec(model_key)
    if spec.backend == "ultralytics":
        return _YoloPoseEstimator(model_key)
    if spec.backend == "onnx":
        return _RTMPoseEstimator(model_key)
    raise ValueError(f"unsupported pose backend: {spec.backend!r}")


# --- YOLO-pose (Ultralytics) --------------------------------------------


class _YoloPoseEstimator:
    """Run YOLO26-pose on a person crop, map keypoints back to frame coords."""

    def __init__(self, model_key: str) -> None:
        self._model = get_model(model_key)
        spec = get_spec(model_key)
        self._conf = spec.extras["confidence_default"]

    def estimate_bbox(
        self, frame: np.ndarray, bbox_xyxy: np.ndarray
    ) -> PoseDetection | None:
        x1, y1, x2, y2 = _expand_bbox(bbox_xyxy, frame.shape, pad=0.10)
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return None
        results = self._model.predict(crop, conf=self._conf, verbose=False)
        for r in results:
            if r.keypoints is None:
                continue
            kp_data = r.keypoints.data.cpu().numpy()  # (n, 17, 3)
            if len(kp_data) == 0:
                continue
            best = int(np.argmax(kp_data[:, :, 2].mean(axis=1)))
            kp = kp_data[best].copy()
            kp[:, 0] += x1
            kp[:, 1] += y1
            return PoseDetection(keypoints=kp, bbox_xyxy=np.asarray(bbox_xyxy, dtype=float))
        return None


# --- RTMPose (ONNX) ------------------------------------------------------

_RTMPOSE_MEAN = np.array([123.675, 116.28, 103.53], dtype=np.float32)
_RTMPOSE_STD = np.array([58.395, 57.12, 57.375], dtype=np.float32)


class _RTMPoseEstimator:
    """RTMPose-x ONNX inference with affine crop and SIMCC decoder."""

    def __init__(self, model_key: str) -> None:
        self._session = get_model(model_key)
        spec = get_spec(model_key)
        self._input_w, self._input_h = spec.extras["input_size"]  # (W, H) = (288, 384)
        self._simcc_split = spec.extras["simcc_split"]
        self._input_name = self._session.get_inputs()[0].name

    def estimate_bbox(
        self, frame: np.ndarray, bbox_xyxy: np.ndarray
    ) -> PoseDetection | None:
        forward, inverse = _bbox_affine(
            bbox_xyxy, output_size=(self._input_w, self._input_h), pad=1.25
        )
        warped = cv2.warpAffine(
            frame, forward, (self._input_w, self._input_h), flags=cv2.INTER_LINEAR
        )
        if warped.size == 0:
            return None

        # BGR -> RGB, normalize, NCHW float32
        rgb = cv2.cvtColor(warped, cv2.COLOR_BGR2RGB).astype(np.float32)
        rgb = (rgb - _RTMPOSE_MEAN) / _RTMPOSE_STD
        x = rgb.transpose(2, 0, 1)[None, ...]  # (1, 3, H, W)

        simcc_x, simcc_y = self._session.run(None, {self._input_name: x})
        kp_input, conf = _decode_simcc(simcc_x, simcc_y, split=self._simcc_split)

        # Map keypoints from input-image coords back to original frame
        kp_input_h = np.column_stack([kp_input[0], np.ones(len(kp_input[0]))])
        mapped = kp_input_h @ inverse.T  # (17, 2)
        keypoints = np.column_stack([mapped, conf[0]])

        return PoseDetection(keypoints=keypoints, bbox_xyxy=np.asarray(bbox_xyxy, dtype=float))


# --- helpers -------------------------------------------------------------


def _expand_bbox(
    bbox_xyxy: np.ndarray, frame_shape: tuple[int, int, int], pad: float
) -> tuple[int, int, int, int]:
    """Expand bbox by `pad` (fraction) and clamp to frame, returning ints."""
    x1, y1, x2, y2 = bbox_xyxy
    w = x2 - x1
    h = y2 - y1
    fh, fw = frame_shape[:2]
    return (
        max(0, int(x1 - w * pad)),
        max(0, int(y1 - h * pad)),
        min(fw, int(x2 + w * pad)),
        min(fh, int(y2 + h * pad)),
    )


def _bbox_affine(
    bbox_xyxy: np.ndarray, output_size: tuple[int, int], pad: float
) -> tuple[np.ndarray, np.ndarray]:
    """Affine transform mapping a padded bbox (aspect-corrected) to output_size.

    Returns (forward 2x3, inverse 2x3) suitable for cv2.warpAffine and inverse mapping.
    """
    x1, y1, x2, y2 = bbox_xyxy
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    bw = (x2 - x1) * pad
    bh = (y2 - y1) * pad

    out_w, out_h = output_size
    target_aspect = out_w / out_h
    if bw / bh > target_aspect:
        bh = bw / target_aspect
    else:
        bw = bh * target_aspect

    src = np.array(
        [
            [cx - bw / 2, cy - bh / 2],
            [cx + bw / 2, cy - bh / 2],
            [cx - bw / 2, cy + bh / 2],
        ],
        dtype=np.float32,
    )
    dst = np.array([[0, 0], [out_w, 0], [0, out_h]], dtype=np.float32)
    forward = cv2.getAffineTransform(src, dst)
    inverse = cv2.getAffineTransform(dst, src)
    return forward, inverse


def _decode_simcc(
    simcc_x: np.ndarray, simcc_y: np.ndarray, split: float
) -> tuple[np.ndarray, np.ndarray]:
    """Decode SIMCC heatmaps to (keypoints_xy, confidence).

    Parameters
    ----------
    simcc_x : np.ndarray, shape (B, K, Wx)
    simcc_y : np.ndarray, shape (B, K, Hy)
    split : float
        SIMCC split factor (RTMPose default is 2.0).

    Returns
    -------
    keypoints : np.ndarray, shape (B, K, 2)
        (x, y) in input-image pixel coordinates.
    confidence : np.ndarray, shape (B, K)
        Per-keypoint confidence in [0, 1] (min of x/y peak values).
    """
    x_idx = simcc_x.argmax(axis=-1).astype(np.float32) / split
    y_idx = simcc_y.argmax(axis=-1).astype(np.float32) / split
    conf_x = simcc_x.max(axis=-1)
    conf_y = simcc_y.max(axis=-1)
    conf = np.minimum(conf_x, conf_y)
    keypoints = np.stack([x_idx, y_idx], axis=-1)
    return keypoints, conf
