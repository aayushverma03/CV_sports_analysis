"""Tests for the pose estimator factory + decoders."""
from __future__ import annotations

import numpy as np
import pytest

from src.core.pose.estimator import (
    COCO_KEYPOINT_INDEX,
    COCO_KEYPOINT_NAMES,
    PoseDetection,
    _bbox_affine,
    _decode_simcc,
    create_pose_estimator,
)


# --- COCO indexing -------------------------------------------------------


def test_coco_keypoint_count():
    assert len(COCO_KEYPOINT_NAMES) == 17
    assert len(COCO_KEYPOINT_INDEX) == 17


def test_coco_index_round_trip():
    for i, name in enumerate(COCO_KEYPOINT_NAMES):
        assert COCO_KEYPOINT_INDEX[name] == i


# --- PoseDetection -------------------------------------------------------


def _fake_pose() -> PoseDetection:
    kp = np.zeros((17, 3), dtype=float)
    for i in range(17):
        kp[i] = [float(i), float(i * 2), 0.5 + i * 0.01]
    return PoseDetection(keypoints=kp, bbox_xyxy=np.array([0.0, 0.0, 100.0, 200.0]))


def test_position_returns_xy():
    p = _fake_pose()
    assert np.allclose(p.position("nose"), [0, 0])
    assert np.allclose(p.position("left_knee"), [13, 26])


def test_confidence_of():
    p = _fake_pose()
    assert p.confidence_of("nose") == pytest.approx(0.5)
    assert p.confidence_of("right_ankle") == pytest.approx(0.66)


def test_mean_confidence():
    p = _fake_pose()
    expected = float(np.mean([0.5 + i * 0.01 for i in range(17)]))
    assert p.mean_confidence == pytest.approx(expected)


# --- SIMCC decoder -------------------------------------------------------


def test_decode_simcc_argmax_picks_peak():
    # Build a pair of synthetic SIMCC distributions with known peak indices.
    B, K, Wx, Hy = 1, 17, 576, 768
    simcc_x = np.zeros((B, K, Wx), dtype=np.float32)
    simcc_y = np.zeros((B, K, Hy), dtype=np.float32)
    for k in range(K):
        simcc_x[0, k, k * 10] = 1.0   # peak at x=k*10 (in bins)
        simcc_y[0, k, k * 20] = 0.8   # peak at y=k*20 (in bins)

    kp, conf = _decode_simcc(simcc_x, simcc_y, split=2.0)
    assert kp.shape == (1, 17, 2)
    assert conf.shape == (1, 17)
    # x = bin / split = (k * 10) / 2.0 = k * 5
    # y = (k * 20) / 2.0 = k * 10
    for k in range(K):
        assert kp[0, k, 0] == pytest.approx(k * 5.0)
        assert kp[0, k, 1] == pytest.approx(k * 10.0)
    # conf = min(x_peak, y_peak) = min(1.0, 0.8) = 0.8
    assert np.allclose(conf[0], 0.8)


# --- bbox affine ---------------------------------------------------------


def test_bbox_affine_round_trip():
    bbox = np.array([100.0, 200.0, 300.0, 600.0])  # 200x400 in original
    forward, inverse = _bbox_affine(bbox, output_size=(288, 384), pad=1.0)
    # Center of bbox should map to center of output
    center_orig = np.array([200.0, 400.0, 1.0])
    mapped = forward @ center_orig
    assert mapped[0] == pytest.approx(144.0, abs=1.0)
    assert mapped[1] == pytest.approx(192.0, abs=1.0)
    # Inverse maps output center back
    out_center = np.array([144.0, 192.0, 1.0])
    back = inverse @ out_center
    assert back[0] == pytest.approx(200.0, abs=1.0)
    assert back[1] == pytest.approx(400.0, abs=1.0)


# --- factory dispatch ----------------------------------------------------


def test_factory_returns_yolo_for_pose_default():
    est = create_pose_estimator("pose_default")
    assert type(est).__name__ == "_YoloPoseEstimator"


def test_factory_returns_rtmpose_for_pose_biomech():
    est = create_pose_estimator("pose_biomech")
    assert type(est).__name__ == "_RTMPoseEstimator"


def test_factory_unknown_key_raises():
    with pytest.raises(KeyError):
        create_pose_estimator("does_not_exist")


# --- end-to-end smoke (real models) --------------------------------------


def test_yolo_pose_returns_none_or_detection_on_blank():
    est = create_pose_estimator("pose_default")
    frame = np.full((480, 640, 3), 128, dtype=np.uint8)
    bbox = np.array([100.0, 100.0, 300.0, 400.0])
    out = est.estimate_bbox(frame, bbox)
    assert out is None or isinstance(out, PoseDetection)


def test_rtmpose_returns_detection_with_correct_shape():
    """RTMPose always returns a pose for any bbox (no detection threshold).
    Shape and bbox preservation are what we verify here."""
    est = create_pose_estimator("pose_biomech")
    frame = np.full((480, 640, 3), 128, dtype=np.uint8)
    bbox = np.array([100.0, 100.0, 300.0, 400.0])
    out = est.estimate_bbox(frame, bbox)
    assert isinstance(out, PoseDetection)
    assert out.keypoints.shape == (17, 3)
    assert np.allclose(out.bbox_xyxy, bbox)
