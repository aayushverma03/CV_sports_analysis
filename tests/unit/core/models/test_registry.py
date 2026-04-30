"""Tests for the model registry."""
from __future__ import annotations

import pytest

from src.core.models import registry as reg


def test_registry_keys():
    expected = {
        "object_detector",
        "pose_default",
        "pose_biomech",
        "detector_medicine_ball_v1",
        "detector_plyo_box_v1",
        "detector_cone_v1",
    }
    assert set(reg.REGISTRY) == expected


def test_custom_detectors_have_expected_metadata():
    for key in ("detector_medicine_ball_v1", "detector_plyo_box_v1", "detector_cone_v1"):
        spec = reg.get_spec(key)
        assert spec.backend == "ultralytics"
        assert spec.weights.startswith("custom/")
        assert "classes" in spec.extras
        assert "val_mAP50" in spec.extras


def test_get_spec_returns_modelspec():
    spec = reg.get_spec("pose_biomech")
    assert spec.backend == "onnx"
    assert spec.weights == "rtmpose-x.onnx"
    assert spec.path.name == "rtmpose-x.onnx"


def test_get_spec_unknown_key():
    with pytest.raises(KeyError):
        reg.get_spec("nope")


def test_get_model_loads_and_caches():
    reg.clear_cache()
    a = reg.get_model("object_detector")
    b = reg.get_model("object_detector")
    assert a is b


def test_get_model_dispatches_by_backend():
    reg.clear_cache()
    yolo = reg.get_model("pose_default")
    onnx = reg.get_model("pose_biomech")
    assert type(yolo).__name__ == "YOLO"
    assert type(onnx).__name__ == "InferenceSession"


def test_missing_weights_raises(tmp_path, monkeypatch):
    reg.clear_cache()
    fake_spec = reg.ModelSpec(
        name="ghost", weights="ghost.pt", backend="ultralytics", version="0.0.0"
    )
    monkeypatch.setitem(reg.REGISTRY, "ghost", fake_spec)
    monkeypatch.setattr(reg, "MODELS_DIR", tmp_path)
    monkeypatch.setattr(
        reg.ModelSpec,
        "path",
        property(lambda self: tmp_path / self.weights),
    )
    with pytest.raises(reg.MissingModelError):
        reg.get_model("ghost")
