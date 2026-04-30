"""Model registry — single source of truth for inference models.

See docs/models/MODEL_REGISTRY.md. Every model loader in src/ goes through
`get_model()`; do not hardcode weights paths elsewhere (hard rule #4).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

ROOT = Path(__file__).resolve().parents[3]
MODELS_DIR = ROOT / "models"

Backend = Literal["ultralytics", "onnx"]


@dataclass(frozen=True)
class ModelSpec:
    name: str
    weights: str
    backend: Backend
    version: str
    extras: dict[str, Any] = field(default_factory=dict)

    @property
    def path(self) -> Path:
        return MODELS_DIR / self.weights


REGISTRY: dict[str, ModelSpec] = {
    "object_detector": ModelSpec(
        name="yolo26m",
        weights="yolo26m.pt",
        backend="ultralytics",
        version="26.0.0",
        extras={"task": "detect", "confidence_default": 0.35, "iou_default": 0.45},
    ),
    "pose_default": ModelSpec(
        name="yolo26m-pose",
        weights="yolo26m-pose.pt",
        backend="ultralytics",
        version="26.0.0",
        extras={"task": "pose", "keypoint_count": 17, "confidence_default": 0.30},
    ),
    "pose_biomech": ModelSpec(
        name="rtmpose-x",
        weights="rtmpose-x.onnx",
        backend="onnx",
        version="1.0.0",
        extras={
            "task": "pose",
            "keypoint_count": 17,
            "input_size": (288, 384),
            "simcc_split": 2.0,
            "confidence_default": 0.30,
            "download_url": (
                "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/"
                "onnx_sdk/rtmpose-x_simcc-body7_pt-body7_700e-384x288-"
                "71d7b7e9_20230629.zip"
            ),
        },
    ),
    "detector_medicine_ball_v1": ModelSpec(
        name="medicine_ball_v1",
        weights="custom/medicine_ball_v1.pt",
        backend="ultralytics",
        version="1.0.0",
        extras={
            "task": "detect",
            "classes": ["ball"],
            "confidence_default": 0.35,
            "iou_default": 0.45,
            "trained_on": "data/_labelling/community/medicine_ball_v1/data.yaml",
            "val_mAP50": 0.811,
        },
    ),
    "detector_plyo_box_v1": ModelSpec(
        name="plyo_box_v1",
        weights="custom/plyo_box_v1.pt",
        backend="ultralytics",
        version="1.0.0",
        extras={
            "task": "detect",
            "classes": ["plyo_box"],
            # Smoke-tested: 1/10 frames at conf=0.25, 10/10 at conf=0.10.
            # Model sees the box but is shy on it. Lowering threshold; combined
            # with NMS this is acceptable for the well-bounded box class.
            "confidence_default": 0.10,
            "iou_default": 0.45,
            "trained_on": "data/_labelling/community/plyo_box_v1/data.yaml",
            "val_mAP50": 0.859,
            "notes": "Trained on 30 images. Detection signal exists but low confidence on out-of-distribution frames. Extend dataset (~150-200 instances) for production v1.",
        },
    ),
    "detector_cone_v1": ModelSpec(
        name="cone_v1",
        weights="custom/cone_v1.pt",
        backend="ultralytics",
        version="1.0.0",
        extras={
            "task": "detect",
            "classes": ["cone"],
            "confidence_default": 0.35,
            "iou_default": 0.45,
            "trained_on": "data/_labelling/community/cone_v1/data.yaml",
            "val_mAP50": 0.994,
            # Stopped early at ~epoch 67 of 100; mAP50 plateaued at 0.99 by
            # epoch ~30. Trained only on traffic cones. User's videos also
            # contain flat marker disks (red/green) and yellow slalom poles
            # which are not represented in this training set. Smoke test on
            # real videos: T-Test (traffic cones) 5/5 frames detected, mean
            # conf 0.49; Linear Sprint (flat disks) 0/5; Illinois (mixed) 0/5;
            # Zig-Zag (poles) 1/5. Extend with own-labelled disk + pole
            # frames for v2 to cover those classes under the same `cone`
            # label.
            "notes": "Traffic cones only. Tests using flat marker disks (Linear Sprint, Illinois) or slalom poles (Zig-Zag) will need v2 with extended dataset.",
        },
    ),
    # detector_hurdle_v1 deregistered: the only v1 test that consumed it
    # (45-Second Agility Hurdle Jump) was deferred. Weights stay on disk at
    # models/custom/hurdle_v1.pt; dataset stays at
    # data/_labelling/community/hurdle_v1/. Re-add this entry in v1.1 once
    # the test reactivates with a fresh 200-400 instance dataset.
}


class MissingModelError(FileNotFoundError):
    """Weights not on disk — run `uv run scripts/download_models.py`."""


def get_spec(key: str) -> ModelSpec:
    """Return the registry entry without loading the model."""
    if key not in REGISTRY:
        raise KeyError(f"Unknown model key: {key!r}. Known: {sorted(REGISTRY)}")
    return REGISTRY[key]


@lru_cache(maxsize=None)
def get_model(key: str) -> Any:
    """Lazy-load and cache the model by registry key.

    Dispatches to the matching backend loader. Subsequent calls return the
    cached instance.
    """
    spec = get_spec(key)
    if not spec.path.exists():
        raise MissingModelError(
            f"{spec.weights} not found at {spec.path}. "
            "Run: uv run scripts/download_models.py"
        )
    if spec.backend == "ultralytics":
        from ultralytics import YOLO

        return YOLO(str(spec.path))
    if spec.backend == "onnx":
        import onnxruntime as ort

        providers = ["CoreMLExecutionProvider", "CPUExecutionProvider"]
        return ort.InferenceSession(str(spec.path), providers=providers)
    raise ValueError(f"Unknown backend {spec.backend!r} for key {key!r}")


def clear_cache() -> None:
    """Drop cached model instances (useful for tests / device switches)."""
    get_model.cache_clear()
