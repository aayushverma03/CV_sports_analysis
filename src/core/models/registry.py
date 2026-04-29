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
