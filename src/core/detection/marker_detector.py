"""Open-vocabulary marker detector — wraps YOLO-World with class prompts.

Used by sprint and agility tests for cone / pole calibration. Per hard
rule #4 the model is loaded via the registry; per hard rule #1 no
test-specific constants live here — pipelines pass their own prompts
when the registry default isn't right.
"""
from __future__ import annotations

import numpy as np

from src.core.detection.player_detector import Detection
from src.core.models.registry import get_model, get_spec


class MarkerDetector:
    """Detects calibration markers (cones, poles) via text prompts.

    Parameters
    ----------
    prompts : list[str], optional
        Class prompts. Defaults to the registry's `default_classes`.
    confidence : float, optional
        Confidence threshold. Defaults to the registry's `confidence_default`.
    iou : float, optional
        IoU NMS threshold. Defaults to the registry's `iou_default`.
    model_key : str
        Registry key for the open-vocab detector.
    """

    def __init__(
        self,
        prompts: list[str] | None = None,
        confidence: float | None = None,
        iou: float | None = None,
        model_key: str = "detector_open_vocab_v1",
    ) -> None:
        spec = get_spec(model_key)
        self._model = get_model(model_key)
        self._prompts = list(prompts) if prompts else list(spec.extras["default_classes"])
        # Re-set classes if the caller overrode them — get_model caches the
        # model instance, so a fresh set_classes is required per pipeline.
        self._model.set_classes(self._prompts)
        self._conf = confidence if confidence is not None else spec.extras["confidence_default"]
        self._iou = iou if iou is not None else spec.extras["iou_default"]

    @property
    def prompts(self) -> list[str]:
        return list(self._prompts)

    def detect(self, frame: np.ndarray) -> list[Detection]:
        """Run inference on `frame`. Returns detections sorted by confidence (desc)."""
        results = self._model.predict(
            frame, conf=self._conf, iou=self._iou, verbose=False
        )
        detections: list[Detection] = []
        for r in results:
            if r.boxes is None:
                continue
            for box in r.boxes:
                detections.append(
                    Detection(
                        bbox_xyxy=box.xyxy[0].cpu().numpy(),
                        confidence=float(box.conf[0]),
                        class_id=int(box.cls[0]),
                    )
                )
        detections.sort(key=lambda d: d.confidence, reverse=True)
        return detections

    def class_name(self, class_id: int) -> str:
        """Human-readable name for a detection's `class_id` (= prompt index)."""
        return self._prompts[class_id]
