"""Marker detectors — open-vocabulary (YOLO-World) and custom-trained.

Used by sprint and agility tests for cone / pole calibration. Per hard
rule #4 models are loaded via the registry; per hard rule #1 no test-
specific constants live here — pipelines pass their own model keys
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


class CustomMarkerDetector:
    """Run one or more own-trained YOLO marker detectors per frame and
    merge their detections.

    Wraps custom models registered under `detector_<name>_v1`. Each
    sub-detector contributes its own bboxes; class_id in the merged
    output is the index into the supplied `model_keys` list (so
    callers can recover which class fired).

    This is the production replacement for `MarkerDetector` (YOLO-
    World) once we have per-class trained models. Single-class trained
    detectors are far more accurate and faster than text-prompted
    open-vocabulary detection.

    Parameters
    ----------
    model_keys : list[str]
        Registry keys of the marker detectors to run (e.g.
        `["detector_yellow_pole_v1", "detector_green_dome_v1"]`).
    confidence : float | None
        Override the per-model `confidence_default`. None = each model
        uses its own registry default.
    iou : float | None
        Override the per-model `iou_default`.
    """

    def __init__(
        self,
        model_keys: list[str],
        confidence: float | None = None,
        iou: float | None = None,
    ) -> None:
        if not model_keys:
            raise ValueError("model_keys must contain at least one entry")
        self._model_keys = list(model_keys)
        self._models = []
        self._confs: list[float] = []
        self._ious: list[float] = []
        self._class_names: list[str] = []
        for key in self._model_keys:
            spec = get_spec(key)
            self._models.append(get_model(key))
            self._confs.append(
                confidence if confidence is not None
                else spec.extras.get("confidence_default", 0.30)
            )
            self._ious.append(
                iou if iou is not None
                else spec.extras.get("iou_default", 0.45)
            )
            classes = spec.extras.get("classes") or [spec.name]
            self._class_names.append(classes[0])

    @property
    def class_names(self) -> list[str]:
        return list(self._class_names)

    def detect(self, frame: np.ndarray) -> list[Detection]:
        """Run every sub-detector and return the merged set, sorted by
        confidence. The returned `Detection.class_id` is the index into
        `self.class_names` (i.e. which sub-detector fired)."""
        out: list[Detection] = []
        for ci, (model, conf, iou) in enumerate(
            zip(self._models, self._confs, self._ious, strict=True),
        ):
            results = model.predict(
                frame, conf=conf, iou=iou, verbose=False,
            )
            for r in results:
                if r.boxes is None:
                    continue
                for box in r.boxes:
                    out.append(Detection(
                        bbox_xyxy=box.xyxy[0].cpu().numpy(),
                        confidence=float(box.conf[0]),
                        class_id=ci,
                    ))
        out.sort(key=lambda d: d.confidence, reverse=True)
        return out

    def class_name(self, class_id: int) -> str:
        return self._class_names[class_id]
