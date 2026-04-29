"""Player (athlete) detector — wraps the registry object detector for COCO `person`.

Per hard rule #4: model loaded via `src.core.models.registry.get_model`,
never hardcoded. Per hard rule #1: no test-specific logic here.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.core.models.registry import get_model, get_spec

PERSON_CLASS_ID = 0  # COCO class index for "person"


@dataclass(frozen=True)
class Detection:
    """A single bbox detection."""

    bbox_xyxy: np.ndarray  # shape (4,): x1, y1, x2, y2 in pixels
    confidence: float
    class_id: int

    @property
    def center(self) -> np.ndarray:
        x1, y1, x2, y2 = self.bbox_xyxy
        return np.array([(x1 + x2) / 2.0, (y1 + y2) / 2.0])

    @property
    def width(self) -> float:
        return float(self.bbox_xyxy[2] - self.bbox_xyxy[0])

    @property
    def height(self) -> float:
        return float(self.bbox_xyxy[3] - self.bbox_xyxy[1])


def detect_players(
    frame: np.ndarray,
    confidence: float | None = None,
    iou: float | None = None,
) -> list[Detection]:
    """Run the registry detector on `frame` and return person detections.

    Parameters
    ----------
    frame : np.ndarray
        BGR image, HxWx3, uint8.
    confidence : float, optional
        Override the registry default confidence threshold.
    iou : float, optional
        Override the registry default IoU NMS threshold. Ignored by NMS-free
        models (YOLO26).

    Returns
    -------
    list[Detection]
        Person detections, sorted by descending confidence.
    """
    spec = get_spec("object_detector")
    conf = confidence if confidence is not None else spec.extras["confidence_default"]
    iou_t = iou if iou is not None else spec.extras["iou_default"]

    model = get_model("object_detector")
    results = model.predict(
        frame, classes=[PERSON_CLASS_ID], conf=conf, iou=iou_t, verbose=False
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
