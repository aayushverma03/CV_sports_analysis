"""ByteTrack wrapper — multi-frame stable IDs over the registry detector.

Stateful: one instance per pipeline run. Internally drives Ultralytics'
`model.track(persist=...)`; the first `update()` call resets any tracker
state left over from a prior instance, then subsequent calls persist.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.core.detection.player_detector import PERSON_CLASS_ID, Detection
from src.core.models.registry import get_model, get_spec


@dataclass(frozen=True)
class TrackedDetection(Detection):
    """A detection with a stable ID across frames."""

    track_id: int


class ByteTrackTracker:
    """Stateful tracker built on Ultralytics + ByteTrack.

    Parameters
    ----------
    classes : list[int], optional
        COCO class IDs to track. Defaults to `[PERSON_CLASS_ID]`.
    confidence : float, optional
        Override the registry default confidence threshold.
    iou : float, optional
        Override the registry default IoU threshold.
    """

    def __init__(
        self,
        classes: list[int] | None = None,
        confidence: float | None = None,
        iou: float | None = None,
    ) -> None:
        spec = get_spec("object_detector")
        self._model = get_model("object_detector")
        self._classes = classes if classes is not None else [PERSON_CLASS_ID]
        self._conf = confidence if confidence is not None else spec.extras["confidence_default"]
        self._iou = iou if iou is not None else spec.extras["iou_default"]
        self._initialized = False

    def update(self, frame: np.ndarray) -> list[TrackedDetection]:
        """Run detection + tracking on one frame.

        First call resets the tracker (persist=False); subsequent calls
        carry track state forward (persist=True). Returns only detections
        with an assigned track ID.
        """
        results = self._model.track(
            frame,
            persist=self._initialized,
            tracker="bytetrack.yaml",
            classes=self._classes,
            conf=self._conf,
            iou=self._iou,
            verbose=False,
        )
        self._initialized = True

        out: list[TrackedDetection] = []
        for r in results:
            if r.boxes is None or r.boxes.id is None:
                continue
            for box in r.boxes:
                out.append(
                    TrackedDetection(
                        bbox_xyxy=box.xyxy[0].cpu().numpy(),
                        confidence=float(box.conf[0]),
                        class_id=int(box.cls[0]),
                        track_id=int(box.id[0]),
                    )
                )
        out.sort(key=lambda d: d.confidence, reverse=True)
        return out
