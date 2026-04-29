"""Video I/O — single-pass frame iteration.

Per ARCHITECTURE.md hard rule #3: every analysis reads each frame exactly
once. Use `frame_iter()` and feed the same frame to all downstream consumers
(detection, pose, tracking, annotation) inside one loop.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, NamedTuple

import cv2
import numpy as np

_FPS_MIN = 1.0
_FPS_MAX = 240.0


class VideoError(RuntimeError):
    """Raised when a video cannot be opened or has invalid metadata."""


class Frame(NamedTuple):
    idx: int
    image: np.ndarray  # BGR, HxWx3
    ts_ms: float


@dataclass(frozen=True)
class VideoInfo:
    path: Path
    fps: float
    width: int
    height: int
    frame_count: int  # OpenCV estimate; treat as approximate for some codecs


def video_info(path: Path | str) -> VideoInfo:
    """Probe metadata without iterating frames."""
    p = Path(path)
    if not p.exists():
        raise VideoError(f"video not found: {p}")
    cap = cv2.VideoCapture(str(p))
    try:
        if not cap.isOpened():
            raise VideoError(f"could not open video: {p}")
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if not _FPS_MIN <= fps <= _FPS_MAX:
            raise VideoError(
                f"FPS {fps:.3f} out of valid range "
                f"[{_FPS_MIN}, {_FPS_MAX}] for {p}"
            )
    finally:
        cap.release()
    return VideoInfo(p, fps, width, height, frame_count)


def frame_iter(path: Path | str) -> Iterator[Frame]:
    """Yield `Frame(idx, BGR_ndarray, ts_ms)` for one forward pass.

    Timestamps are derived from `idx / fps` (assumes constant frame rate).
    Releases the VideoCapture on completion, exception, or generator close.
    """
    info = video_info(path)
    cap = cv2.VideoCapture(str(info.path))
    try:
        if not cap.isOpened():
            raise VideoError(f"could not reopen video: {info.path}")
        ms_per_frame = 1000.0 / info.fps
        idx = 0
        while True:
            ok, image = cap.read()
            if not ok:
                break
            yield Frame(idx=idx, image=image, ts_ms=idx * ms_per_frame)
            idx += 1
    finally:
        cap.release()
