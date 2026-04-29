"""Tests for video_io."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from src.core.utils.video_io import Frame, VideoError, frame_iter, video_info


def _make_video(path: Path, n_frames: int = 10, fps: float = 30.0,
                size: tuple[int, int] = (64, 64)) -> None:
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, fps, size)
    for i in range(n_frames):
        frame = np.full((size[1], size[0], 3), i * 25 % 255, dtype=np.uint8)
        writer.write(frame)
    writer.release()


def test_video_info(tmp_path):
    p = tmp_path / "clip.mp4"
    _make_video(p, n_frames=10, fps=30.0, size=(64, 64))
    info = video_info(p)
    assert info.path == p
    assert info.width == 64
    assert info.height == 64
    assert info.fps == pytest.approx(30.0, abs=0.5)
    assert info.frame_count == 10


def test_frame_iter_yields_correct_count_and_timestamps(tmp_path):
    p = tmp_path / "clip.mp4"
    _make_video(p, n_frames=10, fps=30.0)
    frames = list(frame_iter(p))
    assert len(frames) == 10
    for i, f in enumerate(frames):
        assert isinstance(f, Frame)
        assert f.idx == i
        assert f.ts_ms == pytest.approx(i * 1000.0 / 30.0, abs=0.1)
        assert f.image.shape == (64, 64, 3)
        assert f.image.dtype == np.uint8


def test_missing_file_raises(tmp_path):
    with pytest.raises(VideoError, match="not found"):
        video_info(tmp_path / "nope.mp4")
    with pytest.raises(VideoError, match="not found"):
        list(frame_iter(tmp_path / "nope.mp4"))


def test_unopenable_file_raises(tmp_path):
    bad = tmp_path / "garbage.mp4"
    bad.write_bytes(b"not a video")
    with pytest.raises(VideoError, match="could not open"):
        video_info(bad)


def test_generator_releases_on_break(tmp_path):
    p = tmp_path / "clip.mp4"
    _make_video(p, n_frames=20, fps=30.0)
    gen = frame_iter(p)
    next(gen)
    next(gen)
    gen.close()  # should not raise; finally releases cap
