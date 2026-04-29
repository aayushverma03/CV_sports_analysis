"""Tests for scripts/extract_frames.py core functions."""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts"))

from extract_frames import extract_video, find_videos  # noqa: E402


def _make_video(path: Path, n_frames: int = 30, fps: float = 30.0,
                size: tuple[int, int] = (64, 64)) -> None:
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, fps, size)
    for i in range(n_frames):
        writer.write(np.full((size[1], size[0], 3), i % 255, dtype=np.uint8))
    writer.release()


def test_extract_video_at_1_fps(tmp_path):
    video = tmp_path / "clip.mp4"
    _make_video(video, n_frames=30, fps=30.0)
    out_dir = tmp_path / "out"
    n, status = extract_video(video, out_dir, fps_target=1.0)
    assert status == "wrote"
    # 30 frames at 30 fps with stride round(30/1)=30 -> only frame 0 qualifies
    assert n == 1


def test_extract_video_at_2_fps(tmp_path):
    video = tmp_path / "clip.mp4"
    _make_video(video, n_frames=60, fps=30.0)
    out_dir = tmp_path / "out"
    n, _ = extract_video(video, out_dir, fps_target=2.0)
    # stride = round(30/2) = 15; frames 0, 15, 30, 45 -> 4 frames
    assert n == 4
    assert sorted(p.name for p in out_dir.glob("*.jpg")) == [
        "000000.jpg", "000015.jpg", "000030.jpg", "000045.jpg",
    ]


def test_idempotent_skip(tmp_path):
    video = tmp_path / "clip.mp4"
    _make_video(video, n_frames=30, fps=30.0)
    out_dir = tmp_path / "out"
    extract_video(video, out_dir, fps_target=1.0)
    n, status = extract_video(video, out_dir, fps_target=1.0)
    assert status == "skip"
    assert n >= 1


def test_force_reextracts(tmp_path):
    video = tmp_path / "clip.mp4"
    _make_video(video, n_frames=30, fps=30.0)
    out_dir = tmp_path / "out"
    extract_video(video, out_dir, fps_target=1.0)
    n, status = extract_video(video, out_dir, fps_target=1.0, force=True)
    assert status == "wrote"
    assert n >= 1


def test_find_videos_excludes_labelling(tmp_path):
    (tmp_path / "_labelling").mkdir()
    (tmp_path / "_labelling" / "ignore.mp4").touch()
    (tmp_path / "physical").mkdir()
    (tmp_path / "physical" / "keep.mp4").touch()
    (tmp_path / "physical" / "deeper").mkdir()
    (tmp_path / "physical" / "deeper" / "alsokeep.mov").touch()
    (tmp_path / "physical" / "notvideo.txt").touch()

    found = find_videos(tmp_path)
    names = [p.name for p in found]
    assert "keep.mp4" in names
    assert "alsokeep.mov" in names
    assert "ignore.mp4" not in names
    assert "notvideo.txt" not in names
