"""Extract frames at a target fps from data/ for Roboflow labelling.

Walks data/ recursively (skipping data/_labelling/), samples each video at
`--fps`, writes JPGs into data/_labelling/extracted_frames/<source path>/.
Idempotent: a video is skipped if its output dir already contains JPGs;
pass --force to re-extract.

    uv run scripts/extract_frames.py [--fps 1.0] [--force]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.core.utils.video_io import frame_iter, video_info  # noqa: E402

DATA_ROOT = ROOT / "data"
LABELLING_ROOT = DATA_ROOT / "_labelling"
EXTRACTED = LABELLING_ROOT / "extracted_frames"
VIDEO_EXTS = {".mp4", ".mov", ".m4v", ".avi"}


def find_videos(root: Path) -> list[Path]:
    """Return sorted list of video files under `root`, excluding `_labelling/`."""
    labelling = root / "_labelling"
    videos = [
        p for p in root.rglob("*")
        if p.is_file()
        and p.suffix.lower() in VIDEO_EXTS
        and labelling not in p.parents
        and p != labelling
    ]
    return sorted(videos)


def output_dir_for(video: Path) -> Path:
    """data/foo/bar/clip.mp4 -> data/_labelling/extracted_frames/foo/bar/clip/"""
    relative = video.relative_to(DATA_ROOT)
    return EXTRACTED / relative.parent / relative.stem


def extract_video(
    video: Path, out_dir: Path, fps_target: float, force: bool = False
) -> tuple[int, str]:
    """Extract frames at fps_target into out_dir. Returns (n_written, status)."""
    if out_dir.exists() and not force:
        existing = list(out_dir.glob("*.jpg"))
        if existing:
            return (len(existing), "skip")
    out_dir.mkdir(parents=True, exist_ok=True)
    info = video_info(video)
    stride = max(1, round(info.fps / fps_target))
    n = 0
    for f in frame_iter(video):
        if f.idx % stride != 0:
            continue
        cv2.imwrite(str(out_dir / f"{f.idx:06d}.jpg"), f.image)
        n += 1
    return (n, "wrote")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--fps", type=float, default=1.0, help="target sample rate")
    p.add_argument("--force", action="store_true", help="re-extract even if output exists")
    args = p.parse_args()

    videos = find_videos(DATA_ROOT)
    print(f"found {len(videos)} videos under {DATA_ROOT.relative_to(ROOT)}")

    total_written = 0
    for v in videos:
        out_dir = output_dir_for(v)
        n, status = extract_video(v, out_dir, args.fps, args.force)
        print(f"  [{status:5}] {v.relative_to(ROOT)} -> {n} frames")
        if status == "wrote":
            total_written += n

    print(f"total: {total_written} new frames written")
    return 0


if __name__ == "__main__":
    sys.exit(main())
