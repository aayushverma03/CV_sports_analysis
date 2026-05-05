"""Extract video frames for Roboflow training of two cone classes.

Two output folders:
  outputs/roboflow_dataset/yellow_pole/ — frames from videos that
    typically contain yellow vertical poles with disk bases.
  outputs/roboflow_dataset/green_dome/  — frames from videos that
    typically contain green low-dome markers.

Frames are extracted every `STRIDE` frames (defaults to 10 — about
3 fps from a 30 fps source). Filenames embed the source video so you
can trace any image back to its origin.

Some videos appear in both folders because the courses use both
markers; in Roboflow you'll label each class individually so the
double-counting is fine. Frames where the target marker is not
visible can be discarded or used as negative examples.

Usage:
    uv run scripts/extract_roboflow_frames.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]

STRIDE = 10
OUT_DIR = ROOT / "outputs" / "roboflow_dataset"

# Videos likely to contain YELLOW VERTICAL POLES (with yellow or green
# disk base). Add / remove as you find more candidates.
YELLOW_POLE_VIDEOS: list[Path] = [
    ROOT / "data" / "01. Physical Capabilities" / "5 x 10 m Sprint with COD"
    / "VID_20251118_133941531.mp4",
    ROOT / "data" / "02. Technical Skills" / "Zig-Zag Dribbling Test"
    / "VID_20251118_132027985.mp4",
    ROOT / "data" / "02. Technical Skills" / "Zig-Zag Dribbling Test"
    / "VID_20251118_132104351 (2).mp4",
    ROOT / "data" / "02. Technical Skills" / "Figure of 8" / "figure of 8.mp4",
    ROOT / "data" / "01. Physical Capabilities" / "Hurdle-Agility-Run"
    / "Hurdle agility test.mp4",
    ROOT / "data" / "01. Physical Capabilities" / "Repeated Sprint Ability"
    / "VID_20251118_135143903 (1).mp4",
]

# Videos likely to contain GREEN LOW-DOME markers (start / finish gates,
# slalom bases, route markers).
GREEN_DOME_VIDEOS: list[Path] = [
    ROOT / "data" / "02. Technical Skills" / "Zig-Zag Dribbling Test"
    / "VID_20251118_132027985.mp4",
    ROOT / "data" / "02. Technical Skills" / "Zig-Zag Dribbling Test"
    / "VID_20251118_132104351 (2).mp4",
    ROOT / "data" / "02. Technical Skills" / "Figure of 8" / "figure of 8.mp4",
    ROOT / "data" / "02. Technical Skills" / "Straight line dribbling test"
    / "straight line dribble.mp4",
    ROOT / "data" / "01. Physical Capabilities" / "5 x 10 m Sprint with COD"
    / "VID_20251118_133941531.mp4",
    ROOT / "data" / "01. Physical Capabilities" / "Linear Sprint 10, 20, 30, 40 m"
    / "VID_20251118_134552822 (1).mp4",
    ROOT / "data" / "01. Physical Capabilities" / "Bangsbo Sprint Test"
    / "bangsbo sprint test.mp4",
]


def _slug(path: Path) -> str:
    """Filesystem-safe identifier: <test-folder>_<video-stem>, lowercase,
    spaces and special chars to underscores."""
    test_folder = path.parent.name
    stem = path.stem
    raw = f"{test_folder}_{stem}".lower()
    safe = "".join(
        c if c.isalnum() else "_" for c in raw
    )
    while "__" in safe:
        safe = safe.replace("__", "_")
    return safe.strip("_")


def _extract_one(video_path: Path, out_dir: Path, stride: int) -> int:
    if not video_path.exists():
        print(f"  [skip] not found: {video_path}")
        return 0
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"  [skip] could not open: {video_path}")
        return 0
    slug = _slug(video_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    written = 0
    fi = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if fi % stride == 0:
            name = f"{slug}_f{fi:05d}.jpg"
            cv2.imwrite(str(out_dir / name), frame)
            written += 1
        fi += 1
    cap.release()
    return written


def _extract_set(label: str, videos: list[Path]) -> int:
    out_dir = OUT_DIR / label
    print(f"\n=== {label} ===")
    print(f"output: {out_dir}")
    total = 0
    for v in videos:
        n = _extract_one(v, out_dir, STRIDE)
        if n:
            print(f"  [{n:3d} frames] {v.relative_to(ROOT)}")
        total += n
    print(f"total {label}: {total} frames")
    return total


def main() -> int:
    print(f"stride     : every {STRIDE}th frame (~{30 // STRIDE} fps from 30 fps)")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    n_yellow = _extract_set("yellow_pole", YELLOW_POLE_VIDEOS)
    n_green = _extract_set("green_dome", GREEN_DOME_VIDEOS)
    print()
    print(f"done. {n_yellow} yellow_pole frames, {n_green} green_dome frames")
    print(f"output root: {OUT_DIR}")
    print()
    print("Next steps:")
    print(f"  1. Inspect {OUT_DIR}/yellow_pole/ and {OUT_DIR}/green_dome/")
    print("  2. Delete frames that don't contain the target marker")
    print("  3. Upload each folder to Roboflow as a separate dataset")
    print("  4. Label bboxes for the target class only in each dataset")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
