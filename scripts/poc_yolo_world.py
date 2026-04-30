"""Quick POC: does YOLO-World detect the markers in cone-dependent test videos?

Loads `yolov8x-worldv2.pt` (Ultralytics ships it), sets a list of class
prompts, runs inference on a handful of frames sampled from each given
video, and writes annotated PNGs to outputs/poc_yolo_world/ so we can
eyeball whether disks / poles / cones get bounding boxes.

Usage:
    uv run scripts/poc_yolo_world.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ultralytics import YOLOWorld  # noqa: E402

OUT_DIR = ROOT / "outputs" / "poc_yolo_world"

# 6 prompts covering the 4 marker types we know are in your videos
PROMPTS = [
    "orange traffic cone",
    "red flat marker disc",
    "green flat marker disc",
    "yellow slalom pole",
    "training cone",
    "field marker",
]

# Test name -> (video path, # frames to sample)
SAMPLES: list[tuple[str, str, int]] = [
    ("linear-sprint-portrait",
     "data/01. Physical Capabilities/Linear Sprint 10, 20, 30, 40 m/10 m sprint.mp4", 5),
    ("linear-sprint-landscape",
     "data/01. Physical Capabilities/Linear Sprint 10, 20, 30, 40 m/VID_20251118_134552822 (1).mp4", 5),
    ("illinois",
     "data/01. Physical Capabilities/Illinois Agility Test", 5),
    ("zigzag",
     "data/02. Technical Skills/Zig-Zag Dribbling Test", 5),
    ("ttest",
     "data/01. Physical Capabilities/T-Test", 5),
]


def _first_video(folder_or_file: Path) -> Path | None:
    p = Path(folder_or_file)
    if p.is_file():
        return p
    if not p.exists():
        return None
    vids = sorted(p.glob("*.mp4")) + sorted(p.glob("*.MP4"))
    return vids[0] if vids else None


def _sample_frames(video_path: Path, n_samples: int) -> list[tuple[int, "cv2.Mat"]]:
    cap = cv2.VideoCapture(str(video_path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        return []
    # Pick `n_samples` evenly spaced frames, skipping the very first/last
    if n_samples == 1:
        targets = [total // 2]
    else:
        step = total / (n_samples + 1)
        targets = [int(step * (i + 1)) for i in range(n_samples)]
    out: list[tuple[int, "cv2.Mat"]] = []
    for idx in targets:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if ok:
            out.append((idx, frame))
    cap.release()
    return out


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"loading YOLOWorld weights...")
    t0 = time.perf_counter()
    model = YOLOWorld("yolov8x-worldv2.pt")
    model.set_classes(PROMPTS)
    print(f"  loaded in {time.perf_counter() - t0:.1f}s")
    print(f"prompts: {PROMPTS}")
    print()

    summary: list[tuple[str, int, int, int]] = []  # (label, frames_run, hits_total, hits_per_frame_max)

    for label, rel, n_samples in SAMPLES:
        path = ROOT / rel
        video = _first_video(path)
        if video is None:
            print(f"[{label}] no video at {rel}, skipping")
            continue
        print(f"[{label}] {video.name}")
        frames = _sample_frames(video, n_samples)
        if not frames:
            print(f"  no frames extracted")
            continue

        per_frame_hits = []
        for idx, frame in frames:
            results = model.predict(frame, conf=0.10, verbose=False)
            r = results[0]
            n_hits = 0 if r.boxes is None else len(r.boxes)
            per_frame_hits.append(n_hits)

            annotated = r.plot()
            out_file = OUT_DIR / f"{label}_frame_{idx:05d}.jpg"
            cv2.imwrite(str(out_file), annotated)
            classes = []
            if r.boxes is not None and len(r.boxes) > 0:
                classes = [(PROMPTS[int(c)], float(s))
                           for c, s in zip(r.boxes.cls.tolist(), r.boxes.conf.tolist())]
            print(f"  frame {idx:5d}: {n_hits} hits  {classes}")

        total = sum(per_frame_hits)
        peak = max(per_frame_hits) if per_frame_hits else 0
        summary.append((label, len(frames), total, peak))

    print()
    print("=" * 60)
    print(f"{'test':<26} {'frames':>7} {'hits':>6} {'peak':>6}")
    print("-" * 60)
    for label, n_frames, total_hits, peak_hits in summary:
        print(f"{label:<26} {n_frames:>7} {total_hits:>6} {peak_hits:>6}")
    print()
    print(f"annotated frames: {OUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
