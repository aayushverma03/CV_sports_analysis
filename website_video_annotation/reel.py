"""Stitch 1-second windows from every annotated clip into a hero reel.

Output: `reel.mp4` next to the source clips. 1920x1080 @ 24 fps. Hard cuts
between clips, seamless loop (no end card).

Run:
    PYTHONPATH=. uv run website_video_annotation/reel.py
"""
from __future__ import annotations

from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent
OUT_W, OUT_H, OUT_FPS = 1920, 1080, 24
CLIP_SECS = 1.75

# (annotated file, start_seconds in that file).
# Each entry contributes CLIP_SECS of frames to the reel. Start points are
# chosen so the coaching cue is mid-hold and the action is in motion.
CLIPS: list[tuple[str, float]] = [
    ("oneone_annotated.mp4",         1.5),  # BURST — explosive opener
    ("game_annotated.mp4",           1.0),  # SCAN  — eyes up
    ("jumps_annotated.mp4",          1.0),  # EXPLODE — vertical
    ("prowess_annotated.mp4",        6.5),  # ANKLE — juggle close-up
    ("juggle_skill_annotated.mp4",   6.0),  # RHYTHM — tempo
    ("female_juggle_annotated.mp4",  7.5),  # TOUCH — soft contact
    ("annotated.mp4",                3.0),  # match — gameplay
    ("player_dribbling_annotated.mp4", 3.0),  # dribbling — finale
]


def open_writer(dst: Path) -> cv2.VideoWriter:
    writer = cv2.VideoWriter(
        str(dst), cv2.VideoWriter_fourcc(*"avc1"), OUT_FPS, (OUT_W, OUT_H)
    )
    if not writer.isOpened():
        writer = cv2.VideoWriter(
            str(dst), cv2.VideoWriter_fourcc(*"mp4v"), OUT_FPS, (OUT_W, OUT_H)
        )
        print("  (codec) avc1 unavailable; using mp4v")
    else:
        print("  (codec) avc1 / H.264")
    return writer


def append_clip(writer: cv2.VideoWriter, src: Path, start_s: float) -> int:
    """Read CLIP_SECS of frames from `src` starting at `start_s`, resample
    them to OUT_FPS, resize to canvas, and write to the reel.

    We map output frame k -> source time start_s + k / OUT_FPS, then read
    the corresponding source frame. This handles 24 / 25 / 60 fps sources
    in one path.
    """
    cap = cv2.VideoCapture(str(src))
    src_fps = cap.get(cv2.CAP_PROP_FPS) or OUT_FPS
    src_n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    n_out = int(round(CLIP_SECS * OUT_FPS))

    written = 0
    for k in range(n_out):
        t = start_s + k / OUT_FPS
        idx = min(int(round(t * src_fps)), src_n - 1)
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok:
            break
        if frame.shape[1] != OUT_W or frame.shape[0] != OUT_H:
            frame = cv2.resize(frame, (OUT_W, OUT_H), interpolation=cv2.INTER_AREA)
        writer.write(frame)
        written += 1
    cap.release()
    return written


def main() -> None:
    dst = ROOT / "reel.mp4"
    writer = open_writer(dst)
    total = 0
    for name, t in CLIPS:
        src = ROOT / name
        if not src.exists():
            print(f"  skip (missing): {name}")
            continue
        w = append_clip(writer, src, t)
        print(f"  {name} @ {t:.1f}s -> {w} frames")
        total += w
    writer.release()
    print(f"Wrote {dst}  ({total} frames, {total / OUT_FPS:.1f} s, "
          f"{dst.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
