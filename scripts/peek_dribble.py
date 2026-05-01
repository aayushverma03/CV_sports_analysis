"""Quick peek at the dribbling test video to understand framing."""
from __future__ import annotations

import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
VIDEO = (
    ROOT / "data" / "02. Technical Skills" / "Straight line dribbling test"
    / "straight line dribble.mp4"
)
OUT = ROOT / "outputs" / "dribble_peek"
OUT.mkdir(parents=True, exist_ok=True)


def main() -> int:
    cap = cv2.VideoCapture(str(VIDEO))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    targets = [int(total * f) for f in (0.05, 0.20, 0.40, 0.60, 0.85)]
    for idx in targets:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if ok:
            cv2.imwrite(str(OUT / f"frame_{idx:05d}.jpg"), frame)
            print(f"wrote frame {idx}")
    cap.release()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
