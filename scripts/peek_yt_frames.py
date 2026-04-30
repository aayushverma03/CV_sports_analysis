"""Quick frame peek + YOLO-World run on the YouTube sprint video."""
from __future__ import annotations

import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ultralytics import YOLOWorld  # noqa: E402

VIDEO = Path("/tmp/yt_sprint/5tZQLGfWlTE.mp4")
OUT = ROOT / "outputs" / "yt_sprint_peek"
OUT.mkdir(parents=True, exist_ok=True)


def main() -> int:
    cap = cv2.VideoCapture(str(VIDEO))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    targets = [int(total * f) for f in (0.05, 0.15, 0.30, 0.50, 0.70, 0.90)]
    print(f"sampling frames {targets} of {total}")

    model = YOLOWorld("models/yolov8x-worldv2.pt")
    prompts = [
        "yellow stick",
        "white stick",
        "vertical pole",
        "marker pole",
        "orange traffic cone",
        "training cone",
        "yellow slalom pole",
    ]
    model.set_classes(prompts)

    for idx in targets:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok:
            continue
        results = model.predict(frame, conf=0.10, verbose=False)
        r = results[0]
        n = 0 if r.boxes is None else len(r.boxes)
        classes = []
        if n:
            classes = [(prompts[int(c)], float(s))
                       for c, s in zip(r.boxes.cls.tolist(), r.boxes.conf.tolist())]
        print(f"  frame {idx:5d}: {n} hits {classes}")
        cv2.imwrite(str(OUT / f"frame_{idx:05d}.jpg"), r.plot())

    cap.release()
    print(f"\nframes saved: {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
