"""End-to-end Phase 0 smoke test.

Wires up: video read -> tracker -> pose -> annotation -> MP4 writer.
No calibration / cones (Phase 0.5 work). No metrics. Just exercises every
Phase 0 primitive against a real clip.

    uv run scripts/smoke_test.py --video <PATH> --output <OUT.mp4>
        [--pose-backend pose_default|pose_biomech]
        [--max-frames N]
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.core.annotation.overlays import draw_bbox, draw_hud, draw_skeleton  # noqa: E402
from src.core.pose.estimator import create_pose_estimator  # noqa: E402
from src.core.tracking.bytetrack_tracker import ByteTrackTracker  # noqa: E402
from src.core.utils.video_io import frame_iter, video_info  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--video", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--pose-backend", default="pose_default",
                   choices=["pose_default", "pose_biomech"])
    p.add_argument("--max-frames", type=int, default=None)
    args = p.parse_args()

    info = video_info(args.video)
    print(f"input: {args.video.name} {info.width}x{info.height} "
          f"@ {info.fps:.2f} fps, {info.frame_count} frames")

    tracker = ByteTrackTracker()
    pose = create_pose_estimator(args.pose_backend)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(args.output), fourcc, info.fps,
                             (info.width, info.height))

    n_frames = 0
    n_with_athlete = 0
    n_with_pose = 0
    track_ids: set[int] = set()

    t0 = time.perf_counter()
    for f in frame_iter(args.video):
        if args.max_frames is not None and f.idx >= args.max_frames:
            break
        n_frames += 1

        detections = tracker.update(f.image)
        if detections:
            n_with_athlete += 1

        for det in detections:
            track_ids.add(det.track_id)
            draw_bbox(f.image, det.bbox_xyxy, label=f"id={det.track_id}")
            pose_det = pose.estimate_bbox(f.image, det.bbox_xyxy)
            if pose_det is not None and pose_det.mean_confidence > 0.0:
                n_with_pose += 1
                draw_skeleton(f.image, pose_det.keypoints)

        draw_hud(f.image, {
            "frame": f.idx,
            "ts": f"{f.ts_ms / 1000:.2f}s",
            "tracks": len(detections),
        })
        writer.write(f.image)

    writer.release()
    elapsed = time.perf_counter() - t0

    print(f"output: {args.output} ({n_frames} frames written)")
    print(f"player detected on {n_with_athlete}/{n_frames} frames")
    print(f"pose extracted on  {n_with_pose}/{n_frames} frames")
    print(f"unique track ids   {sorted(track_ids)}")
    print(f"throughput         {n_frames / elapsed:.1f} fps "
          f"({elapsed * 1000 / max(n_frames, 1):.1f} ms/frame)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
