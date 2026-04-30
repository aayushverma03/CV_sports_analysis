"""Smoke-run the Linear Sprint pipeline on a real video.

Usage: uv run scripts/smoke_linear_sprint.py [video_path] [distance_m]
       defaults: landscape Linear Sprint video, 10m distance
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.tests.base import AthleteProfile  # noqa: E402
from src.tests.physical.linear_sprint import LinearSprintTest  # noqa: E402

DEFAULT_VIDEO = (
    ROOT / "data" / "01. Physical Capabilities"
    / "Linear Sprint 10, 20, 30, 40 m" / "VID_20251118_134552822 (1).mp4"
)


def main() -> int:
    video = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_VIDEO
    distance_m = float(sys.argv[2]) if len(sys.argv) > 2 else 10.0
    if not video.exists():
        print(f"video not found: {video}")
        return 1

    out_dir = ROOT / "outputs" / "smoke_linear_sprint"
    out_dir.mkdir(parents=True, exist_ok=True)
    athlete = AthleteProfile(gender="M", age=20)

    print(f"video      : {video}")
    print(f"distance   : {distance_m:.0f} m")
    print(f"output_dir : {out_dir}")
    print("loading models + running...")

    test = LinearSprintTest(distance_m=distance_m)
    t0 = time.perf_counter()
    result = test.run(video, athlete, out_dir)
    dt = time.perf_counter() - t0

    print(f"\n--- result ({dt:.1f}s) ---")
    print(f"test_id         : {result.test_id}")
    print(f"duration_s      : {result.diagnostics.duration_s:.2f}")
    print(f"fps_input       : {result.diagnostics.fps_input:.2f}")
    print()
    for mid, mv in result.metrics.items():
        s = result.scores.get(mid)
        if s is None:
            print(f"  {mid:<20} {mv.raw:.3f} {mv.unit}  [no benchmark]")
        else:
            print(
                f"  {mid:<20} {mv.raw:.3f} {mv.unit}  "
                f"score={s.score:.1f}  band={s.band}"
                f"{'  (extrapolated)' if s.extrapolated else ''}"
            )
    print()
    print(
        f"test_score      : {result.test_score.score:.1f} ({result.test_score.band})"
    )
    print(f"annotated video : {result.annotated_video_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
