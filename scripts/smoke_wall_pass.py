"""Smoke-run the Wall Pass pipeline on a real video.

Usage: uv run scripts/smoke_wall_pass.py [video_path] [wall_distance_m]
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.tests.base import AthleteProfile  # noqa: E402
from src.tests.technical.wall_pass import WallPassTest  # noqa: E402

DEFAULT_VIDEO = (
    ROOT / "data" / "02. Technical Skills" / "Wall Pass Test"
    / "wall pass test.mp4"
)


def main() -> int:
    video = (
        Path(sys.argv[1]) if len(sys.argv) > 1 and Path(sys.argv[1]).exists()
        else DEFAULT_VIDEO
    )
    wall_distance_m = float(sys.argv[2]) if len(sys.argv) > 2 else 3.0
    if not video.exists():
        print(f"video not found: {video}")
        return 1

    out_dir = ROOT / "outputs" / "smoke_wall_pass"
    out_dir.mkdir(parents=True, exist_ok=True)
    athlete = AthleteProfile(gender="M", age=20)

    print(f"video           : {video}")
    print(f"wall_distance_m : {wall_distance_m}")
    print(f"output_dir      : {out_dir}")
    print("loading models + running...")

    test = WallPassTest(wall_distance_m=wall_distance_m)
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
            print(f"  {mid:<28} {mv.raw:.3f} {mv.unit}  [no benchmark]")
        else:
            print(
                f"  {mid:<28} {mv.raw:.3f} {mv.unit}  "
                f"score={s.score:.1f}  band={s.band}"
                f"{'  (extrapolated)' if s.extrapolated else ''}"
            )
    print()
    print(f"test_score      : {result.test_score.score:.1f} ({result.test_score.band})")
    print(f"annotated video : {result.annotated_video_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
