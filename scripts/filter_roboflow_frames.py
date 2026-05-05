"""Filter the Roboflow dataset folders down to frames that actually
contain the target markers.

Two stages per folder:
  1. Source-video allowlist — drop frames from videos that don't
     contain the target marker at all (e.g. orange cones on grass for
     green-dome, yellow disks for green-dome).
  2. HSV color check — keep only frames where the target color
     occupies a saturated region of meaningful size. Tightened against
     grass / wall / floor backgrounds.

Rejected frames go to outputs/roboflow_dataset/_rejected/{label}/ so
nothing is permanently deleted; restore false negatives manually.

Usage:
    uv run scripts/filter_roboflow_frames.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "outputs" / "roboflow_dataset"

# Source videos KNOWN to contain yellow vertical poles (slug-prefix
# match against the filename's leading source identifier).
YELLOW_POLE_SOURCE_PREFIXES = (
    "5_x_10_m_sprint",
    "zig_zag_dribbling",
    "figure_of_8",
    "hurdle_agility_run",
)

# Source videos KNOWN to contain green or red low-dome markers. Bangsbo
# (grass field, orange traffic cones) and 5x10m (yellow poles only) are
# excluded — their frames have no relevant target.
GREEN_DOME_SOURCE_PREFIXES = (
    "zig_zag_dribbling",
    "figure_of_8",
    "straight_line_dribbling_test",
)

# A target marker is a SMALL COMPACT region, not a large color field.
# We require at least one connected component whose area falls within
# (MIN_BLOB_PX, MAX_BLOB_FRAC * frame_pixels). Anything bigger is most
# likely turf / wall / floor and disqualifies the frame.
MIN_BLOB_PX = 200          # smaller than this is noise
MAX_BLOB_FRAC = 0.05       # bigger than 5% of frame is background

# HSV ranges (OpenCV: H in [0,180], S/V in [0,255]).
# Yellow pole is vivid; the disk base is the same hue.
YELLOW_LOW = (18, 130, 100)
YELLOW_HIGH = (32, 255, 255)

# Saturated dome green. H raised to 50 to clear the yellow-green
# boundary (yellow disk bases otherwise leak in).
GREEN_LOW = (50, 140, 60)
GREEN_HIGH = (80, 255, 220)

# Saturated red — two ranges for the H wraparound.
RED_LOW_1 = (0, 140, 70)
RED_HIGH_1 = (8, 255, 255)
RED_LOW_2 = (172, 140, 70)
RED_HIGH_2 = (180, 255, 255)


def _has_target_blob(
    bgr: np.ndarray,
    ranges: list[tuple[tuple[int, int, int], tuple[int, int, int]]],
    *,
    min_blob_px: int,
    max_blob_frac: float,
) -> bool:
    """True if at least one connected component of the colour mask is
    in the size range (`min_blob_px`, `max_blob_frac * frame_pixels`).
    Excludes frames where the mask is dominated by a huge background
    region (turf / wall) or by speckle noise."""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    h, w = hsv.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    for low, high in ranges:
        m = cv2.inRange(
            hsv,
            np.asarray(low, dtype=np.uint8),
            np.asarray(high, dtype=np.uint8),
        )
        mask = cv2.bitwise_or(mask, m)
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    n_labels, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    max_blob_px = int(max_blob_frac * h * w)
    # Label 0 is background; iterate the rest.
    for i in range(1, n_labels):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if min_blob_px <= area <= max_blob_px:
            return True
    return False


def _matches_prefix(filename: str, prefixes: tuple[str, ...]) -> bool:
    return any(filename.startswith(p) for p in prefixes)


def _filter_folder(
    folder: Path,
    *,
    label: str,
    source_prefixes: tuple[str, ...],
    color_ranges: list[
        tuple[tuple[int, int, int], tuple[int, int, int]]
    ],
) -> tuple[int, int, int]:
    if not folder.exists():
        print(f"  [skip] {folder} does not exist")
        return 0, 0, 0
    rejected_dir = folder.parent / "_rejected" / folder.name
    rejected_dir.mkdir(parents=True, exist_ok=True)
    files = sorted(folder.glob("*.jpg"))
    print(f"\n=== {label} ({len(files)} files) ===")
    kept = 0
    rejected_source = 0
    rejected_color = 0
    for f in files:
        if not _matches_prefix(f.name, source_prefixes):
            rejected_source += 1
            f.rename(rejected_dir / f.name)
            continue
        bgr = cv2.imread(str(f))
        if bgr is None or not _has_target_blob(
            bgr, color_ranges,
            min_blob_px=MIN_BLOB_PX, max_blob_frac=MAX_BLOB_FRAC,
        ):
            rejected_color += 1
            f.rename(rejected_dir / f.name)
            continue
        kept += 1
    print(f"  kept             : {kept}")
    print(f"  rejected (source): {rejected_source}")
    print(f"  rejected (color) : {rejected_color}")
    return kept, rejected_source, rejected_color


def main() -> int:
    if not DATASET.exists():
        print(f"dataset not found: {DATASET}")
        return 1

    yp = _filter_folder(
        DATASET / "yellow_pole",
        label="yellow_pole",
        source_prefixes=YELLOW_POLE_SOURCE_PREFIXES,
        color_ranges=[(YELLOW_LOW, YELLOW_HIGH)],
    )
    gd = _filter_folder(
        DATASET / "green_dome",
        label="green_dome (green OR red)",
        source_prefixes=GREEN_DOME_SOURCE_PREFIXES,
        color_ranges=[
            (GREEN_LOW, GREEN_HIGH),
            (RED_LOW_1, RED_HIGH_1),
            (RED_LOW_2, RED_HIGH_2),
        ],
    )
    print()
    print("summary:")
    print(f"  yellow_pole: kept {yp[0]}")
    print(f"  green_dome : kept {gd[0]}")
    print()
    print("Rejected frames are in outputs/roboflow_dataset/_rejected/.")
    print("Inspect a sample of `kept` and `_rejected` to confirm the")
    print("filter is well-tuned before uploading to Roboflow.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
