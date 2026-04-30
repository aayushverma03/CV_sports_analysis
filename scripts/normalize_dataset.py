"""Normalize a YOLO-format dataset before training.

Three operations:

- `--keep-class N` (repeatable) — strip all bboxes whose class id is not in
  the keep set. Re-index remaining classes to 0..k-1. Delete images that end
  up with zero labels (they used to contain only filtered-out classes).
- `--split` — auto-split a train-only dataset into train/valid/test. Default
  ratio 0.7 / 0.2 / 0.1, seed 0. No-op if valid/ already populated.
- `--names` — comma-separated final class names; rewritten into data.yaml.

Usage:

    uv run scripts/normalize_dataset.py --dir data/_labelling/community/cone_v1 \\
        --keep-class 0 --names cone

    uv run scripts/normalize_dataset.py --dir data/_labelling/community/hurdle_v1 \\
        --split --names hurdle

    uv run scripts/normalize_dataset.py --dir data/_labelling/community/plyo_box_v1 \\
        --split --names plyo_box
"""
from __future__ import annotations

import argparse
import random
import shutil
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent


def strip_classes(dataset_dir: Path, keep: set[int]) -> tuple[int, int]:
    """Drop bboxes whose class id is not in `keep`, re-index 0..k-1.

    Returns (kept_files, deleted_files). Walks every train/valid/test split.
    """
    keep_sorted = sorted(keep)
    remap = {old: new for new, old in enumerate(keep_sorted)}

    kept_files = 0
    deleted_files = 0
    for split in ("train", "valid", "test"):
        labels_dir = dataset_dir / split / "labels"
        images_dir = dataset_dir / split / "images"
        if not labels_dir.exists():
            continue
        for label_path in labels_dir.glob("*.txt"):
            new_lines: list[str] = []
            for line in label_path.read_text().splitlines():
                line = line.strip()
                if not line:
                    continue
                class_id = int(line.split()[0])
                if class_id not in remap:
                    continue
                rest = line.split(" ", 1)[1]
                new_lines.append(f"{remap[class_id]} {rest}")
            if new_lines:
                label_path.write_text("\n".join(new_lines) + "\n")
                kept_files += 1
            else:
                # No surviving labels — delete image+label (no negative-example training)
                label_path.unlink()
                stem = label_path.stem
                for ext in (".jpg", ".jpeg", ".png"):
                    img = images_dir / f"{stem}{ext}"
                    if img.exists():
                        img.unlink()
                        break
                deleted_files += 1
    return kept_files, deleted_files


def auto_split(
    dataset_dir: Path,
    ratios: tuple[float, float, float] = (0.7, 0.2, 0.1),
    seed: int = 0,
) -> dict[str, int]:
    """Split a train-only dataset into train/valid/test.

    No-op if valid/images already has >0 files (assumed already split).
    Returns counts per split.
    """
    train_imgs = dataset_dir / "train" / "images"
    train_lbls = dataset_dir / "train" / "labels"
    valid_imgs = dataset_dir / "valid" / "images"
    if valid_imgs.exists() and any(valid_imgs.iterdir()):
        return {"already_split": 0}

    items: list[tuple[Path, Path]] = []
    for img in sorted(train_imgs.glob("*")):
        if img.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue
        lbl = train_lbls / f"{img.stem}.txt"
        if not lbl.exists():
            continue
        items.append((img, lbl))

    rng = random.Random(seed)
    rng.shuffle(items)
    n = len(items)
    n_train = int(n * ratios[0])
    n_valid = int(n * ratios[1])
    splits = {
        "train": items[:n_train],
        "valid": items[n_train : n_train + n_valid],
        "test": items[n_train + n_valid :],
    }

    counts: dict[str, int] = {}
    for split, pairs in splits.items():
        out_imgs = dataset_dir / split / "images"
        out_lbls = dataset_dir / split / "labels"
        out_imgs.mkdir(parents=True, exist_ok=True)
        out_lbls.mkdir(parents=True, exist_ok=True)
        for img, lbl in pairs:
            if split != "train":
                shutil.move(str(img), str(out_imgs / img.name))
                shutil.move(str(lbl), str(out_lbls / lbl.name))
        counts[split] = len(pairs)
    return counts


def update_data_yaml(dataset_dir: Path, names: list[str]) -> None:
    yaml_path = dataset_dir / "data.yaml"
    cfg = yaml.safe_load(yaml_path.read_text())
    cfg["names"] = list(names)
    cfg["nc"] = len(names)
    cfg.setdefault("train", "../train/images")
    cfg["val"] = "../valid/images"
    cfg["test"] = "../test/images"
    yaml_path.write_text(yaml.safe_dump(cfg, sort_keys=False))


def summarize(dataset_dir: Path) -> str:
    parts = []
    for split in ("train", "valid", "test"):
        d = dataset_dir / split / "images"
        if d.exists():
            parts.append(f"{split}={len(list(d.glob('*')))}")
    return " ".join(parts)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--dir", required=True, type=Path)
    p.add_argument("--keep-class", type=int, action="append", default=[],
                   help="class id to keep (repeatable); others stripped")
    p.add_argument("--split", action="store_true",
                   help="auto-split train-only into train/valid/test")
    p.add_argument("--names", type=str,
                   help="comma-separated final class names")
    args = p.parse_args()

    args.dir = args.dir.resolve()
    if not args.dir.exists():
        sys.exit(f"dataset dir not found: {args.dir}")

    if args.keep_class:
        keep = set(args.keep_class)
        kept, deleted = strip_classes(args.dir, keep)
        print(f"  [strip] kept {kept} files, deleted {deleted} (no surviving labels)")

    if args.split:
        counts = auto_split(args.dir)
        print(f"  [split] {counts}")

    if args.names:
        names = [n.strip() for n in args.names.split(",")]
        update_data_yaml(args.dir, names)
        print(f"  [names] -> {names}")

    print(f"  [done] {args.dir.relative_to(ROOT)}: {summarize(args.dir)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
