"""Download a Roboflow Universe dataset for training.

Wraps the `roboflow` Python package so the call site stays declarative.
Datasets land under `data/_labelling/community/<local_name>/` with the
canonical layout: `data.yaml`, `train/`, `valid/`, `test/`.

    uv run scripts/download_dataset.py \\
        --workspace karunakar-reddy-ruymd \\
        --project medicine-balls-pwpff \\
        --version 1 \\
        --local-name medicine_ball_v1
        [--format yolov8] [--force]

Loads `ROBOFLOW_API_KEY` from `.env`. Idempotent: skips if the destination
already contains a `data.yaml`. Roboflow has a quirk where it nests the
output path inside itself; this script flattens that automatically.
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
COMMUNITY_DIR = ROOT / "data" / "_labelling" / "community"


def _flatten_nested_output(target: Path) -> None:
    """Roboflow can nest the dataset under <target>/<target>/. Flatten it."""
    nested = target / target.relative_to(target.anchor)
    if not nested.exists():
        return
    for item in nested.iterdir():
        shutil.move(str(item), str(target / item.name))
    # Remove now-empty intermediate dirs
    head = target / target.relative_to(target.anchor).parts[0]
    if head.exists():
        shutil.rmtree(head)


def download(
    workspace: str,
    project: str,
    version: int,
    local_name: str,
    model_format: str = "yolov8",
    force: bool = False,
) -> Path:
    target = COMMUNITY_DIR / local_name
    if (target / "data.yaml").exists() and not force:
        print(f"  [skip] {local_name} already present at {target.relative_to(ROOT)}")
        return target
    if force and target.exists():
        shutil.rmtree(target)

    from dotenv import load_dotenv
    from roboflow import Roboflow

    load_dotenv()
    api_key = os.getenv("ROBOFLOW_API_KEY")
    if not api_key:
        sys.exit("ROBOFLOW_API_KEY not found in .env")

    target.mkdir(parents=True, exist_ok=True)
    rf = Roboflow(api_key=api_key)
    proj = rf.workspace(workspace).project(project)
    proj.version(version).download(model_format=model_format, location=str(target))

    _flatten_nested_output(target)
    print(f"  [done] {local_name} -> {target.relative_to(ROOT)}")
    return target


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--workspace", required=True)
    p.add_argument("--project", required=True)
    p.add_argument("--version", type=int, required=True)
    p.add_argument("--local-name", required=True,
                   help="folder name under data/_labelling/community/")
    p.add_argument("--format", default="yolov8", help="export format (yolov8, yolov11, ...)")
    p.add_argument("--force", action="store_true")
    args = p.parse_args()

    download(
        workspace=args.workspace,
        project=args.project,
        version=args.version,
        local_name=args.local_name,
        model_format=args.format,
        force=args.force,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
