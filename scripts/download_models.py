"""Download model weights to `models/` per docs/models/MODEL_REGISTRY.md.

Iterates `src.core.models.registry.REGISTRY`. Run from the repo root:

    uv run scripts/download_models.py [--force]

Existing weights are skipped unless --force is passed.
"""
from __future__ import annotations

import argparse
import io
import os
import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.core.models.registry import MODELS_DIR, REGISTRY, ModelSpec  # noqa: E402


def download_ultralytics(spec: ModelSpec, force: bool) -> None:
    target = spec.path
    if target.exists() and not force:
        print(f"  [skip] {spec.weights} already present")
        return
    if force and target.exists():
        target.unlink()
    print(f"  [pull] {spec.weights} via ultralytics")
    from ultralytics import YOLO

    cwd = Path.cwd()
    os.chdir(MODELS_DIR)
    try:
        YOLO(spec.weights)
    finally:
        os.chdir(cwd)


def download_onnx(spec: ModelSpec, force: bool) -> None:
    target = spec.path
    if target.exists() and not force:
        print(f"  [skip] {spec.weights} already present")
        return
    url = spec.extras.get("download_url")
    if not url:
        sys.exit(f"no download_url in registry for {spec.name}")
    print(f"  [pull] {spec.weights} from {url}")
    with urllib.request.urlopen(url) as resp:
        data = resp.read()
    if url.endswith(".zip"):
        with zipfile.ZipFile(io.BytesIO(data)) as zf:
            members = [n for n in zf.namelist() if n.endswith(".onnx")]
            if not members:
                sys.exit(f"no .onnx file inside {url}")
            with zf.open(members[0]) as src, target.open("wb") as dst:
                shutil.copyfileobj(src, dst)
    else:
        target.write_bytes(data)
    print(f"  [done] {target.relative_to(ROOT)}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--force", action="store_true", help="Re-download even if present")
    args = parser.parse_args()

    MODELS_DIR.mkdir(exist_ok=True)
    for key, spec in REGISTRY.items():
        if spec.backend == "ultralytics":
            download_ultralytics(spec, args.force)
        elif spec.backend == "onnx":
            download_onnx(spec, args.force)
        else:
            sys.exit(f"unknown backend {spec.backend!r} for {key}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
