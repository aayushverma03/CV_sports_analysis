"""Train a custom YOLO detector from a YAML config.

Wraps Ultralytics `YOLO().train()`. Config YAML lives at
`configs/training/<detector_name>.yaml`. Training output lands at
`models/custom/<detector_name>/`; the final weights are at
`models/custom/<detector_name>/weights/best.pt`.

    uv run scripts/train_yolo.py --config configs/training/medicine_ball_v1.yaml
        [--epochs N]   # override config
        [--force]      # overwrite existing run dir
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
CUSTOM_DIR = ROOT / "models" / "custom"


def load_config(path: Path) -> dict:
    cfg = yaml.safe_load(path.read_text())
    required = {"detector_name", "dataset_yaml", "base_model"}
    missing = required - cfg.keys()
    if missing:
        sys.exit(f"config missing required keys: {sorted(missing)}")
    return cfg


def _resolve_device(spec: str) -> str | int:
    """Map 'auto' to the best available device; pass anything else through."""
    if spec != "auto":
        return spec
    import torch
    if torch.cuda.is_available():
        return 0
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def train(cfg: dict, epochs_override: int | None, force: bool) -> Path:
    detector_name = cfg["detector_name"]
    out_dir = CUSTOM_DIR / detector_name
    if out_dir.exists():
        if force:
            shutil.rmtree(out_dir)
        else:
            sys.exit(
                f"{out_dir.relative_to(ROOT)} already exists. "
                "Pass --force to overwrite."
            )
    out_dir.parent.mkdir(parents=True, exist_ok=True)

    from ultralytics import YOLO

    model = YOLO(cfg["base_model"])
    model.train(
        data=str(ROOT / cfg["dataset_yaml"]),
        epochs=epochs_override if epochs_override is not None else cfg.get("epochs", 100),
        imgsz=cfg.get("imgsz", 640),
        batch=cfg.get("batch", 16),
        patience=cfg.get("patience", 20),
        device=_resolve_device(cfg.get("device", "auto")),
        seed=cfg.get("seed", 0),
        project=str(CUSTOM_DIR),
        name=detector_name,
        exist_ok=True,
        verbose=True,
    )
    weights = out_dir / "weights" / "best.pt"
    if not weights.exists():
        sys.exit(f"training finished but {weights} not found")
    return weights


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--config", required=True, type=Path)
    p.add_argument("--epochs", type=int, default=None,
                   help="override config epochs (handy for smoke tests)")
    p.add_argument("--force", action="store_true",
                   help="overwrite existing run dir at models/custom/<detector_name>/")
    args = p.parse_args()

    cfg = load_config(args.config)
    weights = train(cfg, args.epochs, args.force)
    print(f"\ndone -> {weights.relative_to(ROOT)}")
    print(f"register in src/core/models/registry.py as `detector_{cfg['detector_name']}`")
    return 0


if __name__ == "__main__":
    sys.exit(main())
