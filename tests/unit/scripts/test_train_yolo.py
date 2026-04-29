"""Tests for scripts/train_yolo.py."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts"))

from train_yolo import load_config  # noqa: E402


def test_load_config_returns_dict(tmp_path):
    p = tmp_path / "ok.yaml"
    p.write_text(yaml.safe_dump({
        "detector_name": "x",
        "dataset_yaml": "data/x/data.yaml",
        "base_model": "yolo26n.pt",
        "epochs": 50,
    }))
    cfg = load_config(p)
    assert cfg["detector_name"] == "x"
    assert cfg["epochs"] == 50


def test_load_config_missing_required_key_exits(tmp_path):
    p = tmp_path / "bad.yaml"
    p.write_text(yaml.safe_dump({"detector_name": "x"}))  # missing dataset_yaml + base_model
    with pytest.raises(SystemExit):
        load_config(p)
