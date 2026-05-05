"""Unit tests for Zig-Zag Dribbling pipeline-specific helpers."""
from __future__ import annotations

from src.tests.technical.zig_zag_dribbling import (
    _calibrate,
    _count_cone_passages,
)


# --- _calibrate -------------------------------------------------------


def test_calibrate_uses_cone_pair_when_available():
    """5 cones at 100 px spacing, 2 m apart -> 50 px/m."""
    cones = [(100.0 + i * 100.0, 400.0) for i in range(5)]
    px_per_m, source = _calibrate(
        cones=cones, cone_spacing_m=2.0,
        baseline_bbox_h_px=200.0, assumed_height_m=1.7,
    )
    assert source == "cone-pair"
    assert abs(px_per_m - 50.0) < 1e-6


def test_calibrate_falls_back_to_body_height_when_one_cone():
    """Single cone — fall back to body-height proxy."""
    cones = [(500.0, 400.0)]
    px_per_m, source = _calibrate(
        cones=cones, cone_spacing_m=2.0,
        baseline_bbox_h_px=200.0, assumed_height_m=1.7,
    )
    assert source == "body-height-proxy"
    assert abs(px_per_m - (200.0 / 1.7)) < 1e-6


def test_calibrate_uses_median_gap_robust_to_outlier():
    """4 cones at 100 px spacing, plus 1 stray cluster — median picks
    the real spacing."""
    cones = [
        (100.0, 400.0), (200.0, 400.0), (300.0, 400.0), (400.0, 400.0),
        (1500.0, 400.0),  # stray
    ]
    px_per_m, source = _calibrate(
        cones=cones, cone_spacing_m=2.0,
        baseline_bbox_h_px=200.0, assumed_height_m=1.7,
    )
    assert source == "cone-pair"
    assert abs(px_per_m - 50.0) < 1e-6


# --- _count_cone_passages ---------------------------------------------


def test_count_cone_passages_full_slalom():
    """Athlete ankle passes within proximity of every cone."""
    cones = [(i * 100.0, 400.0) for i in range(5)]
    # Ankle trajectory threads through all 5 cones.
    traj = []
    for i, (cx, cy) in enumerate(cones):
        traj.append((i * 10, cx, cy + 30.0))   # within proximity
    n = _count_cone_passages(
        ankle_traj=traj, cones=cones, baseline_bbox_h_px=200.0,
    )
    assert n == 5


def test_count_cone_passages_partial_when_missing_cone():
    """Athlete misses the last cone — only 4 counted."""
    cones = [(i * 100.0, 400.0) for i in range(5)]
    traj = [(i * 10, cx, cy + 30.0) for i, (cx, cy) in enumerate(cones[:4])]
    n = _count_cone_passages(
        ankle_traj=traj, cones=cones, baseline_bbox_h_px=200.0,
    )
    assert n == 4


def test_count_cone_passages_zero_with_empty_inputs():
    assert _count_cone_passages(
        ankle_traj=[], cones=[(0.0, 0.0)], baseline_bbox_h_px=200.0,
    ) == 0
    assert _count_cone_passages(
        ankle_traj=[(0, 0.0, 0.0)], cones=[], baseline_bbox_h_px=200.0,
    ) == 0
