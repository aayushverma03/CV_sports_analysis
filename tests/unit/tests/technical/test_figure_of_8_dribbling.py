"""Unit tests for Figure of 8 Dribbling pipeline-specific helpers."""
from __future__ import annotations

import math

from src.tests.technical.figure_of_8_dribbling import (
    _calibrate,
    _count_loops,
    _pick_two_cones,
)


# --- _pick_two_cones --------------------------------------------------


def test_pick_two_cones_returns_left_to_right():
    a = (300.0, 400.0)
    b = (100.0, 410.0)
    left, right = _pick_two_cones([a, b])
    assert left == b and right == a


def test_pick_two_cones_picks_widest_when_more_clusters():
    cones = [(100.0, 400.0), (250.0, 400.0), (700.0, 400.0)]
    left, right = _pick_two_cones(cones)
    assert left == (100.0, 400.0)
    assert right == (700.0, 400.0)


def test_pick_two_cones_none_with_too_few():
    assert _pick_two_cones([(100.0, 400.0)]) is None
    assert _pick_two_cones([]) is None


# --- _calibrate -------------------------------------------------------


def test_calibrate_uses_cone_pair():
    """Cones at distance 600 px, 3 m apart -> 200 px/m."""
    pair = ((0.0, 400.0), (600.0, 400.0))
    px_per_m, source = _calibrate(
        cone_pair=pair, cone_spacing_m=3.0,
        baseline_bbox_h_px=200.0, assumed_height_m=1.7,
    )
    assert source == "cone-pair"
    assert abs(px_per_m - 200.0) < 1e-6


def test_calibrate_falls_back_to_body_proxy():
    px_per_m, source = _calibrate(
        cone_pair=None, cone_spacing_m=3.0,
        baseline_bbox_h_px=200.0, assumed_height_m=1.7,
    )
    assert source == "body-height-proxy"
    assert abs(px_per_m - (200.0 / 1.7)) < 1e-6


# --- _count_loops -----------------------------------------------------


def _figure_of_8_traj(
    n_loops: int,
    cone_a: tuple[float, float],
    cone_b: tuple[float, float],
    samples_per_loop: int,
) -> list[tuple[int, float, float]]:
    """Continuous figure-of-8 path: athlete starts at the midpoint,
    makes a CCW revolution around cone_a (returning to the midpoint),
    then a CW revolution around cone_b (returning again). Each loop
    is one such full traversal — both half-circles share the midpoint
    so the path is continuous (no teleports)."""
    pts: list[tuple[int, float, float]] = []
    fi = 0
    half_dist = abs(cone_b[0] - cone_a[0]) / 2
    for _loop in range(n_loops):
        for k in range(samples_per_loop):
            t = (k / samples_per_loop) * 2 * math.pi
            if t < math.pi:
                # Half 1: CCW around cone_a, theta from 0 to 2π.
                theta = 2 * t
                x = cone_a[0] + half_dist * math.cos(theta)
                y = cone_a[1] + half_dist * math.sin(theta)
            else:
                # Half 2: CW around cone_b, theta from π to -π.
                t2 = t - math.pi
                theta = math.pi - 2 * t2
                x = cone_b[0] + half_dist * math.cos(theta)
                y = cone_b[1] + half_dist * math.sin(theta)
            pts.append((fi, x, y))
            fi += 1
    return pts


def test_count_loops_two_clean_loops():
    cone_a = (100.0, 400.0)
    cone_b = (700.0, 400.0)
    traj = _figure_of_8_traj(
        n_loops=2, cone_a=cone_a, cone_b=cone_b,
        samples_per_loop=80,
    )
    n = _count_loops(ankle_traj=traj, cone_pair=(cone_a, cone_b))
    assert n == 2


def test_count_loops_zero_when_no_motion():
    cone_a = (100.0, 400.0)
    cone_b = (700.0, 400.0)
    # Athlete stationary on one side of the line.
    traj = [(i, 100.0, 200.0) for i in range(60)]
    n = _count_loops(ankle_traj=traj, cone_pair=(cone_a, cone_b))
    assert n == 0


def test_count_loops_zero_when_no_cones():
    traj = [(i, 100.0 + i, 400.0) for i in range(60)]
    n = _count_loops(ankle_traj=traj, cone_pair=None)
    assert n == 0
