"""Unit tests for RSA pipeline-specific helpers."""
from __future__ import annotations

from src.tests.physical.repeated_sprint_ability import (
    _Burst,
    _RunWindow,
    _calibrate,
    _current_sprint,
    _detect_bursts,
)


# --- _calibrate -------------------------------------------------------


def test_calibrate_uses_cone_pair():
    """Cones at distance 600 px, 30 m apart -> 20 px/m."""
    pair = ((0.0, 400.0), (600.0, 400.0))
    px_per_m, source = _calibrate(
        cone_pair=pair, cone_spacing_m=30.0,
        baseline_bbox_h_px=200.0, assumed_height_m=1.7,
    )
    assert source == "cone-pair"
    assert abs(px_per_m - 20.0) < 1e-6


def test_calibrate_falls_back_to_body_proxy():
    px_per_m, source = _calibrate(
        cone_pair=None, cone_spacing_m=30.0,
        baseline_bbox_h_px=200.0, assumed_height_m=1.7,
    )
    assert source == "body-height-proxy"


# --- _detect_bursts ---------------------------------------------------


def _stationary(n: int, x: float = 200.0):
    return [(i, x, 400.0, 200.0, 100.0) for i in range(n)]


def _moving(start_frame: int, n: int, start_x: float = 200.0,
            px_per_frame: float = 10.0):
    return [
        (start_frame + i, start_x + i * px_per_frame, 400.0, 200.0, 100.0)
        for i in range(n)
    ]


def _renumber(entries):
    return [(i, x, y, h, w) for i, (_, x, y, h, w) in enumerate(entries)]


def test_detect_bursts_six_clean_sprints():
    rest = 200
    sprint_n = 100
    parts = [_stationary(60)]
    cur_x = 200.0
    fi = 60
    for _ in range(6):
        parts.append(_moving(start_frame=fi, n=sprint_n, start_x=cur_x))
        fi += sprint_n
        cur_x += sprint_n * 10.0
        parts.append(_stationary(rest, x=cur_x))
        fi += rest
    history = _renumber(sum(parts, []))
    bursts = _detect_bursts(history=history, n_expected=6)
    assert len(bursts) == 6


def test_detect_bursts_keeps_top_n_when_more_present():
    rest = 200
    parts = [_stationary(60)]
    fi = 60
    durations = [50, 200, 80, 200, 50, 200, 200, 200]   # 8 bursts
    cur_x = 200.0
    for n in durations:
        parts.append(_moving(start_frame=fi, n=n, start_x=cur_x))
        fi += n
        cur_x += n * 10.0
        parts.append(_stationary(rest, x=cur_x))
        fi += rest
    history = _renumber(sum(parts, []))
    bursts = _detect_bursts(history=history, n_expected=6)
    assert len(bursts) == 6


def test_detect_bursts_zero_when_stationary():
    history = _stationary(200)
    bursts = _detect_bursts(history=history, n_expected=6)
    assert bursts == []


# --- _current_sprint --------------------------------------------------


def test_current_sprint_pre_start():
    run = _RunWindow(track_id=1, bursts=(
        _Burst(start_frame=100, stop_frame=200),
        _Burst(start_frame=300, stop_frame=400),
    ))
    n, _ = _current_sprint(50, run)
    assert n == 0


def test_current_sprint_during_first():
    run = _RunWindow(track_id=1, bursts=(
        _Burst(start_frame=100, stop_frame=200),
        _Burst(start_frame=300, stop_frame=400),
    ))
    n, _ = _current_sprint(150, run)
    assert n == 1


def test_current_sprint_after_last():
    run = _RunWindow(track_id=1, bursts=(
        _Burst(start_frame=100, stop_frame=200),
    ))
    n, _ = _current_sprint(300, run)
    assert n == 2
