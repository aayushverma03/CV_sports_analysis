"""Unit tests for Bangsbo Sprint pipeline-specific helpers."""
from __future__ import annotations

from src.tests.physical.bangsbo_sprint import (
    _Burst,
    _RunWindow,
    _calibrate,
    _current_sprint,
    _detect_bursts,
)


# --- _calibrate -------------------------------------------------------


def test_calibrate_uses_cone_pair_at_known_spacing():
    pair = ((0.0, 400.0), (1500.0, 400.0))
    px_per_m, source = _calibrate(
        cone_pair=pair, cone_spacing_m=34.2,
        baseline_bbox_h_px=200.0, assumed_height_m=1.7,
    )
    assert source == "cone-pair"
    assert abs(px_per_m - (1500.0 / 34.2)) < 1e-6


def test_calibrate_falls_back_to_body_proxy():
    px_per_m, source = _calibrate(
        cone_pair=None, cone_spacing_m=34.2,
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


def test_detect_bursts_finds_three_clean_sprints():
    """Stationary -> sprint -> stationary -> sprint -> stationary -> sprint."""
    rest = 200       # frames between bursts
    sprint_n = 100
    history = _renumber(
        _stationary(60)
        + _moving(start_frame=60, n=sprint_n, px_per_frame=10.0)
        + _stationary(rest, x=200.0 + sprint_n * 10.0)
        + _moving(start_frame=0, n=sprint_n,
                  start_x=200.0 + sprint_n * 10.0, px_per_frame=10.0)
        + _stationary(rest, x=200.0 + 2 * sprint_n * 10.0)
        + _moving(start_frame=0, n=sprint_n,
                  start_x=200.0 + 2 * sprint_n * 10.0, px_per_frame=10.0)
        + _stationary(60, x=200.0 + 3 * sprint_n * 10.0)
    )
    bursts = _detect_bursts(history=history, fps=30.0, n_expected=3)
    assert len(bursts) == 3
    # Bursts should be in chronological order.
    assert bursts[0].start_frame < bursts[1].start_frame < bursts[2].start_frame


def test_detect_bursts_keeps_top_n_when_more_present():
    """5 motion bursts but only 3 expected -> keep the 3 longest."""
    rest = 200
    history_parts = [_stationary(60)]
    fi = 60
    durations = [50, 200, 80, 200, 50]
    cur_x = 200.0
    for n in durations:
        history_parts.append(_moving(start_frame=fi, n=n, start_x=cur_x))
        fi += n
        cur_x += n * 10.0
        history_parts.append(_stationary(rest, x=cur_x))
        fi += rest
    history = _renumber(sum(history_parts, []))
    bursts = _detect_bursts(history=history, fps=30.0, n_expected=3)
    assert len(bursts) == 3


def test_detect_bursts_zero_when_stationary():
    history = _stationary(200)
    bursts = _detect_bursts(history=history, fps=30.0, n_expected=7)
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


def test_current_sprint_between_bursts():
    """Between sprint 1 and sprint 2 — falls into sprint 2's window
    (since current_sprint reports the upcoming sprint)."""
    run = _RunWindow(track_id=1, bursts=(
        _Burst(start_frame=100, stop_frame=200),
        _Burst(start_frame=300, stop_frame=400),
    ))
    n, _ = _current_sprint(250, run)
    assert n == 2


def test_current_sprint_after_last():
    run = _RunWindow(track_id=1, bursts=(
        _Burst(start_frame=100, stop_frame=200),
    ))
    n, _ = _current_sprint(300, run)
    assert n == 2   # n_bursts (1) + 1
