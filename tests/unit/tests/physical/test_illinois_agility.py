"""Unit tests for Illinois Agility multi-track motion analysis.

Same shape as the T-Test test suite — the pipeline is deliberately a
near-copy of t-test (both use the multi-track + peak-motion picker
pattern). Once 4.9 5x10m COD lands, the shared logic should move to
src/tests/families/agility_family.py.
"""
from __future__ import annotations

from src.tests.physical.illinois_agility import (
    _TestRun,
    _find_test_run,
    _longest_motion_run,
    _player_hud_fields,
)


def _stationary_track(n: int, x: float = 200.0, y: float = 400.0, h: float = 200.0):
    return [(i, x, y, h) for i in range(n)]


# --- _longest_motion_run --------------------------------------------


def test_longest_run_finds_motion_segment():
    history = (
        _stationary_track(45)
        + [(45 + i, 200.0 + i * 10.0, 400.0, 200.0) for i in range(200)]
        + _stationary_track(50, x=200.0 + 200 * 10.0)
    )
    history = [(i, x, y, h) for i, (_, x, y, h) in enumerate(history)]
    result = _longest_motion_run(history)
    assert result is not None
    start, stop = result
    assert 30 < start < 90
    assert 200 < stop <= 250


def test_longest_run_returns_none_for_stationary_track():
    history = _stationary_track(120)
    result = _longest_motion_run(history)
    assert result is None


# --- _find_test_run --------------------------------------------------


def test_find_test_run_picks_track_with_highest_peak_motion():
    """Two tracks: bystander walking + athlete sprinting. Athlete wins by peak."""
    bystander = _stationary_track(100, x=200.0, h=80.0)
    bystander += [(100 + i, 200.0 + i * 4.0, 400.0, 80.0) for i in range(400)]
    bystander += _stationary_track(40, x=200.0 + 400 * 4.0, h=80.0)
    bystander = [(i, x, y, h) for i, (_, x, y, h) in enumerate(bystander)]

    athlete_track = _stationary_track(100, x=400.0, h=200.0)
    athlete_track += [(100 + i, 400.0 + i * 30.0, 400.0, 200.0) for i in range(400)]
    athlete_track += _stationary_track(40, x=400.0 + 400 * 30.0, h=200.0)
    athlete_track = [(i, x, y, h) for i, (_, x, y, h) in enumerate(athlete_track)]

    track_history = {1: bystander, 2: athlete_track}
    result = _find_test_run(track_history, fps=30.0, min_run_frames=360)
    assert result is not None
    assert result.track_id == 2
    assert result.duration_frames >= 360


def test_find_test_run_none_below_min_frames():
    """Sprint that's too short to be a valid Illinois run."""
    track = _stationary_track(60) + [
        (60 + i, 200.0 + i * 30.0, 400.0, 200.0) for i in range(200)
    ] + _stationary_track(60, x=200.0 + 200 * 30.0)
    track = [(i, x, y, h) for i, (_, x, y, h) in enumerate(track)]
    track_history = {1: track}
    result = _find_test_run(track_history, fps=30.0, min_run_frames=360)
    assert result is None


# --- HUD --------------------------------------------------------------


def test_hud_during_run():
    run = _TestRun(track_id=1, start_frame=100, stop_frame=550)
    fields = _player_hud_fields(frame_idx=160, fps=30.0, run=run)
    assert fields["phase"] == "running"
    assert "2.00" in fields["time"]


def test_hud_post_run():
    run = _TestRun(track_id=1, start_frame=100, stop_frame=550)
    fields = _player_hud_fields(frame_idx=600, fps=30.0, run=run)
    assert fields["phase"] == "finished"
    # 450 frames / 30 fps = 15.0 s
    assert "15.000" in fields["time"]
