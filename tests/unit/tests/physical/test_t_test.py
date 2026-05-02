"""Unit tests for T-Test multi-track motion analysis."""
from __future__ import annotations

from src.tests.physical.t_test import (
    _TestRun,
    _find_test_run,
    _longest_motion_run,
    _player_hud_fields,
)


def _stationary_track(n: int, x: float = 200.0, y: float = 400.0, h: float = 200.0):
    return [(i, x, y, h) for i in range(n)]


def _moving_track(
    start_frame: int,
    n_frames: int,
    start_x: float = 200.0,
    px_per_frame: float = 8.0,
    y: float = 400.0,
    h: float = 200.0,
):
    return [
        (start_frame + i, start_x + i * px_per_frame, y, h)
        for i in range(n_frames)
    ]


# --- _longest_motion_run --------------------------------------------


def test_longest_run_finds_motion_segment():
    """Stationary -> moving -> stationary returns the moving segment."""
    history = (
        _stationary_track(45)
        + _moving_track(start_frame=45, n_frames=200, px_per_frame=10.0)
        + _stationary_track(50, x=200.0 + 200 * 10.0)
    )
    # Re-key the trailing stationary segment so frames are sequential
    history = [(i, x, y, h) for i, (_, x, y, h) in enumerate(history)]
    result = _longest_motion_run(history)
    assert result is not None
    start, stop = result
    # Run should overlap with the moving frames (45..245 indexes)
    assert 30 < start < 90
    assert 200 < stop <= 250


def test_longest_run_returns_none_for_stationary_track():
    history = _stationary_track(120)
    result = _longest_motion_run(history)
    assert result is None


def test_longest_run_picks_longer_of_two_motion_segments():
    """Two motion segments separated by a LONG gap — pick the longer one.

    Gap is 200 frames, well above the 30-frame merge threshold.
    """
    short_motion = _moving_track(start_frame=20, n_frames=40, px_per_frame=10.0)
    long_motion = _moving_track(
        start_frame=260, n_frames=180, start_x=600.0, px_per_frame=10.0
    )
    history = (
        _stationary_track(20)
        + short_motion
        + _stationary_track(200, x=600.0)
        + long_motion
        + _stationary_track(40, x=600.0 + 180 * 10.0)
    )
    history = [(i, x, y, h) for i, (_, x, y, h) in enumerate(history)]
    result = _longest_motion_run(history)
    assert result is not None
    start, stop = result
    duration = stop - start
    assert duration > 100


def test_gap_merge_joins_close_motion_segments():
    """Two motion segments separated by a short pause (10 frames) should
    merge into one continuous run via the gap-merge step.
    """
    seg1 = _moving_track(start_frame=20, n_frames=80, px_per_frame=10.0)
    last1_x = 200.0 + 80 * 10.0
    pause = _stationary_track(10, x=last1_x)  # short < _GAP_MERGE_FRAMES
    seg2 = _moving_track(start_frame=110, n_frames=80, start_x=last1_x, px_per_frame=10.0)
    last2_x = last1_x + 80 * 10.0
    history = _stationary_track(20) + seg1 + pause + seg2 + _stationary_track(40, x=last2_x)
    history = [(i, x, y, h) for i, (_, x, y, h) in enumerate(history)]
    result = _longest_motion_run(history)
    assert result is not None
    start, stop = result
    # Merged should span seg1 + pause + seg2 (~170 frames), much longer
    # than either individual segment (~80 frames each).
    assert stop - start > 130


# --- _find_test_run --------------------------------------------------


def test_find_test_run_picks_longest_among_tracks():
    """Two tracks: coach (short motion) + player (long motion). Player wins."""
    coach_track = _stationary_track(100)
    coach_track += _moving_track(start_frame=100, n_frames=30, px_per_frame=4.0)
    coach_track += _stationary_track(70, x=100 * 0 + 30 * 4.0 + 200.0)
    coach_track = [(i, x, y, h) for i, (_, x, y, h) in enumerate(coach_track)]

    player_track = _stationary_track(100, x=400.0)
    player_track += _moving_track(
        start_frame=100, n_frames=300, start_x=400.0, px_per_frame=10.0
    )
    player_track += _stationary_track(40, x=400.0 + 300 * 10.0)
    player_track = [(i, x, y, h) for i, (_, x, y, h) in enumerate(player_track)]

    track_history = {1: coach_track, 2: player_track}
    result = _find_test_run(track_history, fps=30.0, min_run_frames=180)
    assert result is not None
    assert result.track_id == 2
    assert result.duration_frames >= 180


def test_find_test_run_none_below_min_frames():
    """Both tracks have only short motion -> no candidate."""
    track = _stationary_track(60) + _moving_track(
        start_frame=60, n_frames=60, px_per_frame=10.0
    ) + _stationary_track(60, x=60 * 10.0 + 200.0)
    track = [(i, x, y, h) for i, (_, x, y, h) in enumerate(track)]

    track_history = {1: track}
    result = _find_test_run(track_history, fps=30.0, min_run_frames=180)
    assert result is None


def test_find_test_run_returns_none_when_no_tracks():
    result = _find_test_run({}, fps=30.0, min_run_frames=180)
    assert result is None


# --- _player_hud_fields ---------------------------------------------


def test_hud_pre_start():
    run = _TestRun(track_id=1, start_frame=100, stop_frame=400)
    fields = _player_hud_fields(frame_idx=50, fps=30.0, run=run)
    assert fields["phase"] == "ready"
    assert fields["time"] == "-"


def test_hud_during_run():
    run = _TestRun(track_id=1, start_frame=100, stop_frame=400)
    # 60 frames after start at 30 fps = 2 s
    fields = _player_hud_fields(frame_idx=160, fps=30.0, run=run)
    assert fields["phase"] == "running"
    assert "2.00" in fields["time"]


def test_hud_post_run():
    run = _TestRun(track_id=1, start_frame=100, stop_frame=400)
    fields = _player_hud_fields(frame_idx=500, fps=30.0, run=run)
    assert fields["phase"] == "finished"
    # Run is 300 frames at 30 fps = 10 s
    assert "10.000" in fields["time"]
