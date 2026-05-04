"""Tests for the shared run-window finder + cone clusterer."""
from __future__ import annotations

from src.core.tracking.run_window import (
    cluster_object_positions,
    find_run_on_track,
    longest_motion_run,
)


def _stationary_track(n: int, x: float = 200.0, y: float = 400.0,
                      h: float = 200.0, w: float = 100.0):
    return [(i, x, y, h, w) for i in range(n)]


def _moving_track(
    start_frame: int,
    n_frames: int,
    start_x: float = 200.0,
    px_per_frame: float = 8.0,
    y: float = 400.0,
    h: float = 200.0,
    w: float = 100.0,
):
    return [
        (start_frame + i, start_x + i * px_per_frame, y, h, w)
        for i in range(n_frames)
    ]


def _renumber(entries):
    return [(i, x, y, h, w) for i, (_, x, y, h, w) in enumerate(entries)]


# --- longest_motion_run --------------------------------------------


def test_longest_run_finds_motion_segment():
    history = _renumber(
        _stationary_track(45)
        + _moving_track(start_frame=45, n_frames=200, px_per_frame=10.0)
        + _stationary_track(50, x=200.0 + 200 * 10.0)
    )
    result = longest_motion_run(history)
    assert result is not None
    start, stop = result
    assert 30 < start < 90
    assert 200 < stop <= 250


def test_longest_run_returns_none_for_stationary_track():
    history = _stationary_track(120)
    assert longest_motion_run(history) is None


def test_longest_run_picks_longer_of_two_motion_segments():
    short_motion = _moving_track(start_frame=20, n_frames=40, px_per_frame=10.0)
    long_motion = _moving_track(
        start_frame=260, n_frames=180, start_x=600.0, px_per_frame=10.0
    )
    history = _renumber(
        _stationary_track(20)
        + short_motion
        + _stationary_track(200, x=600.0)
        + long_motion
        + _stationary_track(40, x=600.0 + 180 * 10.0)
    )
    result = longest_motion_run(history)
    assert result is not None
    start, stop = result
    assert stop - start > 100


def test_gap_merge_joins_close_motion_segments():
    seg1 = _moving_track(start_frame=20, n_frames=80, px_per_frame=10.0)
    last1_x = 200.0 + 80 * 10.0
    pause = _stationary_track(10, x=last1_x)
    seg2 = _moving_track(
        start_frame=110, n_frames=80, start_x=last1_x, px_per_frame=10.0
    )
    last2_x = last1_x + 80 * 10.0
    history = _renumber(
        _stationary_track(20) + seg1 + pause + seg2 + _stationary_track(40, x=last2_x)
    )
    result = longest_motion_run(history)
    assert result is not None
    start, stop = result
    assert stop - start > 130


def test_teleport_breaks_run_so_two_attempts_are_not_merged():
    """Track with an ID-swap teleport between two motion segments must
    return only the longer of the two, never the merged span."""
    seg1 = _moving_track(start_frame=20, n_frames=120, px_per_frame=10.0)
    last1_x = 200.0 + 120 * 10.0
    teleport_x = 3000.0
    seg2 = _moving_track(
        start_frame=140, n_frames=240, start_x=teleport_x, px_per_frame=10.0
    )
    history = _renumber(
        _stationary_track(20) + seg1 + seg2
        + _stationary_track(40, x=teleport_x + 240 * 10.0)
    )
    result = longest_motion_run(history)
    assert result is not None
    start, stop = result
    assert 200 <= (stop - start) <= 280


# --- find_run_on_track ----------------------------------------------


def test_find_run_on_track_returns_window():
    history = _renumber(
        _stationary_track(60)
        + _moving_track(start_frame=60, n_frames=240, px_per_frame=10.0)
        + _stationary_track(40, x=200.0 + 240 * 10.0)
    )
    result = find_run_on_track(history, min_run_frames=180)
    assert result is not None
    start, stop = result
    assert stop - start >= 180


def test_find_run_on_track_none_below_min():
    history = _renumber(
        _stationary_track(60)
        + _moving_track(start_frame=60, n_frames=60, px_per_frame=10.0)
        + _stationary_track(60, x=200.0 + 60 * 10.0)
    )
    assert find_run_on_track(history, min_run_frames=180) is None


# --- cluster_object_positions ---------------------------------------


def test_cluster_collapses_close_detections():
    detections = [(100.0, 200.0), (105.0, 205.0), (95.0, 198.0),
                  (102.0, 201.0), (98.0, 199.0)]
    out = cluster_object_positions(detections, radius_px=30.0, min_count=3)
    assert len(out) == 1
    cx, cy = out[0]
    assert 95 < cx < 110 and 195 < cy < 210


def test_cluster_separates_distant_groups():
    detections = [(100.0, 200.0)] * 5 + [(500.0, 200.0)] * 5
    out = cluster_object_positions(detections, radius_px=30.0, min_count=3)
    assert len(out) == 2


def test_cluster_filters_below_min_count():
    detections = [(100.0, 200.0)] * 2 + [(500.0, 200.0)] * 5
    out = cluster_object_positions(detections, radius_px=30.0, min_count=3)
    assert len(out) == 1
    assert abs(out[0][0] - 500.0) < 5
