"""Tests for the shared player picker."""
from __future__ import annotations

from src.core.tracking.player_picker import (
    pick_by_area_dominance,
    pick_by_object_proximity,
    pick_player,
)


def _track(
    frames: range,
    cx: float = 320.0,
    cy: float = 400.0,
    h: float = 200.0,
    w: float = 100.0,
):
    return [(i, cx, cy, h, w) for i in frames]


# --- pick_by_area_dominance ----------------------------------------


def test_area_dominance_picks_largest_in_most_frames():
    """Track 1: small (h=80,w=40). Track 2: large (h=200,w=100). Both visible
    in all frames. Track 2 wins per-frame in every frame -> dominance 100%."""
    track_history = {
        1: _track(range(100), h=80, w=40),
        2: _track(range(100), h=200, w=100),
    }
    assert pick_by_area_dominance(track_history, min_dominance_frac=0.5) == 2


def test_area_dominance_returns_none_when_no_majority():
    """Track 1 wins frames 0-49, Track 2 wins frames 50-99. Tied 50/50."""
    track_history = {
        1: [
            *_track(range(50), h=200, w=100),
            *_track(range(50, 100), h=80, w=40),
        ],
        2: [
            *_track(range(50), h=80, w=40),
            *_track(range(50, 100), h=200, w=100),
        ],
    }
    # Both win 50% of frames; threshold 0.6 means neither qualifies.
    assert pick_by_area_dominance(track_history, min_dominance_frac=0.6) is None


def test_area_dominance_split_60_40():
    """Track 1 wins 60% of frames. Above 50% threshold -> track 1."""
    track_history = {
        1: [
            *_track(range(60), h=200, w=100),
            *_track(range(60, 100), h=80, w=40),
        ],
        2: [
            *_track(range(60), h=80, w=40),
            *_track(range(60, 100), h=200, w=100),
        ],
    }
    assert pick_by_area_dominance(track_history, min_dominance_frac=0.5) == 1


# --- pick_by_object_proximity ---------------------------------------


def test_proximity_picks_track_close_to_object_with_motion():
    """One track sprints close to an object; another stands far away.
    Sprinter wins."""
    sprinter = _track(range(100), cx=200.0, cy=400.0, h=200, w=100)
    # Sprint motion: cx moves
    sprinter = [(i, 200.0 + i * 30.0, 400.0, 200.0, 100.0) for i in range(100)]

    bystander = _track(range(100), cx=600.0, cy=400.0, h=200, w=100)

    # Object near the sprinter's path
    object_positions = {i: [(300.0 + i * 30.0, 400.0)] for i in range(100)}

    track_history = {1: sprinter, 2: bystander}
    assert pick_by_object_proximity(track_history, object_positions) == 1


# --- pick_player orchestrator ---------------------------------------


def test_pick_player_uses_area_when_clear_winner():
    track_history = {
        1: _track(range(100), h=200, w=100),
        2: _track(range(100), h=80, w=40),
    }
    assert pick_player(track_history) == 1


def test_pick_player_falls_back_to_proximity_when_area_split():
    """Both tracks tied on per-frame area but one is closer to objects."""
    a = [(i, 200.0 + i * 20.0, 400.0, 200.0, 100.0) for i in range(100)]
    b = [(i, 600.0, 400.0, 200.0, 100.0) for i in range(100)]
    object_positions = {i: [(300.0 + i * 20.0, 400.0)] for i in range(100)}
    # Both have identical area. Step 1 returns None (no dominance).
    # Step 2: a is close to objects + moving. Should win.
    track_history = {1: a, 2: b}
    assert pick_player(track_history, object_positions) == 1


def test_pick_player_returns_none_when_all_tracks_short():
    track_history = {1: _track(range(20))}
    assert pick_player(track_history, min_history_frames=60) is None
