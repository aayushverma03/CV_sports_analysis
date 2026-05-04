"""Unit tests for 5 x 10 m Sprint with COD helpers.

Pure helpers: cone-pair selection, turn detection on a synthetic
trajectory, HUD field formatting. Player-picker and run-window are
tested separately under tests/unit/core/tracking/.
"""
from __future__ import annotations

from src.tests.physical.sprint_5x10_cod import (
    _N_SHUTTLES,
    _TestRun,
    _detect_turns,
    _pick_cone_pair,
    _player_hud_fields,
)


# --- _pick_cone_pair --------------------------------------------------


def test_pick_cone_pair_two_clusters_returns_left_to_right():
    a = (300.0, 400.0)
    b = (100.0, 400.0)
    left, right = _pick_cone_pair([a, b])
    assert left == b and right == a


def test_pick_cone_pair_four_cones_returns_end_centroids():
    """4 cones in 2 pairs (start + finish, each ~80px wide, 10m gap):
    return the START centroid and FINISH centroid, not the outermost
    cones."""
    cones = [
        (100.0, 400.0), (180.0, 405.0),    # start pair
        (950.0, 410.0), (1030.0, 408.0),   # finish pair
    ]
    left, right = _pick_cone_pair(cones)
    assert abs(left[0] - 140.0) < 1.0    # start midpoint
    assert abs(right[0] - 990.0) < 1.0   # finish midpoint


def test_pick_cone_pair_three_cones_one_pair_one_loner():
    """2 cones at start + 1 at finish: centroid the pair, single is its
    own end."""
    cones = [(100.0, 400.0), (180.0, 400.0), (1000.0, 400.0)]
    left, right = _pick_cone_pair(cones)
    assert abs(left[0] - 140.0) < 1.0
    assert abs(right[0] - 1000.0) < 1.0


def test_pick_cone_pair_evenly_spaced_falls_back_to_max_separation():
    """No clear gap (cones evenly spaced): use max-x-separation."""
    cones = [(100.0, 400.0), (300.0, 400.0), (500.0, 400.0), (700.0, 400.0)]
    left, right = _pick_cone_pair(cones)
    assert left == (100.0, 400.0)
    assert right == (700.0, 400.0)


def test_pick_cone_pair_too_few():
    assert _pick_cone_pair([(100.0, 400.0)]) == (None, None)
    assert _pick_cone_pair([]) == (None, None)


# --- _detect_turns ----------------------------------------------------


def _shuttle_trajectory(
    start_frame: int,
    cone_left_x: float,
    cone_right_x: float,
    rep_frames: int,
    tail_frames: int = 30,
):
    """Synthetic 5-shuttle x-trajectory: linear ramps between cones,
    overshooting each cone by ~15% of the lane before reversing
    (above the detector's 10% threshold). A short post-finish tail
    keeps the athlete past the final cone so the smoothing window can
    settle on a value above the overshoot threshold (mirrors a real
    video that continues to record past the test end).
    """
    overshoot = 0.15 * (cone_right_x - cone_left_x)
    fis: list[int] = []
    xs: list[float] = []
    fi = start_frame
    targets = [cone_right_x, cone_left_x] * 3
    targets = targets[:_N_SHUTTLES]
    cur = cone_left_x
    for tgt in targets:
        end = tgt + overshoot if tgt > cur else tgt - overshoot
        for k in range(rep_frames):
            t = (k + 1) / rep_frames
            xs.append(cur + (end - cur) * t)
            fis.append(fi)
            fi += 1
        cur = end
    for _ in range(tail_frames):
        xs.append(cur)
        fis.append(fi)
        fi += 1
    return [(int(f), float(x), 400.0, 200.0, 100.0) for f, x in zip(fis, xs)]


def test_detect_turns_finds_five_shuttles():
    cone_left = (100.0, 400.0)
    cone_right = (700.0, 400.0)
    rep_frames = 60
    history = _shuttle_trajectory(
        start_frame=10,
        cone_left_x=cone_left[0],
        cone_right_x=cone_right[0],
        rep_frames=rep_frames,
    )
    result = _detect_turns(
        history=history,
        start_frame=history[0][0],
        stop_frame=history[-1][0],
        cone_a=cone_left,
        cone_b=cone_right,
        cone_dist_px=cone_right[0] - cone_left[0],
        fps=30.0,
    )
    # 5 shuttles -> 6 boundaries (start + 5 ends).
    assert len(result) == _N_SHUTTLES + 1
    # Boundaries are monotonically increasing.
    assert all(result[i] < result[i + 1] for i in range(len(result) - 1))


def test_detect_turns_returns_empty_on_no_motion():
    cone_left = (100.0, 400.0)
    cone_right = (700.0, 400.0)
    # Stationary track at the start cone.
    history = [(i, 100.0, 400.0, 200.0, 100.0) for i in range(120)]
    result = _detect_turns(
        history=history,
        start_frame=0,
        stop_frame=119,
        cone_a=cone_left,
        cone_b=cone_right,
        cone_dist_px=600.0,
        fps=30.0,
    )
    # No turns detected -> just the start boundary, or fewer than 6.
    assert len(result) < _N_SHUTTLES + 1


# --- HUD --------------------------------------------------------------


def test_hud_pre_start():
    run = _TestRun(track_id=1, start_frame=100, stop_frame=400,
                   rep_boundary_frames=(100, 160, 220, 280, 340, 400))
    fields = _player_hud_fields(50, 30.0, run, history_by_frame={}, px_per_m=10.0)
    assert fields["phase"] == "ready"
    assert fields["shuttle"] == "-"


def test_hud_during_run_shows_current_shuttle():
    run = _TestRun(track_id=1, start_frame=100, stop_frame=400,
                   rep_boundary_frames=(100, 160, 220, 280, 340, 400))
    # frame 230 is in shuttle 3 (220 < 230 <= 280).
    fields = _player_hud_fields(230, 30.0, run, history_by_frame={}, px_per_m=10.0)
    assert fields["phase"] == "running"
    assert fields["shuttle"] == "3/5"


def test_hud_post_run_total_time():
    run = _TestRun(track_id=1, start_frame=100, stop_frame=400,
                   rep_boundary_frames=(100, 160, 220, 280, 340, 400))
    fields = _player_hud_fields(500, 30.0, run, history_by_frame={}, px_per_m=10.0)
    assert fields["phase"] == "finished"
    assert fields["shuttle"] == "5/5"
    # 300 frames / 30 fps = 10.0 s
    assert "10.000" in fields["time"]
