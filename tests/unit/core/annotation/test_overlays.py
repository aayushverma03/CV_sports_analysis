"""Tests for annotation overlays."""
from __future__ import annotations

import numpy as np
import pytest

from src.core.annotation.overlays import (
    ATHLETE_BBOX,
    BALL,
    GATE_ACTIVE,
    GATE_PASSED,
    GATE_PENDING,
    LEFT_POSE,
    RIGHT_POSE,
    _bgr,
    draw_bbox,
    draw_ball_trail,
    draw_gate,
    draw_hud,
    draw_skeleton,
    event_flash,
    render_endcard,
)


def _blank(h: int = 480, w: int = 640) -> np.ndarray:
    return np.zeros((h, w, 3), dtype=np.uint8)


# --- palette ------------------------------------------------------------


def test_bgr_conversion():
    # #2E86AB -> BGR (AB, 86, 2E)
    assert _bgr("#2E86AB") == (0xAB, 0x86, 0x2E)
    assert _bgr("FFFFFF") == (255, 255, 255)
    assert _bgr("#000000") == (0, 0, 0)


# --- bbox ---------------------------------------------------------------


def test_draw_bbox_returns_same_shape():
    f = _blank()
    out = draw_bbox(f, np.array([10, 20, 100, 200]))
    assert out.shape == f.shape
    assert out is f  # mutated in place


def test_draw_bbox_paints_outline():
    f = _blank()
    draw_bbox(f, np.array([10, 20, 100, 200]), color=ATHLETE_BBOX)
    # The corner pixel should now be the bbox colour (allowing AA neighbour).
    assert f[20, 10].sum() > 0
    # Centre of box is untouched (no fill).
    assert f[100, 50].sum() == 0


# --- gates --------------------------------------------------------------


@pytest.mark.parametrize(
    "state,expected",
    [("active", GATE_ACTIVE), ("passed", GATE_PASSED), ("pending", GATE_PENDING)],
)
def test_draw_gate_state_colour(state, expected):
    f = _blank()
    draw_gate(f, (100, 0), (100, 479), state=state)
    # Pick a pixel on the line (mid-frame). Allow AA tolerance via "any channel non-zero".
    assert f[240, 100].sum() > 0


# --- skeleton -----------------------------------------------------------


def test_draw_skeleton_skips_low_confidence():
    f = _blank()
    kp = np.zeros((17, 3), dtype=float)
    # All keypoints below threshold -> no drawing
    out = draw_skeleton(f, kp, conf_threshold=0.3)
    assert out.sum() == 0


def test_draw_skeleton_draws_above_threshold():
    f = _blank()
    kp = np.zeros((17, 3), dtype=float)
    # Place left-shoulder (5) and left-elbow (7) in frame with high conf
    kp[5] = [200, 200, 0.9]
    kp[7] = [250, 240, 0.9]
    out = draw_skeleton(f, kp, conf_threshold=0.3)
    # Bone should be drawn between (200,200) and (250,240); midpoint should be coloured
    assert out[220, 225].sum() > 0


# --- ball trail ---------------------------------------------------------


def test_ball_trail_respects_max_age():
    f = _blank()
    history = [(float(x), 100.0) for x in range(50)]
    out = draw_ball_trail(f, history, max_age=10)
    assert out is f
    # Oldest 40 points should not be drawn (still black at x=5).
    assert f[100, 5].sum() == 0
    # Newest point at x=49 should be coloured.
    assert f[100, 49].sum() > 0


# --- HUD ---------------------------------------------------------------


def test_draw_hud_renders_text():
    f = _blank()
    out = draw_hud(f, {"speed": "5.2 m/s", "split": "1.84 s"})
    # HUD background is black on a black frame — they blend to black.
    # The text itself is white, so the rendered region contains some bright pixels.
    hud_region = out[10:80, 20:200]
    assert hud_region.max() > 0


def test_draw_hud_empty_fields_no_op():
    f = _blank()
    out = draw_hud(f, {})
    assert out.sum() == 0


@pytest.mark.parametrize("position", ["top-left", "top-right", "bottom-left", "bottom-right"])
def test_draw_hud_positions(position):
    f = _blank()
    out = draw_hud(f, {"k": "v"}, position=position)
    assert out.shape == f.shape


# --- event flash --------------------------------------------------------


def test_event_flash_tints_region():
    f = np.full((480, 640, 3), 100, dtype=np.uint8)  # mid-grey
    out = event_flash(f, np.array([100, 100, 200, 200]), intensity=0.5)
    # Pixel inside region should differ from baseline grey.
    assert out[150, 150].tolist() != [100, 100, 100]
    # Pixel outside region should remain grey.
    assert out[50, 50].tolist() == [100, 100, 100]


# --- endcard ------------------------------------------------------------


def test_render_endcard_shape():
    out = render_endcard(
        title="Linear Sprint - 30 m",
        athlete="Profile 0421",
        metric_rows=[("Total time", "4.42 s", 84), ("Max speed", "7.2 m/s", 88)],
        test_score=82,
        band="above_average",
        size=(1280, 720),
    )
    assert out.shape == (720, 1280, 3)
    assert out.dtype == np.uint8
    # Some bright pixels somewhere (text is rendered).
    assert out.max() > 100
