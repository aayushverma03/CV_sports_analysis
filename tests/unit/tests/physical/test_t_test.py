"""Unit tests for T-Test pipeline-specific helpers.

Player picking lives in src/core/tracking/player_picker.py and the
run-window finder in src/core/tracking/run_window.py — both are tested
separately. This file covers only the test-specific HUD.
"""
from __future__ import annotations

from src.tests.physical.t_test import _TestRun, _player_hud_fields


def test_hud_pre_start():
    run = _TestRun(track_id=1, start_frame=100, stop_frame=400)
    fields = _player_hud_fields(frame_idx=50, fps=30.0, run=run)
    assert fields["phase"] == "ready"


def test_hud_during_run():
    run = _TestRun(track_id=1, start_frame=100, stop_frame=400)
    fields = _player_hud_fields(frame_idx=160, fps=30.0, run=run)
    assert fields["phase"] == "running"
    assert "2.00" in fields["time"]


def test_hud_post_run():
    run = _TestRun(track_id=1, start_frame=100, stop_frame=400)
    fields = _player_hud_fields(frame_idx=500, fps=30.0, run=run)
    assert fields["phase"] == "finished"
    assert "10.000" in fields["time"]
