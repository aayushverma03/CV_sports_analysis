"""Tests for the shared agility-family pipeline base.

Tests cover the default HUD formatter and the abstract-class guard.
The full pipeline (run() method) is exercised end-to-end by the
agility test smoke scripts.
"""
from __future__ import annotations

import pytest

from src.tests.families.agility_family import AgilityFamilyTest, AgilityRun


class _DummyAgility(AgilityFamilyTest):
    test_id = "dummy"
    endcard_title = "Dummy"
    min_run_frames = 60


def _hud_for(frame_idx: int, run: AgilityRun, fps: float = 30.0) -> dict[str, str]:
    # Bypass __init__ (which loads pose/tracker models) by calling the
    # HUD method as if unbound.
    return AgilityFamilyTest._hud_fields(  # type: ignore[arg-type]
        None, frame_idx, fps, run,
    )


def test_hud_pre_start():
    run = AgilityRun(track_id=1, start_frame=100, stop_frame=400)
    fields = _hud_for(50, run)
    assert fields["phase"] == "ready"
    assert fields["time"] == "-"


def test_hud_during_run():
    run = AgilityRun(track_id=1, start_frame=100, stop_frame=400)
    fields = _hud_for(160, run)
    assert fields["phase"] == "running"
    assert "2.00" in fields["time"]


def test_hud_post_run():
    run = AgilityRun(track_id=1, start_frame=100, stop_frame=400)
    fields = _hud_for(500, run)
    assert fields["phase"] == "finished"
    assert "10.000" in fields["time"]


def test_subclass_must_set_test_id_and_min_run_frames():
    """Bare subclass without overrides should raise on instantiation."""
    class _Bad(AgilityFamilyTest):
        pass
    with pytest.raises(NotImplementedError):
        _Bad()
