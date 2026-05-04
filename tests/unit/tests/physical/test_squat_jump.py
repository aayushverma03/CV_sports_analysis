"""Unit tests for Squat Jump pipeline-specific helpers."""
from __future__ import annotations

from src.tests.physical.squat_jump import (
    _detect_countermovement,
    _min_knee_angle_during_hold,
)


# --- _min_knee_angle_during_hold --------------------------------------


def test_min_knee_angle_finds_held_squat():
    """30 frames of stable ~110 deg knee angle followed by takeoff —
    return the min of the held window."""
    samples = [(i, 110.0 + (i % 3) * 0.1) for i in range(50)]
    samples += [(50 + i, 130.0 + i * 5) for i in range(20)]   # rising to takeoff
    result = _min_knee_angle_during_hold(
        knee_samples=samples, takeoff_frame=70, hold_min_frames=30,
    )
    assert result is not None
    assert 109 < result < 112


def test_min_knee_angle_none_if_no_hold():
    """Knee angle never settles below the squat threshold — no hold."""
    samples = [(i, 170.0 + (i % 3)) for i in range(60)]
    result = _min_knee_angle_during_hold(
        knee_samples=samples, takeoff_frame=60, hold_min_frames=30,
    )
    assert result is None


def test_min_knee_angle_none_if_window_too_jittery():
    """Knee angle below threshold but oscillates outside the stability
    band — not a hold."""
    samples = [(i, 100.0 + (i % 30) * 5.0) for i in range(60)]
    result = _min_knee_angle_during_hold(
        knee_samples=samples, takeoff_frame=60, hold_min_frames=30,
    )
    assert result is None


# --- _detect_countermovement ------------------------------------------


def test_no_countermovement_when_hip_rises_monotonically():
    """Clean SJ: hip y decreases (rises in world) into takeoff."""
    samples = [(i, 500.0 - i * 2.0, 200.0) for i in range(20)]
    assert _detect_countermovement(
        hip_samples=samples, takeoff_frame=19, lookback_frames=20,
    ) is False


def test_countermovement_when_hip_drops_then_rises():
    """Hip dips down (y increases) before rising for takeoff — flagged."""
    samples = []
    for i in range(20):
        if i < 10:
            y = 500.0 + i * 2.0    # dipping down (countermovement)
        else:
            y = 520.0 - (i - 10) * 5.0
        samples.append((i, y, 200.0))
    assert _detect_countermovement(
        hip_samples=samples, takeoff_frame=19, lookback_frames=20,
    ) is True


def test_countermovement_ignores_tiny_jitter():
    """Small wobbles below the threshold don't trigger a flag."""
    samples = [(i, 500.0 + (i % 2) * 0.5, 200.0) for i in range(20)]
    assert _detect_countermovement(
        hip_samples=samples, takeoff_frame=19, lookback_frames=20,
    ) is False
