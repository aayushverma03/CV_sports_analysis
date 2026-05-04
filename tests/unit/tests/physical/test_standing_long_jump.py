"""Unit tests for Standing Long Jump pipeline-specific helpers."""
from __future__ import annotations

from src.tests.physical.standing_long_jump import (
    _horizontal_distance_cm,
    _peak_height_cm,
)


def _samples(
    points: list[tuple[int, float, float]],
    bbox_h: float = 200.0,
) -> list[tuple[int, float, float, float]]:
    return [(fi, ax, ay, bbox_h) for (fi, ax, ay) in points]


# --- _horizontal_distance_cm ------------------------------------------


def test_distance_matches_pixel_displacement_at_known_scale():
    """200 px horizontal displacement at 100 px/m -> 200 cm."""
    samples = _samples([
        (10, 100.0, 500.0),  # takeoff
        (15, 150.0, 480.0),
        (25, 200.0, 470.0),
        (35, 250.0, 480.0),
        (50, 300.0, 500.0),  # landing — moved 200 px right
    ])
    d = _horizontal_distance_cm(
        ankle_samples=samples, takeoff_frame=10, landing_frame=50,
        px_per_m=100.0,
    )
    assert abs(d - 200.0) < 0.5


def test_distance_uses_absolute_value():
    """Athlete jumps to the LEFT (decreasing x) — distance is positive."""
    samples = _samples([
        (10, 500.0, 500.0),
        (50, 300.0, 500.0),
    ])
    d = _horizontal_distance_cm(
        ankle_samples=samples, takeoff_frame=10, landing_frame=50,
        px_per_m=100.0,
    )
    assert abs(d - 200.0) < 0.5


def test_distance_falls_back_to_nearest_when_exact_frame_missing():
    """Takeoff frame index isn't sampled exactly — use closest sample."""
    samples = _samples([
        (8, 100.0, 500.0),    # nearest to takeoff 10
        (50, 300.0, 500.0),
    ])
    d = _horizontal_distance_cm(
        ankle_samples=samples, takeoff_frame=10, landing_frame=50,
        px_per_m=100.0,
    )
    assert abs(d - 200.0) < 0.5


# --- _peak_height_cm --------------------------------------------------


def test_peak_height_from_ankle_dip_in_y():
    """Ankle starts at y=500, rises to y=400 (100 px up), back to 500."""
    samples = _samples([
        (10, 100.0, 500.0),  # takeoff
        (20, 150.0, 450.0),
        (25, 200.0, 400.0),  # peak
        (35, 250.0, 450.0),
        (50, 300.0, 500.0),  # landing
    ])
    h = _peak_height_cm(
        ankle_samples=samples, takeoff_frame=10, landing_frame=50,
        px_per_m=100.0,
    )
    assert h is not None
    # 100 px / 100 px/m = 1m = 100 cm
    assert abs(h - 100.0) < 0.5


def test_peak_height_zero_when_no_rise():
    """Ankle stays at the same height through the 'flight' (degenerate)."""
    samples = _samples([
        (10, 100.0, 500.0),
        (50, 300.0, 500.0),
    ])
    h = _peak_height_cm(
        ankle_samples=samples, takeoff_frame=10, landing_frame=50,
        px_per_m=100.0,
    )
    assert h == 0.0


def test_peak_height_none_with_too_few_samples():
    samples = _samples([(10, 100.0, 500.0)])
    h = _peak_height_cm(
        ankle_samples=samples, takeoff_frame=10, landing_frame=50,
        px_per_m=100.0,
    )
    assert h is None
