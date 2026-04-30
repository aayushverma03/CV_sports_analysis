"""Tests for the Jump metric batch (Phase 2.8 - 2.11)."""
from __future__ import annotations

import pytest

from src.metrics.jump.flight_time_s import flight_time_s
from src.metrics.jump.ground_contact_time_s import ground_contact_time_s
from src.metrics.jump.jump_height_cm import GRAVITY_MS2, jump_height_cm
from src.metrics.jump.rsi import rsi


# --- flight_time_s ------------------------------------------------------


def test_flight_time_s_basic():
    assert flight_time_s(takeoff_frame=100, landing_frame=121, fps=30.0) == pytest.approx(0.7)


def test_flight_time_s_zero_when_landing_before_takeoff():
    assert flight_time_s(takeoff_frame=130, landing_frame=120, fps=30.0) == 0.0


# --- jump_height_cm -----------------------------------------------------


def test_jump_height_cm_zero_flight_is_zero():
    assert jump_height_cm(0.0) == 0.0


def test_jump_height_cm_known_flight():
    # Round numbers: 0.6 s flight -> h = 9.81 * 0.36 / 8 * 100 ≈ 44.145 cm
    expected = GRAVITY_MS2 * 0.36 / 8.0 * 100.0
    assert jump_height_cm(0.6) == pytest.approx(expected)


def test_jump_height_cm_doubles_flight_quadruples_height():
    h1 = jump_height_cm(0.5)
    h2 = jump_height_cm(1.0)
    assert h2 / h1 == pytest.approx(4.0)


@pytest.mark.parametrize(
    "flight,expected_cm",
    [
        # User benchmark cross-checks: flight_time threshold should yield
        # close-to-listed jump_height threshold for the same gender/band.
        (0.535, 35.10),   # Female P (height P=35)
        (0.700, 60.07),   # Male   E (height E=60)
        (0.780, 74.61),   # Male   L (height L=75)
    ],
)
def test_jump_height_cm_realistic_ranges(flight, expected_cm):
    """Sanity-check the formula against benchmark flight times from user data."""
    assert jump_height_cm(flight) == pytest.approx(expected_cm, abs=0.1)


# --- ground_contact_time_s ----------------------------------------------


def test_ground_contact_time_s_basic():
    # 30 fps, 6-frame contact -> 0.2 s
    assert ground_contact_time_s(landing_frame=100, rebound_takeoff_frame=106, fps=30.0) == pytest.approx(0.2)


def test_ground_contact_time_s_zero_when_takeoff_before_landing():
    assert ground_contact_time_s(landing_frame=110, rebound_takeoff_frame=100, fps=30.0) == 0.0


# --- rsi ----------------------------------------------------------------


def test_rsi_basic():
    # 30 cm rebound / 0.2 s contact = 1.5 m/s
    assert rsi(rebound_height_cm=30.0, ground_contact_time_s=0.2) == pytest.approx(1.5)


def test_rsi_elite_range():
    # User benchmark: male elite RSI L = 3.0. 60 cm rebound / 0.2 s -> 3.0 ✓
    assert rsi(rebound_height_cm=60.0, ground_contact_time_s=0.2) == pytest.approx(3.0)


def test_rsi_higher_is_better():
    """Shorter contact time at same rebound height = higher RSI = better."""
    fast = rsi(40.0, 0.15)
    slow = rsi(40.0, 0.30)
    assert fast > slow
