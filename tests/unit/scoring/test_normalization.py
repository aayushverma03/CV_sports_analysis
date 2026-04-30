"""Tests for src/scoring/normalization.py — P/E/A/L formula, both directions."""
from __future__ import annotations

from dataclasses import dataclass

import pytest

from src.scoring.normalization import (
    NormalisedScore,
    _higher_score,
    _lower_score,
    normalise,
    score_to_band,
)


@dataclass
class _T:
    """Minimal stand-in matching the _HasThresholds Protocol."""
    P: float
    E: float
    A: float
    L: float
    direction: str


# --- score_to_band -------------------------------------------------------


@pytest.mark.parametrize(
    "score,expected",
    [
        (1, "poor"),
        (39.99, "poor"),
        (40, "expected"),
        (55, "expected"),
        (69.99, "expected"),
        (70, "above_expected"),
        (85, "above_expected"),
        (89.99, "above_expected"),
        (90, "elite"),
        (95, "elite"),
        (100, "elite"),
    ],
)
def test_score_to_band(score, expected):
    assert score_to_band(score) == expected


# --- higher_is_better ----------------------------------------------------


# CMJ-male thresholds: P=50, E=60, A=70, L=75; F = max(0, 50 - 10) = 40
HB = (50.0, 60.0, 70.0, 75.0)


@pytest.mark.parametrize(
    "value,expected_score,expected_extrapolated",
    [
        (35,  1.0,    True),    # below F
        (40,  1.0,    True),    # exactly F
        (45,  20.5,   False),   # mid-Poor
        (50,  40.0,   False),   # exactly P -> 40 (Expected boundary)
        (55,  55.0,   False),   # mid-Expected
        (60,  70.0,   False),   # exactly E -> 70 (Above boundary)
        (65,  80.0,   False),   # mid-Above
        (70,  90.0,   False),   # exactly A -> 90 (Elite boundary)
        (72,  94.0,   False),   # mid-Elite
        (75,  100.0,  False),   # exactly L
        (80,  100.0,  True),    # above L (capped)
    ],
)
def test_higher_score(value, expected_score, expected_extrapolated):
    score, extrap = _higher_score(value, *HB)
    assert score == pytest.approx(expected_score, abs=1e-9)
    assert extrap == expected_extrapolated


def test_higher_F_clamps_to_zero_when_extrapolation_negative():
    # If E - P > P, F would go negative; max(0, ...) clamps.
    # Use P=10, E=30 -> F = max(0, 10-20) = 0
    score, extrap = _higher_score(0, 10, 30, 50, 70)
    assert score == 1.0
    assert extrap is True
    score2, _ = _higher_score(5, 10, 30, 50, 70)
    # F=0 so score = 1 + 39 * (5-0)/(10-0) = 1 + 19.5 = 20.5
    assert score2 == pytest.approx(20.5)


# --- lower_is_better -----------------------------------------------------


# Linear sprint 10m male: P=2.4, E=2.1, A=1.8, L=1.6; C = 2.4 + 0.3 = 2.7
LB = (2.4, 2.1, 1.8, 1.6)


@pytest.mark.parametrize(
    "value,expected_score,expected_extrapolated",
    [
        (2.85,  1.0,    True),    # past C
        (2.7,   1.0,    True),    # exactly C
        (2.5,   27.0,   False),   # mid-Poor (40 - 39*0.1/0.3 = 27)
        (2.4,   40.0,   False),   # exactly P
        (2.25,  55.0,   False),   # mid-Expected
        (2.1,   70.0,   False),   # exactly E
        (1.95,  80.0,   False),   # mid-Above
        (1.8,   90.0,   False),   # exactly A
        (1.7,   95.0,   False),   # mid-Elite
        (1.6,   100.0,  False),   # exactly L
        (1.5,   100.0,  True),    # below L
    ],
)
def test_lower_score(value, expected_score, expected_extrapolated):
    score, extrap = _lower_score(value, *LB)
    assert score == pytest.approx(expected_score, abs=1e-9)
    assert extrap == expected_extrapolated


# --- normalise() (public API) -------------------------------------------


def test_normalise_higher_returns_normalised_score():
    t = _T(P=50, E=60, A=70, L=75, direction="higher_is_better")
    out = normalise(65, t)
    assert isinstance(out, NormalisedScore)
    assert out.raw_value == 65.0
    assert out.score == pytest.approx(80.0)
    assert out.band == "above_expected"
    assert out.extrapolated is False


def test_normalise_lower_returns_normalised_score():
    t = _T(P=2.4, E=2.1, A=1.8, L=1.6, direction="lower_is_better")
    out = normalise(2.25, t)
    assert out.score == pytest.approx(55.0)
    assert out.band == "expected"
    assert out.extrapolated is False


def test_normalise_unknown_direction_raises():
    t = _T(P=1, E=2, A=3, L=4, direction="target_value")
    with pytest.raises(ValueError, match="unknown direction"):
        normalise(2.5, t)


def test_normalise_with_real_benchmark():
    """Round-trip test using benchmarks.lookup() output."""
    from src.scoring import benchmarks as bm

    b = bm.lookup("counter-movement-jump", "jump_height_cm", "male")
    out = normalise(65, b)
    assert out.score == pytest.approx(80.0)
    assert out.band == "above_expected"
    assert out.extrapolated is False
