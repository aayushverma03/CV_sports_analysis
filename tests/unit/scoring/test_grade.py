"""Tests for src/scoring/grade.py — band display + roll-up aggregation."""
from __future__ import annotations

import math

import pytest

from src.scoring.grade import (
    aggregate_metric_scores,
    aggregate_subareas_to_area,
    aggregate_tests_to_subarea,
    band_colour_hex,
    format_band,
)


# --- Display -------------------------------------------------------------


@pytest.mark.parametrize(
    "band,expected",
    [
        ("poor", "Poor"),
        ("expected", "Expected"),
        ("above_expected", "Above Expected"),
        ("elite", "Elite"),
    ],
)
def test_format_band(band, expected):
    assert format_band(band) == expected


def test_band_colour_hex_returns_valid_hex():
    for band in ("poor", "expected", "above_expected", "elite"):
        c = band_colour_hex(band)
        assert c.startswith("#")
        assert len(c) == 7


# --- Metric aggregation -------------------------------------------------


def test_metric_mean():
    scores = {"a": 60, "b": 80, "c": 70}
    assert aggregate_metric_scores(scores, method="mean") == 70.0


def test_metric_weighted_mean_full_weights():
    scores = {"a": 60, "b": 80}
    weights = {"a": 0.25, "b": 0.75}
    assert aggregate_metric_scores(scores, method="weighted_mean", weights=weights) == 75.0


def test_metric_weighted_mean_partial_renormalises():
    # Test ran metrics a and b but weights also covered c (which wasn't computed).
    scores = {"a": 60, "b": 80}
    weights = {"a": 0.25, "b": 0.25, "c": 0.50}
    # Effective weights: a=0.25, b=0.25 -> renormalise to a=0.5, b=0.5 -> mean = 70
    assert aggregate_metric_scores(scores, method="weighted_mean", weights=weights) == 70.0


def test_metric_empty_scores_is_nan():
    assert math.isnan(aggregate_metric_scores({}, method="mean"))


def test_metric_unknown_method_raises():
    with pytest.raises(ValueError, match="unknown aggregation method"):
        aggregate_metric_scores({"a": 1}, method="median")


def test_metric_weighted_without_weights_raises():
    with pytest.raises(ValueError, match="weighted_mean requires"):
        aggregate_metric_scores({"a": 1}, method="weighted_mean")


# --- Test → subarea aggregation ----------------------------------------


def test_aggregate_tests_to_endurance_full():
    """Both endurance tests ran. Yo-Yo (60%) + Multistage (40%)."""
    scores = {"yo-yo-intermittent": 70.0, "multistage-fitness": 80.0}
    out = aggregate_tests_to_subarea(scores, "endurance")
    # 70 * 0.6 + 80 * 0.4 = 42 + 32 = 74
    assert out == pytest.approx(74.0)


def test_aggregate_tests_to_endurance_partial_renormalises():
    """Only Yo-Yo ran. Weight re-scales to 1.0 → score = Yo-Yo's score."""
    scores = {"yo-yo-intermittent": 70.0}
    assert aggregate_tests_to_subarea(scores, "endurance") == 70.0


def test_aggregate_tests_to_subarea_no_relevant_tests():
    """Tests provided don't match the subarea — return NaN."""
    scores = {"counter-movement-jump": 80.0}  # CMJ isn't in endurance subarea
    assert math.isnan(aggregate_tests_to_subarea(scores, "endurance"))


def test_aggregate_tests_empty_is_nan():
    assert math.isnan(aggregate_tests_to_subarea({}, "endurance"))


# --- Subarea → area aggregation ----------------------------------------


def test_aggregate_subareas_to_foundation_full():
    """All four foundation subareas present."""
    sub = {
        "speed_agility":      80.0,
        "strength_power":     70.0,
        "endurance":          60.0,
        "mobility_stability": 50.0,
    }
    out = aggregate_subareas_to_area(sub, "foundation")
    # 0.35*80 + 0.30*70 + 0.20*60 + 0.15*50 = 28 + 21 + 12 + 7.5 = 68.5
    assert out == pytest.approx(68.5)


def test_aggregate_subareas_partial_foundation_renormalises():
    """Athlete only did speed_agility tests. Score = that subarea's score."""
    sub = {"speed_agility": 80.0}
    assert aggregate_subareas_to_area(sub, "foundation") == 80.0


def test_aggregate_subareas_to_technical():
    """All three technical subareas present."""
    sub = {
        "dribbling":    75.0,
        "passing":      60.0,
        "ball_control": 90.0,
    }
    out = aggregate_subareas_to_area(sub, "technical")
    # 0.45*75 + 0.35*60 + 0.20*90 = 33.75 + 21 + 18 = 72.75
    assert out == pytest.approx(72.75)


def test_aggregate_subareas_empty_is_nan():
    assert math.isnan(aggregate_subareas_to_area({}, "foundation"))
