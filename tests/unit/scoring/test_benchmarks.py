"""Tests for src/scoring/benchmarks.py — loader, lookup, validation."""
from __future__ import annotations

import pytest

from src.scoring import benchmarks as bm


# --- lookup happy paths --------------------------------------------------


def test_lookup_linear_sprint_male():
    b = bm.lookup("linear-sprint", "time_10m_s", "male")
    assert b.P == 2.4
    assert b.E == 2.1
    assert b.A == 1.8
    assert b.L == 1.6
    assert b.direction == "lower_is_better"
    assert b.unit == "s"
    assert b.gender == "male"


def test_lookup_higher_is_better():
    b = bm.lookup("counter-movement-jump", "jump_height_cm", "female")
    assert b.P == 35
    assert b.L == 60
    assert b.direction == "higher_is_better"


def test_lookup_gender_aliases_resolve():
    a = bm.lookup("t-test", "total_completion_time_s", "M")
    b = bm.lookup("t-test", "total_completion_time_s", "male")
    c = bm.lookup("t-test", "total_completion_time_s", "MALE")
    assert a == b == c


def test_lookup_unknown_gender_raises():
    with pytest.raises(bm.BenchmarkLookupError, match="unsupported gender"):
        bm.lookup("t-test", "total_completion_time_s", "X")


def test_lookup_unknown_test_raises():
    with pytest.raises(bm.BenchmarkLookupError, match="no benchmark file"):
        bm.lookup("does-not-exist", "anything", "male")


def test_lookup_unknown_metric_raises():
    with pytest.raises(bm.BenchmarkLookupError, match="not in benchmarks"):
        bm.lookup("linear-sprint", "fake_metric", "male")


# --- threshold ordering validation ---------------------------------------


def test_validate_higher_is_better_correct():
    bm._validate_thresholds(50, 60, 70, 75, "higher_is_better")  # no raise


def test_validate_higher_is_better_wrong_order():
    with pytest.raises(bm.BenchmarkSchemaError):
        bm._validate_thresholds(75, 70, 60, 50, "higher_is_better")


def test_validate_lower_is_better_correct():
    bm._validate_thresholds(2.4, 2.1, 1.8, 1.6, "lower_is_better")  # no raise


def test_validate_lower_is_better_wrong_order():
    with pytest.raises(bm.BenchmarkSchemaError):
        bm._validate_thresholds(1.6, 1.8, 2.1, 2.4, "lower_is_better")


# --- aggregation ---------------------------------------------------------


def test_load_aggregation_validates_sums():
    bm.clear_cache()
    agg = bm.load_aggregation()
    for area, subs in agg.areas.items():
        assert abs(sum(subs.values()) - 1.0) < 1e-6
    for sub, tests in agg.subareas.items():
        assert abs(sum(tests.values()) - 1.0) < 1e-6


def test_load_aggregation_has_expected_areas():
    agg = bm.load_aggregation()
    assert set(agg.areas) == {"foundation", "technical"}
    assert set(agg.areas["foundation"]) == {
        "speed_agility", "strength_power", "endurance", "mobility_stability",
    }


def test_load_test_aggregation_default_is_mean():
    method, weights = bm.load_test_aggregation("illinois-agility")
    assert method == "mean"
    assert weights == {}


def test_load_test_aggregation_weighted_mean():
    method, weights = bm.load_test_aggregation("linear-sprint")
    assert method == "weighted_mean"
    assert abs(sum(weights.values()) - 1.0) < 1e-6
    assert "time_10m_s" in weights


# --- discovery -----------------------------------------------------------


def test_list_tests_returns_all_21():
    tests = bm.list_tests()
    assert len(tests) == 21
    assert "linear-sprint" in tests
    assert "landing-error-scoring-system" in tests
    assert "45-second-agility-hurdle-jump" in tests
    assert "wall-pass" in tests


# --- caching -------------------------------------------------------------


def test_lookup_uses_cache():
    bm.clear_cache()
    a = bm.lookup("linear-sprint", "time_10m_s", "male")
    b = bm.lookup("linear-sprint", "time_10m_s", "male")
    assert a == b
    # Calling _load_file again should be a cache hit (same object id by lru)
    p = bm._find_file("linear-sprint")
    obj1 = bm._load_file(p)
    obj2 = bm._load_file(p)
    assert obj1 is obj2
