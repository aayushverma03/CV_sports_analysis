"""Tests for the BaseTest contract + score_test rollup."""
from __future__ import annotations

from pathlib import Path

import pytest

from src.tests.base import (
    AnalysisDiagnostics,
    AnalysisResult,
    AthleteProfile,
    BaseTest,
    MetricValue,
    score_test,
)


# --- score_test ---------------------------------------------------------


def test_score_test_linear_sprint_male():
    """All four splits hit `P` thresholds -> agg score == 40 (top of poor / bottom of expected)."""
    metrics = {
        "time_10m_s": MetricValue(raw=2.4, unit="s"),
        "time_20m_s": MetricValue(raw=4.1, unit="s"),
        "time_30m_s": MetricValue(raw=4.4, unit="s"),
        "time_40m_s": MetricValue(raw=5.8, unit="s"),
    }
    scores, test_score = score_test(metrics, "linear-sprint", "M")
    assert set(scores) == set(metrics)
    for s in scores.values():
        assert s.score == pytest.approx(40.0)
        assert s.band == "expected"
    assert test_score.score == pytest.approx(40.0)
    assert test_score.band == "expected"


def test_score_test_skips_metrics_without_benchmark():
    metrics = {
        "time_10m_s": MetricValue(raw=2.4, unit="s"),
        "time_20m_s": MetricValue(raw=4.1, unit="s"),
        "time_30m_s": MetricValue(raw=4.4, unit="s"),
        "time_40m_s": MetricValue(raw=5.8, unit="s"),
        "max_speed_ms": MetricValue(raw=8.0, unit="m/s"),  # no benchmark
    }
    scores, _ = score_test(metrics, "linear-sprint", "M")
    assert "max_speed_ms" not in scores
    assert len(scores) == 4


def test_score_test_raises_when_no_metric_scored():
    metrics = {"unknown_metric": MetricValue(raw=1.0, unit="s")}
    with pytest.raises(ValueError, match="no metric"):
        score_test(metrics, "linear-sprint", "M")


def test_score_test_partial_metrics_renormalises():
    """One of four splits present -> aggregator drops missing weights."""
    metrics = {"time_10m_s": MetricValue(raw=2.4, unit="s")}
    scores, test_score = score_test(metrics, "linear-sprint", "M")
    assert len(scores) == 1
    # Weighted mean over the single relevant metric == that metric's score.
    assert test_score.score == pytest.approx(scores["time_10m_s"].score)


# --- BaseTest -----------------------------------------------------------


def test_basetest_cannot_instantiate_without_run():
    with pytest.raises(TypeError):
        BaseTest()  # abstract


def test_basetest_subclass_runs_end_to_end(tmp_path):
    """A minimal subclass produces an AnalysisResult via score_test."""

    class FakeSprint(BaseTest):
        test_id = "linear-sprint"

        def run(self, video_path, athlete):
            metrics = {
                "time_10m_s": MetricValue(raw=2.4, unit="s"),
                "time_20m_s": MetricValue(raw=4.1, unit="s"),
                "time_30m_s": MetricValue(raw=4.4, unit="s"),
                "time_40m_s": MetricValue(raw=5.8, unit="s"),
            }
            scores, test_score = score_test(metrics, self.test_id, athlete.gender)
            return AnalysisResult(
                test_id=self.test_id,
                athlete=athlete,
                metrics=metrics,
                scores=scores,
                test_score=test_score,
                annotated_video_path=tmp_path / "out.mp4",
                diagnostics=AnalysisDiagnostics(fps_input=30.0, duration_s=6.0),
            )

    athlete = AthleteProfile(gender="M", age=18)
    result = FakeSprint().run(Path("dummy.mp4"), athlete)

    assert result.test_id == "linear-sprint"
    assert result.athlete is athlete
    assert result.test_score.band == "expected"
    assert len(result.scores) == 4
    assert result.completed_at is not None
