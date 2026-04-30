"""Shared types + base class for all test pipelines.

Every `src/tests/<domain>/<test>.py` subclasses `BaseTest` and returns an
`AnalysisResult`. Scoring (raw metric -> 0-100, band, test-level
aggregation) is centralised in `score_test` so each subclass owns only
CV + metric computation.

Schema mirrors `docs/api/API_SPEC.md`. Pydantic wrappers belong in
`src/api/schemas.py` when the API layer arrives; pure dataclasses here
keep the test layer dependency-light.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from src.scoring.benchmarks import (
    BenchmarkLookupError,
    load_test_aggregation,
    lookup,
)
from src.scoring.grade import aggregate_metric_scores
from src.scoring.normalization import Band, normalise, score_to_band

Gender = Literal["M", "F", "X"]


@dataclass(frozen=True)
class AthleteProfile:
    """The athlete the analysis is being run for.

    Required for every pipeline call (hard rule #8). `gender` keys benchmark
    lookup; `age` is informational in v1 (no age bands).
    """

    gender: Gender
    age: int
    athlete_id: str | None = None


@dataclass(frozen=True)
class MetricValue:
    """One raw metric output, paired with its unit string."""

    raw: float
    unit: str


@dataclass(frozen=True)
class MetricScore:
    """Per-metric scoring output: raw value + 0-100 score + band."""

    raw_value: float
    raw_unit: str
    score: float
    band: Band
    extrapolated: bool


@dataclass(frozen=True)
class TestScore:
    """Test-level rolled-up score (mean / weighted_mean of metric scores)."""

    score: float
    band: Band


@dataclass(frozen=True)
class AnalysisDiagnostics:
    """Per-run quality signals. Subclasses populate what they can measure."""

    fps_input: float
    duration_s: float


@dataclass(frozen=True)
class AnalysisResult:
    """Final return type from `BaseTest.run`."""

    test_id: str
    athlete: AthleteProfile
    metrics: dict[str, MetricValue]
    scores: dict[str, MetricScore]
    test_score: TestScore
    annotated_video_path: Path
    diagnostics: AnalysisDiagnostics
    completed_at: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )


def score_test(
    metrics: dict[str, MetricValue],
    test_id: str,
    gender: str,
) -> tuple[dict[str, MetricScore], TestScore]:
    """Apply benchmark normalisation per metric, then test-level aggregation.

    Metrics without a benchmark entry are skipped (informational metrics
    are allowed to coexist with scored ones). The returned `scores` dict
    contains only metrics that successfully scored. `test_score` aggregates
    those using the YAML's `aggregation` block (method + weights).

    Raises
    ------
    ValueError
        If no metrics scored (the test would have no overall score).
    """
    scores: dict[str, MetricScore] = {}
    for metric_id, mv in metrics.items():
        try:
            bench = lookup(test_id, metric_id, gender)
        except BenchmarkLookupError:
            continue
        ns = normalise(mv.raw, bench)
        scores[metric_id] = MetricScore(
            raw_value=ns.raw_value,
            raw_unit=mv.unit,
            score=ns.score,
            band=ns.band,
            extrapolated=ns.extrapolated,
        )

    if not scores:
        raise ValueError(
            f"no metric in {test_id!r} matched a benchmark entry for gender={gender!r}"
        )

    method, weights = load_test_aggregation(test_id)
    score_dict = {k: s.score for k, s in scores.items()}
    agg = aggregate_metric_scores(
        score_dict, method=method, weights=weights or None
    )
    return scores, TestScore(score=agg, band=score_to_band(agg))


class BaseTest(ABC):
    """Abstract base for every test pipeline.

    Subclasses set `test_id` (class attribute) and implement `run`.
    Family base classes (`SprintFamily`, etc.) will land in Phase 4
    extracting whatever is genuinely shared by their first concrete test.
    """

    test_id: str

    @abstractmethod
    def run(
        self, video_path: Path, athlete: AthleteProfile
    ) -> AnalysisResult:
        """Execute the pipeline and return the final result."""
