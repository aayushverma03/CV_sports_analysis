"""Raw metric value -> 0-100 score, per `docs/scoring/NORMALIZATION.md`.

Two directions implemented in v1: `higher_is_better` and `lower_is_better`.
Both use the four-band P/E/A/L scheme with extrapolated edges (F or C) so
values outside [P, L] still produce a defined score in [1, 100].
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol

Direction = Literal["higher_is_better", "lower_is_better"]
Band = Literal["poor", "expected", "above_expected", "elite"]


class _HasThresholds(Protocol):
    P: float
    E: float
    A: float
    L: float
    direction: Direction


@dataclass(frozen=True)
class NormalisedScore:
    raw_value: float
    score: float                    # 1.0..100.0
    band: Band
    extrapolated: bool              # True when value fell outside [F, C]


def score_to_band(score: float) -> Band:
    """Map a 1..100 score to its performance band.

    Boundaries fall into the higher band: 40 -> expected, 70 -> above_expected,
    90 -> elite.
    """
    if score < 40:
        return "poor"
    if score < 70:
        return "expected"
    if score < 90:
        return "above_expected"
    return "elite"


def _higher_score(value: float, P: float, E: float, A: float, L: float) -> tuple[float, bool]:
    F = max(0.0, P - (E - P))
    if value <= F:
        return 1.0, True
    if value <= P:
        return 1.0 + 39.0 * (value - F) / (P - F), False
    if value <= E:
        return 40.0 + 30.0 * (value - P) / (E - P), False
    if value <= A:
        return 70.0 + 20.0 * (value - E) / (A - E), False
    if value <= L:
        return 90.0 + 10.0 * (value - A) / (L - A), False
    return 100.0, True


def _lower_score(value: float, P: float, E: float, A: float, L: float) -> tuple[float, bool]:
    C = P + (P - E)
    if value >= C:
        return 1.0, True
    if value >= P:
        return 40.0 - 39.0 * (value - P) / (C - P), False
    if value >= E:
        return 40.0 + 30.0 * (P - value) / (P - E), False
    if value >= A:
        return 70.0 + 20.0 * (E - value) / (E - A), False
    if value >= L:
        return 90.0 + 10.0 * (A - value) / (A - L), False
    return 100.0, True


def normalise(value: float, thresholds: _HasThresholds) -> NormalisedScore:
    """Convert `value` to a NormalisedScore using `thresholds.direction` + P/E/A/L.

    `thresholds` is duck-typed: any object with `P`, `E`, `A`, `L`, and
    `direction` attributes works (e.g. `src.scoring.benchmarks.Benchmark`).
    """
    if thresholds.direction == "higher_is_better":
        score, extrapolated = _higher_score(
            value, thresholds.P, thresholds.E, thresholds.A, thresholds.L
        )
    elif thresholds.direction == "lower_is_better":
        score, extrapolated = _lower_score(
            value, thresholds.P, thresholds.E, thresholds.A, thresholds.L
        )
    else:
        raise ValueError(f"unknown direction: {thresholds.direction!r}")
    return NormalisedScore(
        raw_value=float(value),
        score=score,
        band=score_to_band(score),
        extrapolated=extrapolated,
    )
