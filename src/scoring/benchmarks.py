"""Benchmark loader and lookup.

Per docs/benchmarks/BENCHMARKS_GUIDE.md. Lookup key:
`(test_id, metric_id, gender)` — no age band in v1.

YAML files are loaded lazily and cached in-process. Schema is validated on
load (P/E/A/L ordering must match `direction`). Misses raise
`BenchmarkLookupError`.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Literal

import yaml

ROOT = Path(__file__).resolve().parents[2]
BENCHMARKS_DIR = ROOT / "benchmarks"
DOMAINS = ("physical", "technical")

Direction = Literal["higher_is_better", "lower_is_better"]
Gender = Literal["male", "female"]


class BenchmarkLookupError(KeyError):
    """No benchmark exists for the requested (test, metric, gender)."""


class BenchmarkSchemaError(ValueError):
    """A benchmark YAML failed schema validation."""


@dataclass(frozen=True)
class Benchmark:
    """One `(test_id, metric_id, gender)` entry."""

    P: float
    E: float
    A: float
    L: float
    direction: Direction
    unit: str
    test_id: str
    metric_id: str
    gender: Gender


@dataclass(frozen=True)
class AggregationWeights:
    """Test → subarea → area roll-up weights from `benchmarks/_aggregation.yaml`."""

    areas: dict[str, dict[str, float]] = field(default_factory=dict)
    subareas: dict[str, dict[str, float]] = field(default_factory=dict)


# --- internals -----------------------------------------------------------


def _gender_normalize(g: str) -> Gender:
    s = g.lower()
    if s in ("m", "male"):
        return "male"
    if s in ("f", "female"):
        return "female"
    raise BenchmarkLookupError(f"unsupported gender: {g!r}")


def _validate_thresholds(
    P: float, E: float, A: float, L: float, direction: Direction
) -> None:
    if direction == "higher_is_better" and not (P < E < A < L):
        raise BenchmarkSchemaError(
            f"higher_is_better requires P<E<A<L, got P={P} E={E} A={A} L={L}"
        )
    if direction == "lower_is_better" and not (P > E > A > L):
        raise BenchmarkSchemaError(
            f"lower_is_better requires P>E>A>L, got P={P} E={E} A={A} L={L}"
        )


@lru_cache(maxsize=None)
def _find_file(test_id: str) -> Path:
    for domain in DOMAINS:
        candidate = BENCHMARKS_DIR / domain / f"{test_id}.yaml"
        if candidate.exists():
            return candidate
    raise BenchmarkLookupError(f"no benchmark file for test_id {test_id!r}")


@lru_cache(maxsize=None)
def _load_file(path: Path) -> dict:
    return yaml.safe_load(path.read_text())


# --- public API ---------------------------------------------------------


def lookup(test_id: str, metric_id: str, gender: str) -> Benchmark:
    """Return the Benchmark entry for `(test_id, metric_id, gender)`."""
    g = _gender_normalize(gender)
    cfg = _load_file(_find_file(test_id))

    metrics = cfg.get("metrics", {})
    if metric_id not in metrics:
        raise BenchmarkLookupError(
            f"metric {metric_id!r} not in benchmarks/{test_id}.yaml"
        )
    m = metrics[metric_id]
    if g not in m:
        raise BenchmarkLookupError(
            f"no benchmark for ({test_id!r}, {metric_id!r}, {g!r})"
        )

    direction = m.get("direction")
    if direction not in ("higher_is_better", "lower_is_better"):
        raise BenchmarkSchemaError(
            f"unknown direction {direction!r} in {test_id}/{metric_id}"
        )
    thresholds = m[g]
    P, E, A, L = (
        float(thresholds["P"]),
        float(thresholds["E"]),
        float(thresholds["A"]),
        float(thresholds["L"]),
    )
    _validate_thresholds(P, E, A, L, direction)

    return Benchmark(
        P=P, E=E, A=A, L=L,
        direction=direction,
        unit=m.get("unit", ""),
        test_id=test_id, metric_id=metric_id, gender=g,
    )


def load_test_aggregation(test_id: str) -> tuple[str, dict[str, float]]:
    """Return `(method, weights)` for combining metric scores within a test.

    Defaults to `("mean", {})` if no aggregation block exists in the YAML.
    Validates that weights sum to 1.0 if `weighted_mean` is declared.
    """
    cfg = _load_file(_find_file(test_id))
    agg = cfg.get("aggregation") or {}
    method = agg.get("method", "mean")
    weights = agg.get("weights", {})
    if method == "weighted_mean":
        s = sum(weights.values())
        if abs(s - 1.0) > 1e-6:
            raise BenchmarkSchemaError(
                f"{test_id} weighted_mean weights sum to {s}, not 1.0"
            )
    return method, dict(weights)


@lru_cache(maxsize=None)
def load_aggregation() -> AggregationWeights:
    """Load and validate `benchmarks/_aggregation.yaml`."""
    cfg = yaml.safe_load((BENCHMARKS_DIR / "_aggregation.yaml").read_text())
    for area, subs in cfg["areas"].items():
        s = sum(subs.values())
        if abs(s - 1.0) > 1e-6:
            raise BenchmarkSchemaError(f"area {area!r} sums to {s}, not 1.0")
    for sub, tests in cfg["subareas"].items():
        s = sum(tests.values())
        if abs(s - 1.0) > 1e-6:
            raise BenchmarkSchemaError(f"subarea {sub!r} sums to {s}, not 1.0")
    return AggregationWeights(areas=cfg["areas"], subareas=cfg["subareas"])


def list_tests() -> list[str]:
    """Return all `test_id` strings with a benchmark file under `benchmarks/`."""
    out: list[str] = []
    for domain in DOMAINS:
        d = BENCHMARKS_DIR / domain
        if d.exists():
            out.extend(p.stem for p in d.glob("*.yaml"))
    return sorted(out)


def clear_cache() -> None:
    """Drop cached file paths + parses (useful for tests)."""
    _find_file.cache_clear()
    _load_file.cache_clear()
    load_aggregation.cache_clear()
