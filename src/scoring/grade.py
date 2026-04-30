"""Band display + score roll-up (metric → test → subarea → area).

Pure logic. Aggregation weights come from `src.scoring.benchmarks`.
Partial test sets re-normalise: if only some tests in a subarea ran, weights
of the missing tests are dropped and remaining weights re-scaled to 1.0 so
the score remains comparable across athletes who didn't take every test.
"""
from __future__ import annotations

import math

from src.scoring.benchmarks import load_aggregation
from src.scoring.normalization import Band

# --- Display -------------------------------------------------------------

_BAND_DISPLAY: dict[Band, str] = {
    "poor": "Poor",
    "expected": "Expected",
    "above_expected": "Above Expected",
    "elite": "Elite",
}

# Hex strings (RGB), aligned with the palette in VIDEO_ANNOTATION_SPEC.md.
_BAND_COLOUR_HEX: dict[Band, str] = {
    "poor": "#E76F51",            # warm red — alert
    "expected": "#F4A261",         # warm orange — neutral
    "above_expected": "#2A9D8F",   # teal — positive
    "elite": "#264653",            # dark teal — top
}


def format_band(band: Band) -> str:
    """Human-readable band label."""
    return _BAND_DISPLAY[band]


def band_colour_hex(band: Band) -> str:
    """Hex RGB colour for the band, for UI / annotation use."""
    return _BAND_COLOUR_HEX[band]


# --- Aggregation ---------------------------------------------------------


def aggregate_metric_scores(
    scores: dict[str, float],
    method: str = "mean",
    weights: dict[str, float] | None = None,
) -> float:
    """Combine metric scores within one test.

    Parameters
    ----------
    scores : dict[metric_id, score]
        Per-metric scores in [1, 100]. Pass only metrics actually computed
        (drop missing ones — partial reads are normal).
    method : 'mean' | 'weighted_mean'
        From the test's benchmark YAML.
    weights : dict[metric_id, weight], optional
        Required for `weighted_mean`. Weights for missing metrics are
        dropped and remaining weights re-normalised to sum to 1.

    Returns
    -------
    float
        Aggregated test score, NaN if `scores` is empty.
    """
    if not scores:
        return math.nan
    if method == "mean":
        return sum(scores.values()) / len(scores)
    if method == "weighted_mean":
        if not weights:
            raise ValueError("weighted_mean requires `weights`")
        relevant = {k: w for k, w in weights.items() if k in scores}
        if not relevant:
            return math.nan
        total_w = sum(relevant.values())
        return sum(scores[k] * w for k, w in relevant.items()) / total_w
    raise ValueError(f"unknown aggregation method: {method!r}")


def aggregate_tests_to_subarea(
    test_scores: dict[str, float],
    subarea: str,
) -> float:
    """Combine test scores into a subarea score using `_aggregation.yaml` weights.

    Tests not in `test_scores` are dropped from the average; remaining
    weights re-scale.
    """
    if not test_scores:
        return math.nan
    weights = load_aggregation().subareas[subarea]
    relevant = {tid: w for tid, w in weights.items() if tid in test_scores}
    if not relevant:
        return math.nan
    total_w = sum(relevant.values())
    return sum(test_scores[tid] * w for tid, w in relevant.items()) / total_w


def aggregate_subareas_to_area(
    subarea_scores: dict[str, float],
    area: str,
) -> float:
    """Combine subarea scores into the parent area score."""
    if not subarea_scores:
        return math.nan
    weights = load_aggregation().areas[area]
    relevant = {sub: w for sub, w in weights.items() if sub in subarea_scores}
    if not relevant:
        return math.nan
    total_w = sum(relevant.values())
    return sum(subarea_scores[sub] * w for sub, w in relevant.items()) / total_w
