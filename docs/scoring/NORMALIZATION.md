# Score Normalisation (0–100)

This document specifies how raw metric values become 0–100 scores. It is the
contract between the metric library and the report layer.

## Goal

Every metric, regardless of unit or direction, must end up as a number in
`[0, 100]` where:

- `0–20` = well below typical for the gender/age band
- `40–60` = around the median
- `80–100` = elite for the gender/age band
- `100` = at or beyond the elite anchor (does not extrapolate further)

This makes radar charts, summary cards, and AI summary inputs comparable
across metrics that have wildly different units.

## Inputs

```python
def normalise(
    raw_value: float,
    distribution: Distribution,        # parsed benchmark entry
    score_direction: ScoreDirection,   # 'lower_is_better' | 'higher_is_better' | 'target_value'
) -> NormalisedScore: ...
```

Where `Distribution` is one of:

```python
class PercentileDist:
    p10: float
    p50: float
    p90: float

class GaussianDist:
    mean: float
    std: float
```

## Algorithms

### Mode A — `lower_is_better` with percentiles

Sprint times, completion times, reaction times.

```
if raw <= p90:  score = 90 + (p90 - raw) / (p90 - p_elite_anchor) * 10
                # cap at 100, where p_elite_anchor = p90 - (p50 - p90) (i.e. mirror of p50→p90 distance)
elif raw <= p50: score = 50 + (p50 - raw) / (p50 - p90) * 40
elif raw <= p10: score = 10 + (p10 - raw) / (p10 - p50) * 40
else:           score = max(0, 10 - (raw - p10) / (p10 - p50) * 10)
```

### Mode B — `higher_is_better` with percentiles

Jump heights, distances, max speed, throw distance. Mirror image of Mode A.

```
if raw >= p90:  score = 90 + (raw - p90) / (p_elite_anchor - p90) * 10   # cap at 100
elif raw >= p50: score = 50 + (raw - p50) / (p90 - p50) * 40
elif raw >= p10: score = 10 + (raw - p10) / (p50 - p10) * 40
else:            score = max(0, 10 - (p10 - raw) / (p50 - p10) * 10)
```

### Mode C — Gaussian distributions

If the benchmark provides `mean + std`, convert to a z-score then map.

```
z = (raw - mean) / std
# Flip sign for lower_is_better
if direction == 'lower_is_better':
    z = -z
# Map z ∈ [-2, +2] → score ∈ [10, 90] linearly; clamp outside
score = clamp(50 + 20 * z, 0, 100)
```

This is monotonic and gives reasonable elite-tail behaviour at z ≈ +2.5 → 100.

### Mode D — `target_value`

For metrics where deviation in either direction is bad (e.g. trunk lean,
asymmetry%).

```
delta = abs(raw - target)
if delta <= tolerance:        score = 100 - (delta / tolerance) * 20    # 80..100
elif delta <= 2 * tolerance:  score = 80 - ((delta - tolerance) / tolerance) * 60  # 20..80
else:                          score = max(0, 20 - (delta - 2*tolerance) * k)
```

## Test-level aggregation

A single test produces multiple metric scores. Aggregate to one test score
using the rule from the benchmark file's `aggregation:` block.

- **`mean`** (default) — straight arithmetic mean
- **`weighted_mean`** — uses provided weights (must sum to 1.0; validate)
- **`min`** — for tests where weakest link defines performance (rarely used)
- **`max`** — for tests where best attempt is reported

If aggregation is not specified, default to `mean`.

## Output schema

```python
class NormalisedScore:
    raw_value: float
    raw_unit: str
    score: float                    # 0–100
    band: str                       # 'elite' | 'above_average' | 'average' | 'below_average' | 'poor'
    percentile_estimate: float      # 0–100, derived from inverse of distribution
    extrapolated: bool              # True if value falls outside p10..p90 range
    benchmark_confidence: str       # echoes 'high' | 'medium' | 'low' | 'provisional'
```

Bands:
- `elite` ≥ 85
- `above_average` 65–84
- `average` 35–64
- `below_average` 15–34
- `poor` < 15

## Edge cases

- **Missing benchmark for `(gender, age band)`**: try the nearest age band;
  set `extrapolated=True`. If still missing, raise `BenchmarkLookupError` —
  do not fabricate.
- **Single attempt vs multiple attempts**: if the test produces multiple
  attempts, score the *best* attempt for `higher_is_better` metrics, the
  *best (lowest)* for `lower_is_better`, unless the test spec specifies
  otherwise (e.g. RSA scores the *mean* over sprints).
- **Pose-driven metrics with low confidence**: if more than 30% of frames in
  the metric's window were below the pose confidence threshold, flag
  `low_confidence: true` in the score and let the AI summary call it out.

## Implementation

`src/scoring/normalization.py` — implement all four modes.
`src/scoring/grade.py` — band mapping.
`tests/unit/scoring/` — exhaustive parameterised tests.
