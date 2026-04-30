# Score Normalisation (0–100)

The contract between the metric library and the report layer: every raw
metric value becomes a number in `[1, 100]` keyed only on `(test_id,
metric_id, gender)`. **Age band is not a lookup dimension in v1.**

## Performance bands

| Band | Score range |
|---|---|
| **Poor** | 1–40 |
| **Expected** | 40–70 |
| **Above Expected** | 70–90 |
| **Elite** | 90–100 |

A score of 100 is the cap — the elite anchor `L` and beyond all map to 100.

## Benchmark thresholds

Each `(test_id, metric_id, gender)` benchmark provides four thresholds:

- **P** — Poor threshold (boundary between Poor and Expected)
- **E** — Expected threshold (boundary between Expected and Above)
- **A** — Above-Expected threshold (boundary between Above and Elite)
- **L** — Elite threshold (anything past L is capped Elite at 100)

Direction-dependent ordering:

- `higher_is_better`: `P < E < A < L`
- `lower_is_better`: `P > E > A > L`

## Algorithms

### Mode A — `higher_is_better`

Larger value = better. Used for: jump height, throw distance, max speed,
total taps, total successful jumps, etc.

```
F = max(0, P − (E − P))            # extrapolated lower bound

if value <= F:        score = 1
elif value <= P:      score = 1  + 39 * (value − F) / (P − F)        # 1..40
elif value <= E:      score = 40 + 30 * (value − P) / (E − P)        # 40..70
elif value <= A:      score = 70 + 20 * (value − E) / (A − E)        # 70..90
elif value <= L:      score = 90 + 10 * (value − A) / (L − A)        # 90..100
else:                 score = 100
```

`F` clamps to 0 so we never extrapolate below zero. Boundaries on each band
match exactly: at `value = P` the formula gives `40` from both adjacent
bands, etc.

### Mode B — `lower_is_better`

Smaller value = better. Used for: sprint times, completion times, fatigue
percentages, decision time, ball–foot distance.

```
C = P + (P − E)                    # extrapolated upper bound

if value >= C:        score = 1
elif value >= P:      score = 40 − 39 * (value − P) / (C − P)        # 1..40
elif value >= E:      score = 40 + 30 * (P − value) / (P − E)        # 40..70
elif value >= A:      score = 70 + 20 * (E − value) / (E − A)        # 70..90
elif value >= L:      score = 90 + 10 * (A − value) / (A − L)        # 90..100
else:                 score = 100
```

Symmetric to Mode A. `C` is the mirror of `F`: a hard floor for the Poor
band.

## Worked example — `lower_is_better`, Linear Sprint 10m, Male

Thresholds: `P=2.4, E=2.1, A=1.8, L=1.6` (seconds).

`C = 2.4 + (2.4 − 2.1) = 2.7`

| Raw time | Band | Score | Why |
|---|---|---|---|
| 2.85 s | Poor | 1 | beyond C |
| 2.5 s | Poor | `40 − 39 × (2.5 − 2.4) / (2.7 − 2.4)` = `40 − 13` = `27` | inside Poor zone |
| 2.4 s | Poor → Expected boundary | 40 | exactly at P |
| 2.25 s | Expected | `40 + 30 × (2.4 − 2.25) / (2.4 − 2.1)` = `55` | mid-Expected |
| 1.95 s | Above | `70 + 20 × (2.1 − 1.95) / (2.1 − 1.8)` = `80` | mid-Above |
| 1.7 s | Elite | `90 + 10 × (1.8 − 1.7) / (1.8 − 1.6)` = `95` | mid-Elite |
| 1.55 s | Elite (capped) | 100 | beyond L |

## Worked example — `higher_is_better`, CMJ jump height, Male

Thresholds: `P=50, E=60, A=70, L=75` (cm).

`F = max(0, 50 − (60 − 50)) = max(0, 40) = 40`

| Raw height | Band | Score |
|---|---|---|
| 35 cm | Poor (cap) | 1 |
| 45 cm | Poor | `1 + 39 × (45 − 40) / (50 − 40)` = `20.5` |
| 50 cm | Poor → Expected | 40 |
| 55 cm | Expected | `40 + 30 × 5/10` = `55` |
| 65 cm | Above | `70 + 20 × 5/10` = `80` |
| 72 cm | Elite | `90 + 10 × 2/5` = `94` |
| 80 cm | Elite (capped) | 100 |

## API

```python
def normalise(
    raw_value: float,
    benchmark: Benchmark,         # parsed entry from benchmarks/<domain>/<test_id>.yaml
    direction: Direction,         # 'higher_is_better' | 'lower_is_better'
) -> NormalisedScore: ...

@dataclass(frozen=True)
class Benchmark:
    P: float
    E: float
    A: float
    L: float

@dataclass(frozen=True)
class NormalisedScore:
    raw_value: float
    raw_unit: str
    score: float           # 1.0..100.0
    band: Literal["poor", "expected", "above_expected", "elite"]
    extrapolated: bool     # True if value falls outside [F, C] (i.e. score = 1 or 100)
```

## Test-level aggregation

A test produces multiple metric scores. They aggregate via the rule in the
benchmark YAML's `aggregation` block — typically `weighted_mean` with
metric-specific weights, or simple `mean` if every metric matters equally.
Default: `mean`.

## Subarea and Area aggregation

Test scores roll up into subarea scores via fixed weights, then subareas
roll up into the area (Foundation / Technical) score. The full weight tree
lives in `docs/benchmarks/BENCHMARKS_GUIDE.md` §"Aggregation weights".

## Other modes (not in v1)

- **`target_value`** — score peaks at a target, falls off in either direction
  (e.g. trunk lean of 5° is ideal, ±5° tolerance). Required when we add tests
  that benchmark optimal posture / asymmetry. Deferred until first metric
  needs it.
- **`banded`** — fixed letter cutoffs (e.g. LESS rubric: 0–3 = excellent, 4–5
  = good, ...). May be useful for the LESS subset score; deferred and decided
  in Phase 4 LESS implementation.

## Edge cases

- **Missing benchmark for `(test_id, metric_id, gender)`**: raise
  `BenchmarkLookupError`. No fabrication, no fallback to opposite gender.
- **Single attempt vs multiple attempts**: spec-driven. Per
  `METRICS_CATALOG.md`, jump tests use one attempt → `jump_height_cm`
  (no `best_*` aggregation needed).
- **Pose-driven metrics with low confidence**: if `pose_confidence_low_pct >
  30`, flag `low_confidence: true` on the score and let the AI summary
  soften its language.

## Implementation

- `src/scoring/normalization.py` — implement `normalise()` for both directions
- `src/scoring/grade.py` — band mapping (`score_to_band(score) -> str`)
- `src/scoring/benchmarks.py` — YAML loader and lookup
- `tests/unit/scoring/` — exhaustive parametrised tests covering boundary
  values, both directions, all five bands
