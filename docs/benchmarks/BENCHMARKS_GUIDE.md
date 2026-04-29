# Benchmarks Guide

Benchmarks are reference distributions of expected performance values, broken
down by gender and age band, used to convert raw metric values into 0–100
scores. They live as YAML files under `benchmarks/`.

## File layout

```
benchmarks/
├── physical/
│   ├── linear-sprint.yaml
│   ├── counter-movement-jump.yaml
│   └── ...
├── technical/
│   └── ...
└── cognitive/
    └── ...
```

One file per `test_id`. The filename matches the test ID exactly.

## Schema

```yaml
test_id: linear-sprint
display_name: "Linear Sprint (10/20/30 m)"
score_direction: lower_is_better      # lower_is_better | higher_is_better | target_value
units:
  total_completion_time: s
  split_10m: s
  split_20m: s
  split_30m: s
  max_speed: m/s

# Optional aggregation rule that combines per-metric scores
# into a single test-level score. If omitted, default = mean of metric scores.
aggregation:
  method: weighted_mean
  weights:
    total_completion_time: 0.4
    max_speed: 0.3
    split_10m: 0.15
    split_20m: 0.15

age_bands:
  - U12
  - U14
  - U16
  - U18
  - U23
  - Senior

# The actual norms.
# Each entry: gender × age_band × metric_id → distribution stats.
benchmarks:
  M:
    U12:
      total_completion_time: { p10: 5.40, p50: 5.10, p90: 4.75 }
      split_10m:              { p10: 2.05, p50: 1.95, p90: 1.85 }
      split_20m:              { p10: 3.55, p50: 3.40, p90: 3.20 }
      split_30m:              { p10: 5.40, p50: 5.10, p90: 4.75 }
      max_speed:              { p10: 5.8,  p50: 6.4,  p90: 7.0 }
    # ... U14, U16, U18, U23, Senior
  F:
    U12:
      total_completion_time: { p10: 5.80, p50: 5.45, p90: 5.10 }
      # ...
```

## Distribution representation

Three percentiles is the minimum viable spec: **p10, p50 (median), p90**. The
normaliser interpolates between these to produce a 0–100 score.

If you have richer data, you can replace the `{p10, p50, p90}` block with:

```yaml
total_completion_time:
  mean: 5.10
  std: 0.18
  source: "Internal cohort 2024-2025 (N=420)"
```

The normaliser handles both forms — see `docs/scoring/NORMALIZATION.md`.

## Gender

Use ISO codes: `M`, `F`. Add `X` (non-binary / unspecified) only if you have
evidence-based norms; do not invent them by averaging M and F.

## Age bands

Default bands are youth-football oriented (U12, U14, U16, U18, U23, Senior).
Override per test if the protocol's source uses different bands (e.g. Cooper
test has its own age-decade bands — document in the test spec).

If an athlete falls outside all bands, the scorer falls back to the closest
band and emits a `WARNING` with a `band_extrapolated: true` flag in the
result JSON.

## Score direction modes

- **`lower_is_better`** — sprint times, completion times, reaction times. p10 (slowest 10%) → score ~10; p90 (fastest 10%) → score ~90.
- **`higher_is_better`** — jump heights, distances, throw distances, max speed. p10 (worst 10%) → score ~10; p90 (best 10%) → score ~90.
- **`target_value`** — for tests where deviation in either direction is bad (e.g. trunk lean angle in a specific range). Specify a `target` field and a `tolerance`.

```yaml
score_direction: target_value
target: 0          # zero asymmetry is the goal
tolerance: 15      # full score within ±15%, linearly degrading to ±30%
```

## Sourcing data

Every benchmark file should declare its data source:

```yaml
metadata:
  source: "DFB U17–U19 testing battery, 2022 cohort"
  sample_size: 1240
  confidence: high   # high | medium | low | provisional
  last_updated: 2025-09-01
```

Provisional benchmarks are usable but the AI summary should not give them
strong narrative weight — see `docs/ai_summary/AI_SUMMARY_SPEC.md`.

## Lookups (what the scoring layer does)

Given `(test_id, metric_id, gender, age_at_test)`:

1. Open `benchmarks/<domain>/<test_id>.yaml`
2. Resolve age to an age band (use the test file's `age_bands`)
3. Index `benchmarks[gender][age_band][metric_id]`
4. Pass distribution + score direction to `src/scoring/normalization.py`

Implementation: `src/scoring/benchmarks.py`.

## Worked example

See `benchmarks/physical/linear-sprint.yaml` — that file is the reference
implementation of this schema and should be kept in lockstep with this guide.
