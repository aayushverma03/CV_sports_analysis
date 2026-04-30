# Benchmarks Guide

Benchmarks are the reference values used to convert raw metric outputs into
0–100 scores via `docs/scoring/NORMALIZATION.md`. They live as YAML files
under `benchmarks/`.

## Lookup key

Three dimensions: `(test_id, metric_id, gender)`. **Age band is not part of
the v1 lookup** — see the locked decision in `docs/plan/plan.md` §2.

```
benchmarks/
├── physical/
│   ├── linear-sprint.yaml
│   ├── counter-movement-jump.yaml
│   └── ...
└── technical/
    ├── figure-of-8-dribbling.yaml
    └── ...
```

One file per `test_id`; the filename matches the test ID exactly (kebab-case).

## Schema

```yaml
test_id: linear-sprint
display_name: "Linear Sprint (10/20/30/40 m)"

# Optional aggregation rule that combines per-metric scores into a single
# test-level score. If omitted, defaults to `mean` over all metrics.
aggregation:
  method: weighted_mean
  weights:
    time_10m_s: 0.25
    time_20m_s: 0.25
    time_30m_s: 0.25
    time_40m_s: 0.25

metrics:
  time_10m_s:
    direction: lower_is_better
    unit: s
    male:    { P: 2.4, E: 2.1, A: 1.8, L: 1.6 }
    female:  { P: 2.6, E: 2.3, A: 2.0, L: 1.8 }
  time_20m_s:
    direction: lower_is_better
    unit: s
    male:    { P: 4.1, E: 3.6, A: 3.1, L: 2.7 }
    female:  { P: 4.4, E: 3.9, A: 3.4, L: 3.0 }
  # ...

metadata:
  source: "User-provided benchmarks (DFB U17–U19 testing battery, 2022)"
  confidence: high          # high | medium | low | provisional
  last_updated: 2026-04-30
```

## Threshold semantics

Per metric, per gender, four numbers:

- **P** — Poor threshold (boundary between Poor and Expected bands)
- **E** — Expected threshold
- **A** — Above-Expected threshold
- **L** — Elite threshold

Direction-dependent ordering — must be enforced when YAML is loaded:

- `higher_is_better`: `P < E < A < L`
- `lower_is_better`: `P > E > A > L`

A loader that finds violations raises `BenchmarkSchemaError`. A scorer that
gets a missing `(test, metric, gender)` triple raises `BenchmarkLookupError`.

## Gender

Use lowercase keys: `male`, `female`. Add `unspecified` only if you have
evidence-based norms for that group; do not invent them by averaging the
binary genders. If an athlete profile is `unspecified` and no benchmark
exists, raise `BenchmarkLookupError`.

## Aggregation: from metric to test, subarea, area

Three rollup levels:

1. **Metric scores → test score** — declared in each benchmark file's
   `aggregation:` block (default: `mean`).
2. **Test scores → subarea score** — declared once, in
   `benchmarks/_aggregation.yaml`.
3. **Subarea scores → area score** — same file.

### Aggregation weights (locked v1)

```yaml
# benchmarks/_aggregation.yaml

areas:
  foundation:                      # Physical Capabilities
    speed_agility:        0.35
    strength_power:       0.30
    endurance:            0.20
    mobility_stability:   0.15
  technical:
    dribbling:            0.45
    passing:              0.35
    ball_control:         0.20

subareas:
  speed_agility:
    linear-sprint:                   0.20
    5x10m-sprint-cod:                0.18
    repeated-sprint-ability:         0.14
    illinois-agility:                0.12
    t-test:                          0.12
    bangsbo-sprint:                  0.12
    45-second-agility-hurdle-jump:   0.07     # replaces hurdle-agility-run
    foot-tapping:                    0.05
  strength_power:
    counter-movement-jump:           0.30
    drop-jump:                       0.22
    squat-jump:                      0.18
    standing-long-jump:              0.18
    medicine-ball-throw:             0.12
  endurance:
    yo-yo-intermittent:              0.60
    multistage-fitness:              0.40
  mobility_stability:
    landing-error-scoring-system:    1.00
  dribbling:
    zig-zag-dribbling:               0.40
    figure-of-8-dribbling:           0.35
    straight-line-dribbling:         0.25
  passing:
    wall-pass:                       1.00
  ball_control:
    juggling:                        1.00
```

Each subarea's weights must sum to 1.0; same for areas. The loader validates
this on startup.

## Sourcing data

Every benchmark file declares its source:

```yaml
metadata:
  source: "User-provided thresholds, soccer-specific U17–U23 norms"
  confidence: high          # high | medium | low | provisional
  last_updated: 2026-04-30
```

Confidence levels:

- `high` — published academic norms or large internal cohort
- `medium` — coaching-association norms, smaller cohort, or extrapolated within
  one age band
- `low` — single-source heuristic
- `provisional` — first-pass values, expected to be refined; the AI summary
  softens narrative weight ("preliminary indication") when the benchmark is
  provisional

## Lookups (what the scoring layer does)

Given `(test_id, metric_id, gender, raw_value)`:

1. Open `benchmarks/<domain>/<test_id>.yaml`
2. Look up `metrics[metric_id][gender]` → `{P, E, A, L}`
3. Look up `metrics[metric_id][direction]` → `lower_is_better` / `higher_is_better`
4. Pass to `normalise()` per `docs/scoring/NORMALIZATION.md`

Implementation: `src/scoring/benchmarks.py`.

## Worked example

See `benchmarks/physical/linear-sprint.yaml` — that file is the reference
implementation of this schema and is kept in lockstep with this guide.

## Adding a new benchmark

1. Confirm the metric exists in `docs/metrics/METRICS_CATALOG.md` with the
   canonical `metric_id` you'll reference here.
2. Create or extend `benchmarks/<domain>/<test_id>.yaml` with the four
   thresholds per gender.
3. Add an aggregation entry to `_aggregation.yaml` if this is the first
   test in its subarea.
4. Run `tests/unit/scoring/` to confirm the loader validates correctly.
