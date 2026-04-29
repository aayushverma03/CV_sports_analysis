# `reaction-time` — Reaction Time Test

**Domain**: cognitive
**Family**: cognitive
**Status**: spec

## 1. Purpose

Simple stimulus-response latency. Measures pure reaction speed to a known stimulus.

## 2. Equipment & setup

Stimulus display (screen or LED). Athlete-side camera capturing response action (button press, foot tap, hand raise per protocol).

## 3. Protocol

1. Athlete in ready stance
2. Random delay (1–4 s), then stimulus
3. Athlete responds as fast as possible
4. Multiple trials (e.g. 10), mean and best recorded

## 4. CV pipeline requirements

| Capability | Required? | Notes |
|---|---|---|
| Player detection + tracking | yes | |
| Pose estimation | yes | |
| Ball detection + tracking | no | |
| Cone detection | no | |
| Calibration | n/a | |
| Event detection | — | stimulus onsets (timestamp from stimulus system), response actions detected via pose |

## 5. Metrics produced

| Metric ID | Module | Unit |
|---|---|---|
| `reaction_time` | `metrics/cognitive/reaction_time.py` | s (mean) |
| `reaction_time` | `metrics/cognitive/reaction_time.py` | s (best) |
| `response_accuracy` | `metrics/cognitive/response_accuracy.py` | % correct (for choice variants) |
| `trial_consistency` | `metrics/cognitive/trial_consistency.py` | % (1 - SD/mean) |

## 6. Annotation requirements

- Skeleton: as appropriate to the test (see family default in `docs/annotation/VIDEO_ANNOTATION_SPEC.md`)
- Bounding box: yes
- Markers / gates / target zones: yes (test-specific elements above)
- HUD ticker fields: `trial_number`, `last_reaction_time`, `running_mean`
- End-card: all metrics + test score

## 7. Benchmark schema

- File: `benchmarks/cognitive/reaction-time.yaml`
- Score direction: lower_is_better (time); higher_is_better (accuracy, consistency)
- Aggregation: see benchmark file (`weighted_mean` default unless otherwise noted)

## 8. Failure modes

- Missing required entities (cones / ball / wall / box) → `DetectionError` or `CalibrationError`
- Protocol mismatches (duration, sequence) → `ProtocolError`
- Pose confidence below threshold on > 30% of frames → flag, do not abort

## 9. AI summary prompt notes

Use the family template `src/ai_summary/templates/cognitive.md`. See its
cross-metric narrative section for patterns to surface for this test.

## 10. Acceptance criteria

- [ ] All §5 metrics computed on a sample video
- [ ] All §6 overlays render
- [ ] Score normalisation produces sane 0–100 outputs
- [ ] Unit tests cover metric calculations
- [ ] Integration test passes end-to-end

## 11. References

Stimulus and response timestamps must be aligned to the same clock. (Note: this test is **out of CV scope** and will ship later as an in-app game — see `docs/plan/plan.md`. No `cognitive_family` exists in the CV pipeline.)
