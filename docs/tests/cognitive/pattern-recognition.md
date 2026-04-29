# `pattern-recognition` — Pattern Recognition Test

**Domain**: cognitive
**Family**: cognitive
**Status**: spec

## 1. Purpose

Athlete identifies a pattern from a brief visual stimulus. Measures perceptual speed and pattern-matching accuracy.

## 2. Equipment & setup

Stimulus display (tactical board / image / short video clip). Response captured by selection input or verbal call detected by the operator.

## 3. Protocol

1. Stimulus shown for fixed duration (e.g. 2 s)
2. Athlete selects from response options
3. Response time and correctness recorded
4. Multiple trials with varying difficulty

## 4. CV pipeline requirements

| Capability | Required? | Notes |
|---|---|---|
| Player detection + tracking | yes | |
| Pose estimation | no | |
| Ball detection + tracking | no | |
| Cone detection | no | |
| Calibration | n/a | |
| Event detection | — | stimulus onset, stimulus offset, response events |

## 5. Metrics produced

| Metric ID | Module | Unit |
|---|---|---|
| `response_accuracy` | `metrics/cognitive/response_accuracy.py` | % correct |
| `decision_latency` | `metrics/cognitive/decision_latency.py` | s (mean) |
| `difficulty_progression` | `metrics/cognitive/difficulty_progression.py` | ratio of accuracy at hardest vs easiest |

## 6. Annotation requirements

- Skeleton: as appropriate to the test (see family default in `docs/annotation/VIDEO_ANNOTATION_SPEC.md`)
- Bounding box: yes
- Markers / gates / target zones: yes (test-specific elements above)
- HUD ticker fields: `trial_number`, `last_decision_time`, `accuracy_running`
- End-card: all metrics + test score

## 7. Benchmark schema

- File: `benchmarks/cognitive/pattern-recognition.yaml`
- Score direction: higher_is_better (accuracy); lower_is_better (latency)
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

No motion tracking required for the metric itself, only response capture. AI summary should describe perceptual speed, not 'intelligence'.
