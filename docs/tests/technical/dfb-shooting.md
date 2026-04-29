# `dfb-shooting` — DFB Shooting Test

**Domain**: technical
**Family**: skill
**Status**: deferred (out of v1 scope — no sample video; ships in v1.1+)

## 1. Purpose

DFB-protocol shooting accuracy from defined positions and into target zones in the goal.

## 2. Equipment & setup

Goal with target-zone overlays (typically corners and centre). Marked shooting positions. Football.

## 3. Protocol

1. Athlete shoots from each position in sequence
2. Aims for declared target zones
3. Score per shot based on zone hit (zones have different point values per DFB rules)

## 4. CV pipeline requirements

| Capability | Required? | Notes |
|---|---|---|
| Player detection + tracking | yes | |
| Pose estimation | yes | |
| Ball detection + tracking | yes | |
| Cone detection | yes | |
| Calibration | mandatory (goal frame) | |
| Event detection | — | shot release, ball trajectory, goal-line crossing in target zone |

## 5. Metrics produced

| Metric ID | Module | Unit |
|---|---|---|
| `total_score` | `metrics/ball/total_score.py` | int (DFB rubric) |
| `passing_accuracy` | `metrics/ball/passing_accuracy.py` | % |
| `pass_velocity` | `metrics/ball/pass_velocity.py` | m/s (alias for shot velocity) |
| `zone_distribution` | `metrics/ball/zone_distribution.py` | % by zone |

## 6. Annotation requirements

- Skeleton: as appropriate to the test (see family default in `docs/annotation/VIDEO_ANNOTATION_SPEC.md`)
- Bounding box: yes
- Markers / gates / target zones: yes (test-specific elements above)
- HUD ticker fields: `current_position`, `running_score`, `last_shot_velocity`
- End-card: all metrics + test score

## 7. Benchmark schema

- File: `benchmarks/technical/dfb-shooting.yaml`
- Score direction: higher_is_better
- Aggregation: see benchmark file (`weighted_mean` default unless otherwise noted)

## 8. Failure modes

- Missing required entities (cones / ball / wall / box) → `DetectionError` or `CalibrationError`
- Protocol mismatches (duration, sequence) → `ProtocolError`
- Pose confidence below threshold on > 30% of frames → flag, do not abort

## 9. AI summary prompt notes

Use the family template `src/ai_summary/templates/skill.md`. See its
cross-metric narrative section for patterns to surface for this test.

## 10. Acceptance criteria

- [ ] All §5 metrics computed on a sample video
- [ ] All §6 overlays render
- [ ] Score normalisation produces sane 0–100 outputs
- [ ] Unit tests cover metric calculations
- [ ] Integration test passes end-to-end

## 11. References

DFB shooting protocol manual. Goal target zones declared in test config.
