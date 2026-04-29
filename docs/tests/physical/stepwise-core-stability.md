# `stepwise-core-stability` — Stepwise Core Stability

**Domain**: physical
**Family**: mobility
**Status**: deferred (out of v1 scope — no sample video; ships in v1.1+)

## 1. Purpose

Graded core endurance test with progressive postural challenges. Continues until form breaks down.

## 2. Equipment & setup

Mat. Side-on camera capturing trunk, hip, and limb keypoints.

## 3. Protocol

1. Athlete progresses through a defined sequence of plank-style holds with increasing difficulty
2. Each stage held for a target duration (e.g. 15 s)
3. Test ends when athlete cannot maintain target posture (deviation beyond tolerance)

## 4. CV pipeline requirements

| Capability | Required? | Notes |
|---|---|---|
| Player detection + tracking | yes | |
| Pose estimation | yes | |
| Ball detection + tracking | no | |
| Cone detection | no | |
| Calibration | optional | |
| Event detection | — | stage transitions, posture deviation events, fail event |

## 5. Metrics produced

| Metric ID | Module | Unit |
|---|---|---|
| `stage_reached` | `metrics/endurance/stage_reached.py` | stage |
| `total_hold_duration` | `metrics/biomech/total_hold_duration.py` | s |
| `posture_compliance` | `metrics/biomech/posture_compliance.py` | % time within tolerance |
| `deviation_count` | `metrics/biomech/deviation_count.py` | int |

## 6. Annotation requirements

- Skeleton: as appropriate to the test (see family default in `docs/annotation/VIDEO_ANNOTATION_SPEC.md`)
- Bounding box: yes
- Markers / gates / target zones: yes (test-specific elements above)
- HUD ticker fields: `current_stage`, `hold_time_remaining`, `deviation_warning`
- End-card: all metrics + test score

## 7. Benchmark schema

- File: `benchmarks/physical/stepwise-core-stability.yaml`
- Score direction: higher_is_better (stage); higher_is_better (compliance)
- Aggregation: see benchmark file (`weighted_mean` default unless otherwise noted)

## 8. Failure modes

- Missing required entities (cones / ball / wall / box) → `DetectionError` or `CalibrationError`
- Protocol mismatches (duration, sequence) → `ProtocolError`
- Pose confidence below threshold on > 30% of frames → flag, do not abort

## 9. AI summary prompt notes

Use the family template `src/ai_summary/templates/mobility.md`. See its
cross-metric narrative section for patterns to surface for this test.

## 10. Acceptance criteria

- [ ] All §5 metrics computed on a sample video
- [ ] All §6 overlays render
- [ ] Score normalisation produces sane 0–100 outputs
- [ ] Unit tests cover metric calculations
- [ ] Integration test passes end-to-end

## 11. References

McGill core endurance principles, adapted to stepwise format. Tolerance bands per stage live in the test config.
