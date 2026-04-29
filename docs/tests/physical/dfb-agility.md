# `dfb-agility` — DFB Agility Test

**Domain**: physical
**Family**: agility
**Status**: deferred (out of v1 scope — no sample video; ships in v1.1+)

## 1. Purpose

DFB (German FA) standardised agility course. Specific cone layout per DFB testing manual.

## 2. Equipment & setup

Cone layout per DFB protocol — see `configs/dfb_agility.yaml` (to be authored from federation manual).

## 3. Protocol

1. Athlete completes the DFB-defined course
2. Specific COD pattern: 90° and 180° turns
3. Time recorded from start gate to finish

## 4. CV pipeline requirements

| Capability | Required? | Notes |
|---|---|---|
| Player detection + tracking | yes | |
| Pose estimation | yes | |
| Ball detection + tracking | no | |
| Cone detection | yes | |
| Calibration | mandatory | |
| Event detection | — | start, each defined waypoint, finish |

## 5. Metrics produced

| Metric ID | Module | Unit |
|---|---|---|
| `total_completion_time` | `metrics/motion/total_completion_time.py` | s |
| `split_segment_times` | `metrics/motion/split_segment_times.py` | s |
| `max_speed` | `metrics/motion/max_speed.py` | m/s |
| `peak_acceleration` | `metrics/motion/peak_acceleration.py` | m/s² |
| `peak_deceleration` | `metrics/motion/peak_deceleration.py` | m/s² |

## 6. Annotation requirements

- Skeleton: as appropriate to the test (see family default in `docs/annotation/VIDEO_ANNOTATION_SPEC.md`)
- Bounding box: yes
- Markers / gates / target zones: yes (test-specific elements above)
- HUD ticker fields: `elapsed_time`, `current_segment`
- End-card: all metrics + test score

## 7. Benchmark schema

- File: `benchmarks/physical/dfb-agility.yaml`
- Score direction: lower_is_better
- Aggregation: see benchmark file (`weighted_mean` default unless otherwise noted)

## 8. Failure modes

- Missing required entities (cones / ball / wall / box) → `DetectionError` or `CalibrationError`
- Protocol mismatches (duration, sequence) → `ProtocolError`
- Pose confidence below threshold on > 30% of frames → flag, do not abort

## 9. AI summary prompt notes

Use the family template `src/ai_summary/templates/agility.md`. See its
cross-metric narrative section for patterns to surface for this test.

## 10. Acceptance criteria

- [ ] All §5 metrics computed on a sample video
- [ ] All §6 overlays render
- [ ] Score normalisation produces sane 0–100 outputs
- [ ] Unit tests cover metric calculations
- [ ] Integration test passes end-to-end

## 11. References

DFB Talentförderprogramm testing manual. Layout file required before implementation.
