# `incremental-ramp` — Incremental Ramp Test

**Domain**: physical
**Family**: endurance
**Status**: deferred (out of v1 scope — no sample video; ships in v1.1+)

## 1. Purpose

Continuous incremental running pace until volitional exhaustion. Lab-style test executed on a track.

## 2. Equipment & setup

Track or treadmill (treadmill variant out of scope for video — track only). Audio pace cues.

## 3. Protocol

1. Start at low jogging pace
2. Pace increases continuously (e.g. +0.5 km/h every minute)
3. Continues until athlete cannot maintain pace

## 4. CV pipeline requirements

| Capability | Required? | Notes |
|---|---|---|
| Player detection + tracking | yes | |
| Pose estimation | no | |
| Ball detection + tracking | no | |
| Cone detection | yes | |
| Calibration | mandatory | |
| Event detection | — | start, pace-increase boundaries (audio), failure |

## 5. Metrics produced

| Metric ID | Module | Unit |
|---|---|---|
| `max_aerobic_velocity` | `metrics/endurance/max_aerobic_velocity.py` | km/h |
| `total_distance_completed` | `metrics/endurance/total_distance_completed.py` | m |
| `total_time` | `metrics/motion/total_time.py` | s |
| `pacing_variability` | `metrics/endurance/pacing_variability.py` | % |

## 6. Annotation requirements

- Skeleton: as appropriate to the test (see family default in `docs/annotation/VIDEO_ANNOTATION_SPEC.md`)
- Bounding box: yes
- Markers / gates / target zones: yes (test-specific elements above)
- HUD ticker fields: `current_pace_target`, `compliance`, `elapsed_time`
- End-card: all metrics + test score

## 7. Benchmark schema

- File: `benchmarks/physical/incremental-ramp.yaml`
- Score direction: higher_is_better
- Aggregation: see benchmark file (`weighted_mean` default unless otherwise noted)

## 8. Failure modes

- Missing required entities (cones / ball / wall / box) → `DetectionError` or `CalibrationError`
- Protocol mismatches (duration, sequence) → `ProtocolError`
- Pose confidence below threshold on > 30% of frames → flag, do not abort

## 9. AI summary prompt notes

Use the family template `src/ai_summary/templates/endurance.md`. See its
cross-metric narrative section for patterns to surface for this test.

## 10. Acceptance criteria

- [ ] All §5 metrics computed on a sample video
- [ ] All §6 overlays render
- [ ] Score normalisation produces sane 0–100 outputs
- [ ] Unit tests cover metric calculations
- [ ] Integration test passes end-to-end

## 11. References

Lab-derived; field implementation here. Pace compliance is detection-driven.
