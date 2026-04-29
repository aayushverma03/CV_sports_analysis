# `cooper` — Cooper 12-Minute Run

**Domain**: physical
**Family**: endurance
**Status**: deferred (out of v1 scope — no sample video; ships in v1.1+)

## 1. Purpose

Maximal distance covered in 12 minutes of continuous running. VO₂max estimator.

## 2. Equipment & setup

Standard 400 m track (or marked closed course). Camera not strictly required if external GPS used; for video-only mode, a fixed wide-angle camera covering the lap.

## 3. Protocol

1. Athlete runs continuously for 12 minutes
2. Maximal pace, no walking unless necessary
3. Distance covered at the 12-min mark is recorded

## 4. CV pipeline requirements

| Capability | Required? | Notes |
|---|---|---|
| Player detection + tracking | yes | |
| Pose estimation | no | |
| Ball detection + tracking | no | |
| Cone detection | yes | |
| Calibration | mandatory (track marks) | |
| Event detection | — | start, lap crossings, 12:00 minute mark |

## 5. Metrics produced

| Metric ID | Module | Unit |
|---|---|---|
| `total_distance_completed` | `metrics/endurance/total_distance_completed.py` | m |
| `average_speed` | `metrics/motion/average_speed.py` | m/s |
| `vo2_max_estimate` | `metrics/endurance/vo2_max_estimate.py` | ml·kg⁻¹·min⁻¹ |
| `pacing_variability` | `metrics/endurance/pacing_variability.py` | % lap-time SD |

## 6. Annotation requirements

- Skeleton: as appropriate to the test (see family default in `docs/annotation/VIDEO_ANNOTATION_SPEC.md`)
- Bounding box: yes
- Markers / gates / target zones: yes (test-specific elements above)
- HUD ticker fields: `distance_so_far`, `elapsed_time`, `current_lap_pace`
- End-card: all metrics + test score

## 7. Benchmark schema

- File: `benchmarks/physical/cooper.yaml`
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

Cooper (1968). VO₂max regression: VO₂max ≈ (distance_m − 504.9) / 44.73
