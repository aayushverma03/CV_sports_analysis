# `illinois-agility` — Illinois Agility Test

**Domain**: physical
**Family**: agility
**Status**: active

## 1. Purpose

Course with straight sprints and zig-zag through cones. Classic agility assessment.

## 2. Equipment & setup

Rectangle 10 m × 5 m with 4 corner cones; 4 internal cones in a line (3.3 m apart) for the zig-zag.

## 3. Protocol

1. Athlete prone behind start cone (face down, hands by shoulders)
2. On signal, rise and sprint forward 10 m
3. Turn, sprint back 10 m
4. Weave through 4 internal cones (forward then back)
5. Sprint final 10 m to finish

## 4. CV pipeline requirements

| Capability | Required? | Notes |
|---|---|---|
| Player detection + tracking | yes | |
| Pose estimation | yes | |
| Ball detection + tracking | no | |
| Cone detection | yes | |
| Calibration | mandatory | |
| Event detection | — | rise from prone, each cone passage, finish |

## 5. Metrics produced

See `docs/metrics/METRICS_CATALOG.md` for formulas and units.

**Motion:** `total_completion_time_s`, `total_distance_m`, `average_speed_ms`, `max_speed_ms`, `peak_acceleration_ms2`, `peak_deceleration_ms2`, `split_times_s`

**Agility:** `cone_miss_events` (route violation / failure to round on the prescribed side)

**Validity (universal):** `pose_confidence_low_pct`, `calibration_quality`, `tracking_id_drops`, `frames_athlete_offscreen_pct`

## 6. Annotation requirements

- Skeleton: as appropriate to the test (see family default in `docs/annotation/VIDEO_ANNOTATION_SPEC.md`)
- Bounding box: yes
- Markers / gates / target zones: yes (test-specific elements above)
- HUD ticker fields: `elapsed_time`, `current_phase`
- End-card: all metrics + test score

## 7. Benchmark schema

- File: `benchmarks/physical/illinois-agility.yaml`
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

Standard protocol. Pose required for prone-start detection.
