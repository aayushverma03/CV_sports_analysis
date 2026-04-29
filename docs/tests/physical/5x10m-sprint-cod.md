# `5x10m-sprint-cod` — 5 × 10 m Sprint with Change of Direction

**Domain**: physical
**Family**: sprint
**Status**: active

## 1. Purpose

Repeated short sprints (5 shuttles of 10 m) with 180° turns. Measures repeated-effort acceleration, deceleration capacity, and re-acceleration after each turn.

## 2. Equipment & setup

Two cones 10 m apart on flat surface. Camera perpendicular to the lane, full course visible.

## 3. Protocol

1. Athlete starts at cone A in stationary stance
2. Sprint to cone B, plant foot beyond B, 180° turn
3. Sprint back to cone A, plant foot beyond A, 180° turn
4. Repeat until 5 × 10 m = 50 m total covered
5. Time stops when chest crosses the final line

## 4. CV pipeline requirements

| Capability | Required? | Notes |
|---|---|---|
| Player detection + tracking | yes | |
| Pose estimation | yes | |
| Ball detection + tracking | no | |
| Cone detection | yes | |
| Calibration | mandatory | |
| Event detection | — | start, each shuttle turnaround (foot plant past cone), final crossing |

## 5. Metrics produced

See `docs/metrics/METRICS_CATALOG.md` for formulas and units.

**Motion:** `total_completion_time_s`, `total_distance_m`, `average_speed_ms`, `max_speed_ms`, `peak_acceleration_ms2`, `peak_deceleration_ms2`

**Sprint family:** `rep_times_s` (5 reps), `fatigue_drop_off_pct` (two-point: rep5 vs rep1)

**Validity (universal):** `pose_confidence_low_pct`, `calibration_quality`, `tracking_id_drops`, `frames_athlete_offscreen_pct`

## 6. Annotation requirements

- Skeleton: as appropriate to the test (see family default in `docs/annotation/VIDEO_ANNOTATION_SPEC.md`)
- Bounding box: yes
- Markers / gates / target zones: yes (test-specific elements above)
- HUD ticker fields: `current_shuttle`, `elapsed_time`, `current_speed`
- End-card: all metrics + test score

## 7. Benchmark schema

- File: `benchmarks/physical/5x10m-sprint-cod.yaml`
- Score direction: lower_is_better (times); higher_is_better (peak metrics)
- Aggregation: see benchmark file (`weighted_mean` default unless otherwise noted)

## 8. Failure modes

- Missing required entities (cones / ball / wall / box) → `DetectionError` or `CalibrationError`
- Protocol mismatches (duration, sequence) → `ProtocolError`
- Pose confidence below threshold on > 30% of frames → flag, do not abort

## 9. AI summary prompt notes

Use the family template `src/ai_summary/templates/sprint.md`. See its
cross-metric narrative section for patterns to surface for this test.

## 10. Acceptance criteria

- [ ] All §5 metrics computed on a sample video
- [ ] All §6 overlays render
- [ ] Score normalisation produces sane 0–100 outputs
- [ ] Unit tests cover metric calculations
- [ ] Integration test passes end-to-end

## 11. References

Legacy: parts of `agility.py` and `tests/sprint_5x10_test.py`.
