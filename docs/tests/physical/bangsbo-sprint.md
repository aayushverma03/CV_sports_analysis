# `bangsbo-sprint` — Bangsbo Sprint Test (7 × 34.2 m)

**Domain**: physical
**Family**: sprint
**Status**: active

## 1. Purpose

Repeated 7×34.2 m sprints with fixed rest intervals. Football-specific protocol assessing ability to repeat near-maximal sprints.

## 2. Equipment & setup

Marked 34.2 m course with start and finish gates; cone at turnaround.

## 3. Protocol

1. Athlete completes 7 sprints of 34.2 m
2. 25 s rest between sprints (operator-cued or audio)
3. Each sprint timed independently

## 4. CV pipeline requirements

| Capability | Required? | Notes |
|---|---|---|
| Player detection + tracking | yes | |
| Pose estimation | no | |
| Ball detection + tracking | no | |
| Cone detection | yes | |
| Calibration | mandatory | |
| Event detection | — | sprint start × 7, sprint end × 7 |

## 5. Metrics produced

See `docs/metrics/METRICS_CATALOG.md` for formulas and units. Bangsbo (7×34.2m) is a pure sprint test — **no ball metrics**.

**Motion:** `total_completion_time_s`, `total_distance_m`, `average_speed_ms`, `max_speed_ms`, `peak_acceleration_ms2`, `peak_deceleration_ms2`

**Sprint family:** `rep_times_s` (7 reps), `sprint_best_s`, `sprint_worst_s`, `sprint_mean_s`, `pct_sprint_decrement` (Glaister formula)

**Validity (universal):** `pose_confidence_low_pct`, `calibration_quality`, `tracking_id_drops`, `frames_athlete_offscreen_pct`

## 6. Annotation requirements

- Skeleton: as appropriate to the test (see family default in `docs/annotation/VIDEO_ANNOTATION_SPEC.md`)
- Bounding box: yes
- Markers / gates / target zones: yes (test-specific elements above)
- HUD ticker fields: `current_sprint`, `sprint_time`, `fatigue_index_running`
- End-card: all metrics + test score

## 7. Benchmark schema

- File: `benchmarks/physical/bangsbo-sprint.yaml`
- Score direction: lower_is_better (times); higher_is_better (avg & max speed)
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

Bangsbo (1994). Implements own metric `fatigue_index` — add to `metrics/endurance/`.
