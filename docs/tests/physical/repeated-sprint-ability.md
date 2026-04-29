# `repeated-sprint-ability` — Repeated Sprint Ability (RSA)

**Domain**: physical
**Family**: sprint
**Status**: active

## 1. Purpose

Series of maximal sprints with short rests; quantifies anaerobic capacity and resistance to performance decrement.

## 2. Equipment & setup

30 m straight lane (or 6×40 m or 10×20 m depending on protocol variant declared in config).

## 3. Protocol

1. Athlete completes N sprints (default 6 × 30 m)
2. Fixed rest interval between sprints (default 20 s)
3. Each sprint at maximum effort

## 4. CV pipeline requirements

| Capability | Required? | Notes |
|---|---|---|
| Player detection + tracking | yes | |
| Pose estimation | no | |
| Ball detection + tracking | no | |
| Cone detection | yes | |
| Calibration | mandatory | |
| Event detection | — | start × N, end × N |

## 5. Metrics produced

See `docs/metrics/METRICS_CATALOG.md` for formulas and units.

**Motion:** `total_completion_time_s`, `total_distance_m`, `average_speed_ms`, `max_speed_ms`, `peak_acceleration_ms2`, `peak_deceleration_ms2`

**Sprint family:** `rep_times_s`, `sprint_best_s`, `sprint_worst_s`, `sprint_mean_s`, `pct_sprint_decrement` (Glaister formula)

**Validity (universal):** `pose_confidence_low_pct`, `calibration_quality`, `tracking_id_drops`, `frames_athlete_offscreen_pct`

## 6. Annotation requirements

- Skeleton: as appropriate to the test (see family default in `docs/annotation/VIDEO_ANNOTATION_SPEC.md`)
- Bounding box: yes
- Markers / gates / target zones: yes (test-specific elements above)
- HUD ticker fields: `current_sprint`, `best_so_far`, `fatigue_index_running`
- End-card: all metrics + test score

## 7. Benchmark schema

- File: `benchmarks/physical/repeated-sprint-ability.yaml`
- Score direction: lower_is_better for times; lower_is_better for fatigue_index
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

Spencer et al. (2005). Variant config declared in test pipeline.
