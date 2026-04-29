# `hurdle-agility-run` — Hurdle Agility Run

**Domain**: physical
**Family**: agility
**Status**: active

## 1. Purpose

Course combining straight sprints, COD, and low-hurdle clearances. Tests multi-component agility plus jump-and-go transitions.

## 2. Equipment & setup

Cones and 3–5 mini-hurdles (15–30 cm) in a defined sequence. Layout per protocol config.

## 3. Protocol

1. Athlete sprints through course
2. Clears each hurdle without contact
3. Performs cone-defined COD between hurdle sets

## 4. CV pipeline requirements

| Capability | Required? | Notes |
|---|---|---|
| Player detection + tracking | yes | |
| Pose estimation | yes | |
| Ball detection + tracking | no | |
| Cone detection | yes | |
| Calibration | mandatory | |
| Event detection | — | start, each hurdle clearance (foot above hurdle), finish; hurdle contact = penalty |

## 5. Metrics produced

See `docs/metrics/METRICS_CATALOG.md` for formulas and units.

**Motion:** `total_completion_time_s`, `total_distance_m`, `average_speed_ms`, `max_speed_ms`, `peak_acceleration_ms2`, `peak_deceleration_ms2`

**Agility (Hurdle-specific):** `hurdles_cleared`, `non_clearance_count`, `disqualified` (true if `non_clearance_count > 2`), `lead_foot`, `avg_hurdle_time_s`, `per_hurdle_split_times_s`, `stride_cadence_hz`, `avg_ground_contact_time_s`, `knee_symmetry_ratio`

**Biomech:** `trunk_lean_over_hurdle_deg`

**Validity (universal):** `pose_confidence_low_pct`, `calibration_quality`, `tracking_id_drops`, `frames_athlete_offscreen_pct`

**Non-clearance detection** (signal fusion): hurdle bbox displacement >5 cm frame-over-frame OR ankle keypoint passing through hurdle bbox. Side-on camera angle required.

## 6. Annotation requirements

- Skeleton: as appropriate to the test (see family default in `docs/annotation/VIDEO_ANNOTATION_SPEC.md`)
- Bounding box: yes
- Markers / gates / target zones: yes (test-specific elements above)
- HUD ticker fields: `elapsed_time`, `hurdles_cleared`
- End-card: all metrics + test score

## 7. Benchmark schema

- File: `benchmarks/physical/hurdle-agility-run.yaml`
- Score direction: lower_is_better with penalty per contact
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

Custom DFB-style hurdle agility protocol. Hurdle contact via ankle keypoint trajectory + hurdle bbox.
