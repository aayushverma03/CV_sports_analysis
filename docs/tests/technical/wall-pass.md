# `wall-pass` — Wall Pass Test

**Domain**: technical
**Family**: skill
**Status**: active
**Window**: fixed 30 s
**Required setup**: wall distance and target-zone polygon must be configured at run-time (without these, accuracy is undefined)

## 1. Purpose

Repeated passes against a wall, receiving and passing again. Measures passing accuracy + first-touch control speed.

## 2. Equipment & setup

Wall, target zone marked on it. Athlete stands behind a marked line at fixed distance. Football.

## 3. Protocol

1. Athlete passes ball to wall target
2. Receives rebound, controls, passes again
3. Counts successful passes within fixed duration

## 4. CV pipeline requirements

| Capability | Required? | Notes |
|---|---|---|
| Player detection + tracking | yes | |
| Pose estimation | yes | |
| Ball detection + tracking | yes | |
| Cone detection | no | |
| Calibration | mandatory (wall distance) | |
| Event detection | — | pass releases, ball-wall contacts, receptions |

## 5. Metrics produced

See `docs/metrics/METRICS_CATALOG.md` for formulas and units.

**Skill / passing:** `successful_passes`, `passing_accuracy_percent`, `average_decision_time_s`, `average_pass_velocity_ms`, `max_pass_velocity_ms`, `left_leg_utilisation_pct`

**Ball:** `ball_foot_distance_m` (during reception phase)

**Biomech:** `body_approach_angle_deg` (athlete trunk-facing direction vs incoming ball trajectory at reception)

**Validity (universal):** `pose_confidence_low_pct`, `calibration_quality`, `tracking_id_drops`, `frames_athlete_offscreen_pct`

## 6. Annotation requirements

- Skeleton: as appropriate to the test (see family default in `docs/annotation/VIDEO_ANNOTATION_SPEC.md`)
- Bounding box: yes
- Markers / gates / target zones: yes (test-specific elements above)
- HUD ticker fields: `pass_count`, `accuracy_running`, `current_pass_velocity`
- End-card: all metrics + test score

## 7. Benchmark schema

- File: `benchmarks/technical/wall-pass.yaml`
- Score direction: higher_is_better (counts, accuracy, velocity); lower_is_better (decision time)
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

Standard technical battery.
