# `straight-line-dribbling` — Straight Line Dribbling

**Domain**: technical
**Family**: dribbling
**Status**: active

## 1. Purpose

Dribble a ball in a straight line at maximum speed. Combines locomotion with ball-control demands.

## 2. Equipment & setup

30 m straight lane with start/finish gates. Football. Side-on camera.

## 3. Protocol

1. Athlete starts behind line with ball at feet
2. Dribbles maximally to finish line
3. Ball must remain within lane

## 4. CV pipeline requirements

| Capability | Required? | Notes |
|---|---|---|
| Player detection + tracking | yes | |
| Pose estimation | yes | |
| Ball detection + tracking | yes | |
| Cone detection | yes | |
| Calibration | mandatory | |
| Event detection | — | start, ball touches, finish; lane exit = penalty |

## 5. Metrics produced

See `docs/metrics/METRICS_CATALOG.md` for formulas and units.

**Motion:** `total_completion_time_s`, `total_distance_m`, `average_speed_ms`, `max_speed_ms`, `peak_acceleration_ms2`, `peak_deceleration_ms2`, `split_times_s`

**Ball:** `total_ball_touches`, `touches_per_metre`, `ball_foot_distance_m`, `ball_athlete_distance_m`, `control_loss_events`, `player_lane_deviation_m`, `ball_lane_deviation_m`

**Skill:** `left_leg_utilisation_pct`

**Validity (universal):** `pose_confidence_low_pct`, `calibration_quality`, `tracking_id_drops`, `frames_athlete_offscreen_pct`

## 6. Annotation requirements

- Skeleton: as appropriate to the test (see family default in `docs/annotation/VIDEO_ANNOTATION_SPEC.md`)
- Bounding box: yes
- Markers / gates / target zones: yes (test-specific elements above)
- HUD ticker fields: `elapsed_time`, `current_speed`, `touches`
- End-card: all metrics + test score

## 7. Benchmark schema

- File: `benchmarks/technical/straight-line-dribbling.yaml`
- Score direction: lower_is_better (time); higher_is_better (speed); target_value (touches/m)
- Aggregation: see benchmark file (`weighted_mean` default unless otherwise noted)

## 8. Failure modes

- Missing required entities (cones / ball / wall / box) → `DetectionError` or `CalibrationError`
- Protocol mismatches (duration, sequence) → `ProtocolError`
- Pose confidence below threshold on > 30% of frames → flag, do not abort

## 9. AI summary prompt notes

Use the family template `src/ai_summary/templates/dribbling.md`. See its
cross-metric narrative section for patterns to surface for this test.

## 10. Acceptance criteria

- [ ] All §5 metrics computed on a sample video
- [ ] All §6 overlays render
- [ ] Score normalisation produces sane 0–100 outputs
- [ ] Unit tests cover metric calculations
- [ ] Integration test passes end-to-end

## 11. References

Standard skill battery. Existing legacy dribbling pipeline — port.
