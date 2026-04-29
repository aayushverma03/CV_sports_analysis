# `zig-zag-dribbling` — Zig-Zag Dribbling

**Domain**: technical
**Family**: dribbling
**Status**: active

## 1. Purpose

Dribble through a slalom of cones. Tight ball control under directional change.

## 2. Equipment & setup

5–7 cones at 2 m spacing in a slalom pattern. Football. Camera angled to see all cones.

## 3. Protocol

1. Athlete starts behind start gate with ball
2. Weaves through every cone (alternating sides)
3. Returns through finish gate
4. Penalty for missed cones

## 4. CV pipeline requirements

| Capability | Required? | Notes |
|---|---|---|
| Player detection + tracking | yes | |
| Pose estimation | yes | |
| Ball detection + tracking | yes | |
| Cone detection | yes | |
| Calibration | mandatory | |
| Event detection | — | start, cone passages, finish; missed cone = penalty |

## 5. Metrics produced

See `docs/metrics/METRICS_CATALOG.md` for formulas and units.

**Motion:** `total_completion_time_s`, `total_distance_m`, `average_speed_ms`, `max_speed_ms`, `peak_acceleration_ms2`, `peak_deceleration_ms2`, `split_times_s`

**Agility:** `cone_miss_events` (wrong-side passes / missed cones), `avg_cod_angle_deg`

**Ball:** `total_ball_touches`, `touches_per_metre`, `ball_foot_distance_m`, `ball_athlete_distance_m`

**Skill:** `left_leg_utilisation_pct`

**Validity (universal):** `pose_confidence_low_pct`, `calibration_quality`, `tracking_id_drops`, `frames_athlete_offscreen_pct`

## 6. Annotation requirements

- Skeleton: as appropriate to the test (see family default in `docs/annotation/VIDEO_ANNOTATION_SPEC.md`)
- Bounding box: yes
- Markers / gates / target zones: yes (test-specific elements above)
- HUD ticker fields: `elapsed_time`, `cones_passed`, `touches`
- End-card: all metrics + test score

## 7. Benchmark schema

- File: `benchmarks/technical/zig-zag-dribbling.yaml`
- Score direction: lower_is_better (time, with penalty per missed cone)
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

Standard skill assessment.
