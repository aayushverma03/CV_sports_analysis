# `juggling` — Juggling Test

**Domain**: technical
**Family**: skill
**Status**: active
**Drop event**: ball Y-coord reaches ground plane AND ball is no longer within 0.5 m of any foot/knee/head keypoint. Hand catches and double-bounces also end the streak.

## 1. Purpose

Continuous ball juggling without ground contact. Measures ball-control quality and consistency.

## 2. Equipment & setup

Football. Open space. Camera framing the athlete from waist up.

## 3. Protocol

1. Athlete starts juggling on signal
2. Counts each clean touch (foot, thigh, head per protocol)
3. Ends when ball touches the ground OR fixed duration elapses

## 4. CV pipeline requirements

| Capability | Required? | Notes |
|---|---|---|
| Player detection + tracking | yes | |
| Pose estimation | yes | |
| Ball detection + tracking | yes | |
| Cone detection | no | |
| Calibration | optional | |
| Event detection | — | touch events with body-part labels, ball drops |

## 5. Metrics produced

See `docs/metrics/METRICS_CATALOG.md` for formulas and units.

**Ball:** `max_consecutive_touches`, `total_ball_touches`, `touches_per_second`, `ball_foot_distance_m`

**Skill:** `left_leg_utilisation_pct`

**Validity (universal):** `pose_confidence_low_pct`, `calibration_quality`, `tracking_id_drops`, `frames_athlete_offscreen_pct`

## 6. Annotation requirements

- Skeleton: as appropriate to the test (see family default in `docs/annotation/VIDEO_ANNOTATION_SPEC.md`)
- Bounding box: yes
- Markers / gates / target zones: yes (test-specific elements above)
- HUD ticker fields: `touch_count`, `current_streak`, `touches_per_second`
- End-card: all metrics + test score

## 7. Benchmark schema

- File: `benchmarks/technical/juggling.yaml`
- Score direction: higher_is_better
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

Existing legacy `juggling_test.py` — port.
