# `medicine-ball-throw` — Medicine Ball Throw (Seated / Standing)

**Domain**: physical
**Family**: throw
**Status**: active
**Pose backend**: `pose_biomech` (RTMPose-x — release biomechanics)
**Attempts**: 1 (single attempt per test, app-instructed)

## 1. Purpose

Upper-body / total-body power. Distance medicine ball is thrown.

## 2. Equipment & setup

Medicine ball (mass per protocol — typically 2–5 kg). Marked landing zone. Side-on camera.

## 3. Protocol

1. Athlete in seated or standing start position (declare in pipeline config)
2. Throws ball forward from chest as far as possible
3. Distance measured from start line to first landing point
4. 3 trials, best recorded

## 4. CV pipeline requirements

| Capability | Required? | Notes |
|---|---|---|
| Player detection + tracking | yes | |
| Pose estimation | yes | |
| Ball detection + tracking | yes | |
| Cone detection | no | |
| Calibration | mandatory (landing zone) | |
| Event detection | — | release, ball trajectory, landing |

## 5. Metrics produced

See `docs/metrics/METRICS_CATALOG.md` for formulas and units. `release_velocity_ms` is back-solved from the projectile model (distance, release angle, release height) — direct ball tracking through flight is fragile, so the projectile estimate is canonical.

**Throw:** `throw_distance_m`, `release_velocity_ms`, `release_angle_deg`, `flight_time_s`, `max_height_m`, `peak_arm_acceleration_ms2` (athlete-arm via wrist keypoint, not ball)

**Biomech:** `trunk_rotation_deg`

**Validity (universal):** `pose_confidence_low_pct`, `calibration_quality`, `tracking_id_drops`, `frames_athlete_offscreen_pct`

## 6. Annotation requirements

- Skeleton: as appropriate to the test (see family default in `docs/annotation/VIDEO_ANNOTATION_SPEC.md`)
- Bounding box: yes
- Markers / gates / target zones: yes (test-specific elements above)
- HUD ticker fields: `throw_distance_live`, `release_velocity`
- End-card: all metrics + test score

## 7. Benchmark schema

- File: `benchmarks/physical/medicine-ball-throw.yaml`
- Score direction: higher_is_better
- Aggregation: see benchmark file (`weighted_mean` default unless otherwise noted)

## 8. Failure modes

- Missing required entities (cones / ball / wall / box) → `DetectionError` or `CalibrationError`
- Protocol mismatches (duration, sequence) → `ProtocolError`
- Pose confidence below threshold on > 30% of frames → flag, do not abort

## 9. AI summary prompt notes

Use the family template `src/ai_summary/templates/throw.md`. See its
cross-metric narrative section for patterns to surface for this test.

## 10. Acceptance criteria

- [ ] All §5 metrics computed on a sample video
- [ ] All §6 overlays render
- [ ] Score normalisation produces sane 0–100 outputs
- [ ] Unit tests cover metric calculations
- [ ] Integration test passes end-to-end

## 11. References

Existing legacy `ball_throw.py` — port and clean up. Ball detection via the YOLO ball class.
