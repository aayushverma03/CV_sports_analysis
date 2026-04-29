# `squat-jump` — Squat Jump

**Domain**: physical
**Family**: jump
**Status**: active
**Pose backend**: `pose_biomech` (RTMPose-x)
**Attempts**: 1 (single attempt per test, app-instructed)

## 1. Purpose

Vertical jump from a held squat (no countermovement). Isolates concentric force production.

## 2. Equipment & setup

Side-on camera. No box.

## 3. Protocol

1. Athlete drops into ~90° knee flexion
2. Holds for 2–3 s
3. Jumps vertically without further countermovement
4. Lands on same spot

## 4. CV pipeline requirements

| Capability | Required? | Notes |
|---|---|---|
| Player detection + tracking | yes | |
| Pose estimation | yes | |
| Ball detection + tracking | no | |
| Cone detection | no | |
| Calibration | none required | |
| Event detection | — | squat hold (knee angle stable for ≥1.5s), takeoff, landing |

## 5. Metrics produced

See `docs/metrics/METRICS_CATALOG.md` for formulas and units. Validity: pipeline must flag if it detects a hip drop (countermovement) before takeoff — that disqualifies the attempt as a Squat Jump.

**Jump:** `jump_height_cm`, `flight_time_s`, `peak_takeoff_acceleration_ms2`, `min_knee_angle_deg` (squat depth)

**Validity (universal):** `pose_confidence_low_pct`, `calibration_quality`, `tracking_id_drops`, `frames_athlete_offscreen_pct`

## 6. Annotation requirements

- Skeleton: as appropriate to the test (see family default in `docs/annotation/VIDEO_ANNOTATION_SPEC.md`)
- Bounding box: yes
- Markers / gates / target zones: yes (test-specific elements above)
- HUD ticker fields: `jump_height_live`, `flight_time`
- End-card: all metrics + test score

## 7. Benchmark schema

- File: `benchmarks/physical/squat-jump.yaml`
- Score direction: higher_is_better
- Aggregation: see benchmark file (`weighted_mean` default unless otherwise noted)

## 8. Failure modes

- Missing required entities (cones / ball / wall / box) → `DetectionError` or `CalibrationError`
- Protocol mismatches (duration, sequence) → `ProtocolError`
- Pose confidence below threshold on > 30% of frames → flag, do not abort

## 9. AI summary prompt notes

Use the family template `src/ai_summary/templates/jump.md`. See its
cross-metric narrative section for patterns to surface for this test.

## 10. Acceptance criteria

- [ ] All §5 metrics computed on a sample video
- [ ] All §6 overlays render
- [ ] Score normalisation produces sane 0–100 outputs
- [ ] Unit tests cover metric calculations
- [ ] Integration test passes end-to-end

## 11. References

Differs from CMJ by absence of countermovement — flag if pipeline detects hip drop before takeoff.
