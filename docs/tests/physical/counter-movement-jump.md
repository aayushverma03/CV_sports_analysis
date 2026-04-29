# `counter-movement-jump` — Counter Movement Jump (CMJ)

**Domain**: physical
**Family**: jump
**Status**: active
**Pose backend**: `pose_biomech` (RTMPose-x)
**Attempts**: 1 (single attempt per test, app-instructed)

## 1. Purpose

Standard vertical jump from upright stance with countermovement. Estimates lower-body explosive power.

## 2. Equipment & setup

Side-on camera at hip height, full body in frame, plain wall background preferred.

## 3. Protocol

1. Athlete stands upright, hands on hips (or arms free per variant)
2. Drops into shallow squat
3. Jumps vertically as high as possible
4. Lands on same spot

The pipeline must auto-detect:
- **Test-start event**: countermovement onset — hip-midpoint (mean of left+right hip keypoints) downward y-velocity exceeds threshold (e.g. 0.3 m/s, smoothed). The HUD elapsed-time clock and all metric windows count from this event, **not** from video frame 0.
- **Takeoff**: toe leaves ground (ankle keypoint y-velocity flips upward, foot off-floor)
- **Peak**: hip-midpoint reaches max height
- **Landing**: toe contact (ankle y stops rising and stabilises near start baseline)

## 4. CV pipeline requirements

| Capability | Required? | Notes |
|---|---|---|
| Player detection + tracking | yes | |
| Pose estimation | yes | |
| Ball detection + tracking | no | |
| Cone detection | no | |
| Calibration | optional (height via flight time, not pixel measurement) | |
| Event detection | — | start of countermovement (hip drops), takeoff (toe leaves ground), peak, landing (toe contact) |

## 5. Metrics produced

See `docs/metrics/METRICS_CATALOG.md` for formulas and units.

**Jump:** `jump_height_cm`, `flight_time_s`, `peak_takeoff_acceleration_ms2`

**Biomech:** `trunk_lean_takeoff_deg`

**Validity (universal):** `pose_confidence_low_pct`, `calibration_quality`, `tracking_id_drops`, `frames_athlete_offscreen_pct`

## 6. Annotation requirements

- Skeleton: as appropriate to the test (see family default in `docs/annotation/VIDEO_ANNOTATION_SPEC.md`)
- Bounding box: yes
- Markers / gates / target zones: yes (test-specific elements above)
- HUD ticker fields: `jump_height_live`, `flight_time`
- End-card: all metrics + test score

## 7. Benchmark schema

- File: `benchmarks/physical/counter-movement-jump.yaml`
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

Bosco et al. (1983). Existing legacy `streamlit_app.py` jump pipeline — port.
