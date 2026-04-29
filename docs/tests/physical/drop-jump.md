# `drop-jump` — Drop Jump

**Domain**: physical
**Family**: jump
**Status**: active
**Pose backend**: `pose_biomech` (RTMPose-x)
**Attempts**: 1 (single attempt per test, app-instructed)

## 1. Purpose

Athlete drops from a box and immediately rebounds. Measures reactive strength (RSI) and short stretch-shortening cycle.

## 2. Equipment & setup

Plyometric box (height per protocol — typical 30 / 40 / 60 cm). Side-on camera.

## 3. Protocol

1. Athlete stands on box edge
2. Steps off (does not jump down)
3. On landing, immediately rebounds vertically
4. Lands on same spot

## 4. CV pipeline requirements

| Capability | Required? | Notes |
|---|---|---|
| Player detection + tracking | yes | |
| Pose estimation | yes | |
| Ball detection + tracking | no | |
| Cone detection | no | |
| Calibration | box height as reference | |
| Event detection | — | step-off, ground contact, rebound takeoff, peak, landing |

## 5. Metrics produced

See `docs/metrics/METRICS_CATALOG.md` for formulas and units. Drop height (30/40/50 cm) is a setup parameter, not a metric.

**Jump:** `jump_height_cm` (rebound height), `flight_time_s`, `ground_contact_time_s`, `rsi`, `peak_landing_deceleration_ms2`

**Biomech:** `trunk_lean_initial_contact_deg`

**Validity (universal):** `pose_confidence_low_pct`, `calibration_quality`, `tracking_id_drops`, `frames_athlete_offscreen_pct`. Note: single-camera `peak_landing_deceleration_ms2` is approximate (±20%).

## 6. Annotation requirements

- Skeleton: as appropriate to the test (see family default in `docs/annotation/VIDEO_ANNOTATION_SPEC.md`)
- Bounding box: yes
- Markers / gates / target zones: yes (test-specific elements above)
- HUD ticker fields: `contact_time`, `rebound_height`, `rsi`
- End-card: all metrics + test score

## 7. Benchmark schema

- File: `benchmarks/physical/drop-jump.yaml`
- Score direction: higher_is_better (RSI is the headline)
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

Young (1995). Legacy `tests/analyzers/drop_jump.py`.
