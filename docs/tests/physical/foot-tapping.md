# `foot-tapping` — Foot Tapping Test

**Domain**: physical
**Family**: skill
**Status**: active
**Window**: fixed 10 s

## 1. Purpose

Maximum number of alternating foot taps in a fixed time. Measures lower-limb cyclic speed.

## 2. Equipment & setup

A football placed on the ground. Camera side-on or top-down with both feet
and the ball in frame.

## 3. Protocol

1. Athlete stands over the ball
2. Taps the ball alternately with each foot as fast as possible
3. Fixed duration (typically 10 s)
4. Total taps counted

## 4. CV pipeline requirements

| Capability | Required? | Notes |
|---|---|---|
| Player detection + tracking | yes | |
| Pose estimation | yes | ankles (15 left, 16 right) for tap classification |
| Ball detection + tracking | yes | COCO `sports_ball` (no custom training v1) |
| Cone detection | no | |
| Calibration | optional | not needed; counts only |
| Event detection | — | tap event = ankle keypoint within proximity of ball centre, then leaves; left vs right by which ankle was closer at contact |

## 5. Metrics produced

See `docs/metrics/METRICS_CATALOG.md` for formulas and units.

**Coordination:** `total_taps`, `taps_per_second` (benchmarked), `left_taps`, `right_taps` (informational, not benchmarked)

**Validity (universal):** `pose_confidence_low_pct`, `calibration_quality`, `tracking_id_drops`, `frames_athlete_offscreen_pct`

## 6. Annotation requirements

- Skeleton: as appropriate to the test (see family default in `docs/annotation/VIDEO_ANNOTATION_SPEC.md`)
- Bounding box: yes
- Markers / gates / target zones: yes (test-specific elements above)
- HUD ticker fields: `live_tap_count`, `taps_per_second_running`
- End-card: all metrics + test score

## 7. Benchmark schema

- File: `benchmarks/physical/foot-tapping.yaml`
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

Existing legacy `tests/analyzers/tapping.py` — port. Tap detection via ankle keypoint Z-velocity threshold.
