# `t-test` — T-Test (Agility)

**Domain**: physical
**Family**: agility
**Status**: active

## 1. Purpose

T-shaped course tests forward sprint, lateral shuffle, backpedal. Multi-directional agility.

## 2. Equipment & setup

4 cones in T-shape: A (start) → B (10 yd / 9.14 m forward) → C (5 yd / 4.57 m left of B) → D (5 yd right of B). Total path: A→B→C→B→D→B→A.

## 3. Protocol

1. Sprint forward A→B, touch base of cone B with right hand
2. Side-shuffle left to C, touch C with left hand (no crossover)
3. Side-shuffle right to D, touch D with right hand
4. Side-shuffle back to B, touch B with left hand
5. Backpedal B→A through finish line

## 4. CV pipeline requirements

| Capability | Required? | Notes |
|---|---|---|
| Player detection + tracking | yes | |
| Pose estimation | yes | |
| Ball detection + tracking | no | |
| Cone detection | yes | |
| Calibration | mandatory | |
| Event detection | — | start, each cone touch (5 touches), finish |

## 5. Metrics produced

See `docs/metrics/METRICS_CATALOG.md` for formulas and units.

**Motion:** `total_completion_time_s`, `total_distance_m`, `average_speed_ms`, `max_speed_ms`, `peak_acceleration_ms2`, `peak_deceleration_ms2`, `split_times_s`

**T-Test specific:** `segment_completion_times_s` (forward / lat-right / lat-left / backpedal segments)

**Validity (universal):** `pose_confidence_low_pct`, `calibration_quality`, `tracking_id_drops`, `frames_athlete_offscreen_pct`

## 6. Annotation requirements

- Skeleton: as appropriate to the test (see family default in `docs/annotation/VIDEO_ANNOTATION_SPEC.md`)
- Bounding box: yes
- Markers / gates / target zones: yes (test-specific elements above)
- HUD ticker fields: `elapsed_time`, `current_segment`
- End-card: all metrics + test score

## 7. Benchmark schema

- File: `benchmarks/physical/t-test.yaml`
- Score direction: lower_is_better
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

Pauole et al. (2000). Cone-touch detection: hand keypoint within X cm of cone.
