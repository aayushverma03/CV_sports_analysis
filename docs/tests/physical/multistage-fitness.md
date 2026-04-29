# `multistage-fitness` — Multistage Fitness Test (Bleep / Beep Test)

**Domain**: physical
**Family**: endurance
**Status**: active
**Stage detection**: audio (primary) + visual (cross-check); manual fallback

## 1. Purpose

Progressive 20 m shuttle test; classic VO₂max field estimator.

## 2. Equipment & setup

20 m shuttle, audio cues at increasing pace.

## 3. Protocol

1. Athlete shuttles 20 m back and forth in time with audio bleeps
2. Pace increases each level (~1 min per level)
3. Test ends when athlete fails to reach line on the bleep twice consecutively

## 4. CV pipeline requirements

| Capability | Required? | Notes |
|---|---|---|
| Player detection + tracking | yes | |
| Pose estimation | no | |
| Ball detection + tracking | no | |
| Cone detection | yes | |
| Calibration | mandatory | |
| Event detection | — | shuttle starts/ends, bleep timing, fail event |

## 5. Metrics produced

See `docs/metrics/METRICS_CATALOG.md` for formulas and units.

**Endurance:** `final_speed_level`, `missed_beep_count`, `total_distance_m`, `average_speed_ms`, `split_times_per_level_s`

**Validity:** `pose_confidence_low_pct`, `calibration_quality`, `tracking_id_drops`, `frames_athlete_offscreen_pct`, `audio_beep_alignment_quality` (audio mode), `manual_log_consistency_check` (manual mode)

**Stage-detection modes:** same dual-mode design as `yo-yo-intermittent` (audio primary, manual fallback). See that spec for detail.

## 6. Annotation requirements

- Skeleton: as appropriate to the test (see family default in `docs/annotation/VIDEO_ANNOTATION_SPEC.md`)
- Bounding box: yes
- Markers / gates / target zones: yes (test-specific elements above)
- HUD ticker fields: `current_level`, `shuttles_in_level`, `bleep_compliance`
- End-card: all metrics + test score

## 7. Benchmark schema

- File: `benchmarks/physical/multistage-fitness.yaml`
- Score direction: higher_is_better
- Aggregation: see benchmark file (`weighted_mean` default unless otherwise noted)

## 8. Failure modes

- Missing required entities (cones / ball / wall / box) → `DetectionError` or `CalibrationError`
- Protocol mismatches (duration, sequence) → `ProtocolError`
- Pose confidence below threshold on > 30% of frames → flag, do not abort

## 9. AI summary prompt notes

Use the family template `src/ai_summary/templates/endurance.md`. See its
cross-metric narrative section for patterns to surface for this test.

## 10. Acceptance criteria

- [ ] All §5 metrics computed on a sample video
- [ ] All §6 overlays render
- [ ] Score normalisation produces sane 0–100 outputs
- [ ] Unit tests cover metric calculations
- [ ] Integration test passes end-to-end

## 11. References

Léger & Lambert (1982). VO₂max regression per published table.
