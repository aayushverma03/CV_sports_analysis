# `yo-yo-intermittent` — Yo-Yo Intermittent Recovery Test (IR2)

**Domain**: physical
**Family**: endurance
**Status**: active
**Stage detection**: audio (primary) + visual (cross-check); manual fallback

## 1. Purpose

Progressive shuttle test with recovery jogging between sprints; specific to football's intermittent demands.

## 2. Equipment & setup

20 m shuttle marked by cones, 5 m recovery zone behind start cone. Audio cues for pace.

## 3. Protocol — Yo-Yo IR2

Variant locked: **Intermittent Recovery Test, Level 2 (IR2)** — starting speed
13 km/h, faster progression than IR1. More demanding; appropriate for elite /
high-level football populations (which is our target).

1. Athlete completes 2 × 20 m shuttle at audio-cued pace (starting at IR2 speeds)
2. 10 s active recovery jog through 5 m zone
3. Pace increases at each level until athlete fails to reach the line on the cue (twice)

## 4. CV pipeline requirements

| Capability | Required? | Notes |
|---|---|---|
| Player detection + tracking | yes | |
| Pose estimation | no | |
| Ball detection + tracking | no | |
| Cone detection | yes | |
| Calibration | mandatory | |
| Event detection | — | shuttle start, line crossing per cue, recovery zone entry/exit, fail event |

## 5. Metrics produced

See `docs/metrics/METRICS_CATALOG.md` for formulas and units.

**Endurance:** `final_speed_level`, `shuttles_at_final_level`, `num_shuttles_completed`, `missed_beep_count`, `total_distance_m`, `total_completion_time_s`, `vo2max_estimated`, `split_times_s`

**Motion:** `max_speed_ms`, `peak_acceleration_ms2`, `peak_deceleration_ms2`

**Validity:** `pose_confidence_low_pct`, `calibration_quality`, `tracking_id_drops`, `frames_athlete_offscreen_pct`, `audio_beep_alignment_quality` (audio mode), `manual_log_consistency_check` (manual mode)

**Stage-detection modes:**
1. **Audio (primary):** detect beep onsets in video soundtrack → align with athlete line crossings → `successful_shuttle` if crossing precedes next beep, else `missed_beep`. End after 2 consecutive misses.
2. **Manual (fallback):** operator submits final stage; pipeline still computes `num_shuttles_completed` from video and flags `manual_log_consistency_check: inconsistent` if the visual count diverges by more than ±1 level.

## 6. Annotation requirements

- Skeleton: as appropriate to the test (see family default in `docs/annotation/VIDEO_ANNOTATION_SPEC.md`)
- Bounding box: yes
- Markers / gates / target zones: yes (test-specific elements above)
- HUD ticker fields: `current_level`, `shuttles_completed`, `audio_cue_phase`
- End-card: all metrics + test score

## 7. Benchmark schema

- File: `benchmarks/physical/yo-yo-intermittent.yaml`
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

Bangsbo et al. (2008). Two variants (YYIR1 / YYIR2) — declare which in pipeline config.
