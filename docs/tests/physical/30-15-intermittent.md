# `30-15-intermittent` — 30-15 Intermittent Fitness Test

**Domain**: physical
**Family**: endurance
**Status**: deferred (out of v1 scope — no sample video; ships in v1.1+)

## 1. Purpose

30 s shuttle running + 15 s passive recovery, progressive pace. Intermittent fitness benchmark.

## 2. Equipment & setup

40 m shuttle with 3 m tolerance zones at each end. Audio cues.

## 3. Protocol

1. 30 s of shuttle running at cued pace
2. 15 s passive recovery walking
3. Pace increases each stage by 0.5 km/h
4. Test ends when athlete cannot reach tolerance zone three times

## 4. CV pipeline requirements

| Capability | Required? | Notes |
|---|---|---|
| Player detection + tracking | yes | |
| Pose estimation | no | |
| Ball detection + tracking | no | |
| Cone detection | yes | |
| Calibration | mandatory | |
| Event detection | — | stage start/end, recovery transitions, fail events |

## 5. Metrics produced

| Metric ID | Module | Unit |
|---|---|---|
| `vifit_30_15` | `metrics/endurance/vifit_30_15.py` | km/h (final stage velocity) |
| `stage_reached` | `metrics/endurance/stage_reached.py` | stage |
| `vo2_max_estimate` | `metrics/endurance/vo2_max_estimate.py` | ml·kg⁻¹·min⁻¹ |
| `total_distance_completed` | `metrics/endurance/total_distance_completed.py` | m |

## 6. Annotation requirements

- Skeleton: as appropriate to the test (see family default in `docs/annotation/VIDEO_ANNOTATION_SPEC.md`)
- Bounding box: yes
- Markers / gates / target zones: yes (test-specific elements above)
- HUD ticker fields: `current_stage`, `phase (run / rest)`, `distance_in_phase`
- End-card: all metrics + test score

## 7. Benchmark schema

- File: `benchmarks/physical/30-15-intermittent.yaml`
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

Buchheit (2008). VIFT = velocity at last completed stage.
