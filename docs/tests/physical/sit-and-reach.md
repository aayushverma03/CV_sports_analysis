# `sit-and-reach` — Sit-and-Reach

**Domain**: physical
**Family**: mobility
**Status**: deferred (out of v1 scope — no sample video; ships in v1.1+)

## 1. Purpose

Trunk flexion and posterior chain flexibility. Distance reached forward from a seated position.

## 2. Equipment & setup

Sit-and-reach box with cm scale OR floor markings beyond athlete's feet. Side-on camera.

## 3. Protocol

1. Athlete sits with legs extended, feet flat against the box
2. Reaches forward slowly with both hands stacked
3. Holds the maximum reach for 2 s
4. Distance recorded at fingertip relative to feet (zero at toes)

## 4. CV pipeline requirements

| Capability | Required? | Notes |
|---|---|---|
| Player detection + tracking | yes | |
| Pose estimation | yes | |
| Ball detection + tracking | no | |
| Cone detection | no | |
| Calibration | mandatory (cm scale) | |
| Event detection | — | stable seated start, max reach, hold (2 s), release |

## 5. Metrics produced

| Metric ID | Module | Unit |
|---|---|---|
| `reach_distance` | `metrics/biomech/reach_distance.py` | cm (signed; +ve past toes) |
| `trunk_flexion_angle` | `metrics/biomech/trunk_flexion_angle.py` | ° |
| `hold_duration` | `metrics/biomech/hold_duration.py` | s |

## 6. Annotation requirements

- Skeleton: as appropriate to the test (see family default in `docs/annotation/VIDEO_ANNOTATION_SPEC.md`)
- Bounding box: yes
- Markers / gates / target zones: yes (test-specific elements above)
- HUD ticker fields: `live_reach`, `trunk_angle`
- End-card: all metrics + test score

## 7. Benchmark schema

- File: `benchmarks/physical/sit-and-reach.yaml`
- Score direction: higher_is_better
- Aggregation: see benchmark file (`weighted_mean` default unless otherwise noted)

## 8. Failure modes

- Missing required entities (cones / ball / wall / box) → `DetectionError` or `CalibrationError`
- Protocol mismatches (duration, sequence) → `ProtocolError`
- Pose confidence below threshold on > 30% of frames → flag, do not abort

## 9. AI summary prompt notes

Use the family template `src/ai_summary/templates/mobility.md`. See its
cross-metric narrative section for patterns to surface for this test.

## 10. Acceptance criteria

- [ ] All §5 metrics computed on a sample video
- [ ] All §6 overlays render
- [ ] Score normalisation produces sane 0–100 outputs
- [ ] Unit tests cover metric calculations
- [ ] Integration test passes end-to-end

## 11. References

ACSM. Pose must give reliable hip + shoulder + wrist keypoints.
