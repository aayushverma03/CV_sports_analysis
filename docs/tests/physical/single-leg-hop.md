# `single-leg-hop` — Single-Leg Hop

**Domain**: physical
**Family**: jump
**Status**: deferred (out of v1 scope — no sample video; ships in v1.1+)

## 1. Purpose

Horizontal hop on one leg; tests unilateral power and detects left/right asymmetry.

## 2. Equipment & setup

Marked floor with distance scale. Side-on camera.

## 3. Protocol

1. Athlete stands on test leg behind start line
2. Hops as far as possible, landing on same leg
3. Hold landing for 2 s without losing balance
4. Repeat on other leg
5. 3 trials per side; record best

## 4. CV pipeline requirements

| Capability | Required? | Notes |
|---|---|---|
| Player detection + tracking | yes | |
| Pose estimation | yes | |
| Ball detection + tracking | no | |
| Cone detection | no | |
| Calibration | mandatory | |
| Event detection | — | takeoff, landing, balance hold (2 s stable) |

## 5. Metrics produced

| Metric ID | Module | Unit |
|---|---|---|
| `hop_distance_left` | `metrics/jump/hop_distance_left.py` | m |
| `hop_distance_right` | `metrics/jump/hop_distance_right.py` | m |
| `left_right_asymmetry` | `metrics/biomech/left_right_asymmetry.py` | % |
| `balance_hold_quality` | `metrics/biomech/balance_hold_quality.py` | 0..1 |

## 6. Annotation requirements

- Skeleton: as appropriate to the test (see family default in `docs/annotation/VIDEO_ANNOTATION_SPEC.md`)
- Bounding box: yes
- Markers / gates / target zones: yes (test-specific elements above)
- HUD ticker fields: `current_leg`, `hop_distance_live`
- End-card: all metrics + test score

## 7. Benchmark schema

- File: `benchmarks/physical/single-leg-hop.yaml`
- Score direction: higher_is_better (distance); target 0 (asymmetry)
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

Hopper test (Noyes 1991). Asymmetry > 10% commonly used as a flag — but the AI summary must NOT make injury claims.
