# `landing-error-scoring-system` — Landing Error Scoring System (LESS)

**Domain**: physical
**Family**: mobility
**Status**: active (v1 ships a SUBSET score — single-camera reliably covers ~8 of 17 LESS items, enough for the Mobility & Stability subarea contribution. Document which items we score in this spec as Phase 4 work proceeds)

## 1. Purpose

Validated 17-item biomechanical scoring of jump-landing technique. Identifies movement patterns associated with elevated injury risk in research literature.

## 2. Equipment & setup

30 cm box for drop, target line at 50% body height in front. Side-on AND frontal cameras strongly preferred (frontal-plane scoring items).

## 3. Protocol

1. Athlete drops from 30 cm box
2. Lands and immediately jumps for max height
3. Three trials, scored independently and averaged

## 4. CV pipeline requirements

| Capability | Required? | Notes |
|---|---|---|
| Player detection + tracking | yes | |
| Pose estimation | yes | |
| Ball detection + tracking | no | |
| Cone detection | no | |
| Calibration | optional (relative angles only) | |
| Event detection | — | drop, ground contact, peak, second landing |

## 5. Metrics produced

| Metric ID | Module | Unit |
|---|---|---|
| `less_total_score` | `metrics/biomech/less_total_score.py` | 0–17 (lower = better technique) |
| `knee_valgus_left` | `metrics/biomech/knee_valgus_left.py` | ° max |
| `knee_valgus_right` | `metrics/biomech/knee_valgus_right.py` | ° max |
| `trunk_flexion_at_landing` | `metrics/biomech/trunk_flexion_at_landing.py` | ° |
| `ankle_dorsiflexion_at_landing` | `metrics/biomech/ankle_dorsiflexion_at_landing.py` | ° |
| `landing_symmetry` | `metrics/biomech/landing_symmetry.py` | % |

## 6. Annotation requirements

- Skeleton: as appropriate to the test (see family default in `docs/annotation/VIDEO_ANNOTATION_SPEC.md`)
- Bounding box: yes
- Markers / gates / target zones: yes (test-specific elements above)
- HUD ticker fields: `less_running_score`, `trial_number`
- End-card: all metrics + test score

## 7. Benchmark schema

- File: `benchmarks/physical/landing-error-scoring-system.yaml`
- Score direction: lower_is_better (LESS total); target ranges for individual items
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

Padua et al. (2009). 17-item rubric implemented item-by-item. AI summary MUST NOT diagnose injury risk — describe movement quality only.
