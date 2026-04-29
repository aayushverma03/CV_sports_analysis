# Test Spec Template

> Every file under `docs/tests/<domain>/` must follow this template. Claude
> Code reads these specs to implement the corresponding pipeline in
> `src/tests/<domain>/<test_id>.py`.

---

# `<test-id>` — `<Test Display Name>`

**Domain**: physical | technical | cognitive
**Family**: sprint | agility | jump | dribbling | endurance | throw | skill | mobility | cognitive
**Status**: spec | wip | implemented

## 1. Purpose

What this test measures, in one paragraph. Cite the protocol source (DFB,
Bangsbo, FIFA, ACSM, etc.) where applicable.

## 2. Equipment & setup

- Field markings, cone layout, distances (with diagram if non-trivial)
- Camera placement (single fixed camera assumed unless noted)
- Required visible references for calibration (cones at known spacing, line markings)

## 3. Protocol (what the athlete does)

Numbered steps. Be precise about start/end conditions — these define the
detection events the pipeline must catch.

## 4. CV pipeline requirements

| Capability | Required? | Notes |
|---|---|---|
| Player detection + tracking | yes / no | |
| Pose estimation | yes / no | which keypoints matter |
| Ball detection + tracking | yes / no | |
| Cone detection | yes / no | how many, what shape |
| Calibration | mandatory / optional | what reference is used |
| Event detection | list events (e.g. "foot crosses start gate", "ball–foot contact") | |

## 5. Metrics produced

| Metric ID | Module path | Unit | Description |
|---|---|---|---|
| `total_completion_time` | `src/metrics/motion/total_completion_time.py` | s | Time from start trigger to finish trigger |
| ... | ... | ... | ... |

## 6. Annotation requirements

Per-frame overlays the annotated video must include. Reference the primitives
in `docs/annotation/VIDEO_ANNOTATION_SPEC.md`.

- Skeleton: yes / no
- Bounding box on athlete: yes / no
- Gates / markers visualised: yes / no
- Ball trail: yes / no
- HUD ticker fields (live during playback): list metric IDs
- End-card content: which scores, which highlights

## 7. Benchmark schema

- **Benchmark file**: `benchmarks/<domain>/<test-id>.yaml`
- **Score direction**: `lower_is_better` (e.g. sprint times) | `higher_is_better` (e.g. jump heights, distances) | `target_value` (e.g. mobility ranges)
- **Axes**: gender × age band (default age bands: U12, U14, U16, U18, U23, Senior)
- **Test-level aggregation**: how the per-metric 0–100 scores combine into a single test score (mean, weighted mean with weights, max, etc.)

## 8. Failure modes & validation

- What raises `CalibrationError` for this test
- What raises `ProtocolError` (video too short, missing cones, wrong start state)
- Pose-confidence thresholds below which a frame is dropped from metric computation
- Acceptable detection drop-out (% frames without player detection before aborting)

## 9. AI summary prompt notes

Pointers for `src/ai_summary/templates/<family>.md`:
- Domain-specific language to use (e.g. "first-step quickness" for sprints, "eccentric load tolerance" for drop jumps)
- Cross-metric narratives the model should attempt (e.g. "if peak deceleration is high but max speed is low, flag braking-limited rather than acceleration-limited")
- What NOT to claim (no medical diagnoses, no injury risk pronouncements)

## 10. Acceptance criteria

A test pipeline is "done" when:
- [ ] All metrics in §5 are produced for a sample video
- [ ] All overlays in §6 render correctly
- [ ] Score normalisation against the benchmark file produces sane 0–100 outputs
- [ ] Unit tests cover metric calculations
- [ ] An integration test runs end-to-end on a sample video
- [ ] Annotated video plays back without artifacts

## 11. References

Links to the canonical protocol source, validation studies, similar implementations.
