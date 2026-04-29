# `video-based-decision-making` — Video-Based Decision-Making

**Domain**: cognitive
**Family**: cognitive
**Status**: spec

## 1. Purpose

Athlete watches game-situation video clips and makes a tactical decision under time pressure. Measures contextual decision-making, not pure reaction.

## 2. Equipment & setup

Stimulus video clips (game situations). Response capture (selection or verbal).

## 3. Protocol

1. Clip plays up to a decision point (occluded at frame N)
2. Athlete decides next action from given options (or describes it)
3. Multiple clips covering varied scenarios

## 4. CV pipeline requirements

| Capability | Required? | Notes |
|---|---|---|
| Player detection + tracking | yes | |
| Pose estimation | no | |
| Ball detection + tracking | no | |
| Cone detection | no | |
| Calibration | n/a | |
| Event detection | — | clip start, occlusion frame, response event |

## 5. Metrics produced

| Metric ID | Module | Unit |
|---|---|---|
| `response_accuracy` | `metrics/cognitive/response_accuracy.py` | % correct (vs expert-rated answer) |
| `decision_latency` | `metrics/cognitive/decision_latency.py` | s from occlusion to response |
| `scenario_difficulty_score` | `metrics/cognitive/scenario_difficulty_score.py` | weighted by clip difficulty rating |

## 6. Annotation requirements

- Skeleton: as appropriate to the test (see family default in `docs/annotation/VIDEO_ANNOTATION_SPEC.md`)
- Bounding box: yes
- Markers / gates / target zones: yes (test-specific elements above)
- HUD ticker fields: `clip_number`, `last_decision_time`, `accuracy_running`
- End-card: all metrics + test score

## 7. Benchmark schema

- File: `benchmarks/cognitive/video-based-decision-making.yaml`
- Score direction: higher_is_better (accuracy, weighted score); lower_is_better (latency)
- Aggregation: see benchmark file (`weighted_mean` default unless otherwise noted)

## 8. Failure modes

- Missing required entities (cones / ball / wall / box) → `DetectionError` or `CalibrationError`
- Protocol mismatches (duration, sequence) → `ProtocolError`
- Pose confidence below threshold on > 30% of frames → flag, do not abort

## 9. AI summary prompt notes

Use the family template `src/ai_summary/templates/cognitive.md`. See its
cross-metric narrative section for patterns to surface for this test.

## 10. Acceptance criteria

- [ ] All §5 metrics computed on a sample video
- [ ] All §6 overlays render
- [ ] Score normalisation produces sane 0–100 outputs
- [ ] Unit tests cover metric calculations
- [ ] Integration test passes end-to-end

## 11. References

Each clip has an expert-rated 'best decision' in the test config. Difficulty weighting per clip is part of the configuration data, not the code.
