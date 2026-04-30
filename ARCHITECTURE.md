# Architecture

## Pipeline (the only story you need to remember)

```
   ┌──────────┐   ┌──────────────┐   ┌─────────┐   ┌──────────┐   ┌─────────────┐
   │  Video   │──▶│ Core CV      │──▶│ Metrics │──▶│ Scoring  │──▶│ AI Summary  │
   │  +       │   │ Detection    │   │ (per    │   │ (0–100   │   │ (OpenAI)    │
   │  Athlete │   │ Tracking     │   │ test)   │   │ vs M/F   │   │             │
   │  Profile │   │ Pose         │   │         │   │ benchmk) │   │             │
   └──────────┘   └──────────────┘   └─────────┘   └──────────┘   └─────────────┘
                          │                                             │
                          ▼                                             ▼
                  ┌──────────────┐                              ┌─────────────┐
                  │  Annotated   │                              │  Final JSON │
                  │  Video       │                              │  Report     │
                  └──────────────┘                              └─────────────┘
```

Every test follows this exact flow. A test pipeline is just a recipe specifying:
which detectors to run, which metrics to compute, what to draw on the video,
which benchmark file to score against, and which AI summary prompt to use.

## Module boundaries (hard rules)

### `src/core/`
Generic computer vision and video utilities. **Knows nothing about any specific test.**
- `models/` — registry-driven wrappers around YOLO26, YOLO26-pose, RTMPose-x (biomech backend), and ByteTrack
- `detection/` — domain-specific detectors built on those models (ball, cone, player, touch event, segment crossing)
- `tracking/` — multi-object trackers, velocity tracker, ball possession state machine
- `pose/` — pluggable pose estimator (factory by registry key); joint-angle / biomechanical derivations
- `calibration/` — pixel ↔ metre conversion via cone or known-marker calibration
- `annotation/` — overlay primitives (skeletons, boxes, gates, HUD ticker)
- `utils/` — geometry, video I/O, signal smoothing

### `src/metrics/`
Pure functions. Input: time-series arrays + frame metadata. Output: a single named
metric value with units. **No I/O, no model loading, no test-specific logic.**
Grouped by category (motion, ball, jump, biomech, throw, endurance, cognitive).

### `src/tests/`
One module per test, grouped by family. Each test file:
1. Subclasses a family base class (`SprintFamily`, `JumpFamily`, etc.)
2. Specifies its required detectors, gate geometry, expected duration
3. Runs the pipeline and produces a result dict matching the schema in `docs/api/API_SPEC.md`

Family base classes live in `src/tests/families/` and centralise the boilerplate
that every sprint test (or every jump test) shares.

### `src/scoring/`
- `benchmarks.py` — loads YAML files from `benchmarks/`
- `normalization.py` — converts a raw metric to a 0–100 score given the relevant benchmark distribution
- `grade.py` — maps 0–100 to letter grades / percentile bands for the report

### `src/ai_summary/`
Calls OpenAI (`gpt-5-mini`). Inputs: the metrics JSON + scored JSON + athlete
profile. Outputs: the coach-facing summary block in the final report. Prompts
are versioned, per-family templates in `src/ai_summary/templates/`.

### `src/api/` and `src/ui/`
Thin layers. The API exposes test pipelines as endpoints; the Streamlit app
exposes them as a tabbed UI. Neither should contain analysis logic.

## Data flow inside a single test run

1. **Ingest** — receive video path + `AthleteProfile` (gender, age, optional fitness band).
2. **Calibrate** — run camera calibration if cones / known markers are visible. Fail loudly if the test requires real-world units and calibration could not be established.
3. **Detect** — run the detector set declared by the test (always pose; usually player+ball+cone subset).
4. **Track** — assign stable IDs across frames; build per-frame state buffers.
5. **Compute** — call each metric function declared by the test, with the buffered state as input.
6. **Annotate** — render the annotated video frame-by-frame using overlays from `src/core/annotation/`. The HUD ticker reads live metrics computed during the same pass when possible (single video read).
7. **Score** — for each metric, look up the (gender, age band) benchmark and normalise to 0–100. Aggregate to a test-level score per the test's spec.
8. **Summarise** — send metrics + scores + profile to OpenAI (`gpt-5-mini`) with the test's family prompt template; receive natural-language summary.
9. **Return** — `AnalysisResult` containing metrics JSON, scores JSON, summary text, and the path to the annotated video.

## Test families and what they share

| Family | Shared primitives | Tests |
|---|---|---|
| Sprint | gates, splits, peak accel, max speed | Linear Sprint · 5×10 COD · Repeated Sprint Ability · Bangsbo (7×34.2m) |
| Agility | cone layout, COD detection, completion time, splits, hurdle clearance | T-Test · Illinois · 45-Second Agility Hurdle Jump |
| Jump | flight time, contact time, pose at takeoff/landing (uses `pose_biomech`) | CMJ · Drop Jump · Squat Jump · Standing Long Jump |
| Dribble | touches, ball–foot distance, completion time | Straight Line · Zig-Zag · Figure of 8 |
| Endurance | distance covered, pacing, stage reached, audio-beep alignment | Yo-Yo (IR2) · Multistage |
| Throw | release pose, ball trajectory, distance | Medicine Ball Throw |
| Skill | discrete action counting, accuracy zones, ball control | Juggling · Foot Tapping · Wall Pass |
| Mobility / Posture | static / quasi-static joint angles (uses `pose_biomech`) | LESS (subset score) |

The CV pipeline covers eight families across 21 tests. Tests deferred from
v1 (awaiting data, or psychological/cognitive games shipping separately):
30-15 Intermittent, Cooper, DFB Agility, Hurdle Agility Run (replaced by
45-Second variant), Incremental Ramp, Single-Leg Hop, Sit-and-Reach,
Stepwise Core Stability, DFB Shooting, Reaction Time, Pattern Recognition,
Video-Based Decision-Making.

## Concurrency model

A single video runs through the pipeline serially — the bottleneck is the GPU
inference, not Python. The API parallelises **across requests** with a worker
pool; do not try to multi-thread within a single test pipeline.

## What lives where: a sanity-check rule

If a piece of code references "5×10 metres" or "8 cones in a T shape" or
"3 attempts at 5kg medicine ball," it belongs in `src/tests/`. Everything else
belongs in `src/core/` or `src/metrics/`.
