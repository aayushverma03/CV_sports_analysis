# Implementation Plan

End-to-end plan for building the cv_sports_analysis platform. Read after
`CLAUDE.md`, `ARCHITECTURE.md`, `ROADMAP.md`, `CONVENTIONS.md`. This plan
complements `ROADMAP.md` — the phase structure is the same, but this file
adds locked decisions, design notes, and a session-by-session checklist
sized for solo, strictly-sequential work.

## 1. Success criteria

The program is complete when:

1. All **21 CV-pipeline tests** (15 Physical + 5 Technical) produce,
   end-to-end:
   - an annotated `.mp4` (skeleton, bbox, gates/markers, HUD ticker, end-card)
   - a metrics JSON validated against `docs/api/API_SPEC.md`
   - a 0–100 scored report against gender + age-band benchmarks
   - a coach-facing AI summary
2. Each test has at least one sample video in `data/` and an integration
   test that runs without manual setup.
3. Where v1 results are inaccurate, the retrain + logic-change loop
   (Phase 8) is in place: failures get labelled, the relevant detector or
   metric is iterated, the registry version is bumped.
4. A solo developer can run any one test with `scripts/run_test.py` and
   reproduce the result.

## 2. Locked decisions (departures from existing docs)

| Topic | Existing doc says | Plan locks in | Why |
|---|---|---|---|
| Detection model | YOLO11x | **YOLO26** (Ultralytics, Sep 2025) | ~43% faster CPU inference, NMS-free, better small-object detection (cones, distant balls), drop-in API |
| Pose model (default) | YOLO11-pose | **YOLO26-pose** | Drop-in API, fast, single forward pass with detection — fine for sprint, agility, dribbling, endurance |
| Pose model (biomech) | — | **RTMPose-x via ONNX Runtime** registered as `pose_biomech` | Higher keypoint AP than YOLO-pose; used by jump family and Medicine Ball Throw where joint-angle precision matters. ONNX deploy avoids the mmpose/mmcv install pain (no prebuilt wheels for torch 2.11 / Apple Silicon) |
| AI summary provider | Anthropic Claude | **OpenAI `gpt-5-mini`** | User decision; one model for all summaries |
| Capacity model | "items can be parallelised inside a phase" | **Solo, strictly sequential** | User decision |
| Object-training data | (unspecified) | User-provided test videos under `data/<domain>/<test>/` + Roboflow labelling | User decision |
| Benchmarks ingestion | "author YAML files" | User-provided **CSV → YAML converter** in `scripts/` | User will supply CSV; converter produces canonical YAML |
| Test scope | 32 tests | **21 CV-pipeline tests** (15 Physical + 5 Technical) | Drop the 3 cognitive tests (in-app games), the 8 tests with no sample video, and LESS (single-camera scoring is partial). Deferred list documented in `ROADMAP.md` |

The following files reference the old decisions and must be edited as part
of session 0.1: `README.md` (model bumps + scope 32 → 21 CV tests + dropped
tests moved to deferred), `ARCHITECTURE.md` (family list drops Cognitive and
Mobility), `CLAUDE.md`, `docs/models/MODEL_REGISTRY.md` (add `pose_biomech`
RTMPose-ONNX entry alongside YOLO26-pose), `docs/ai_summary/AI_SUMMARY_SPEC.md`,
`pyproject.toml`, `requirements.txt` (add `onnxruntime`).

## 3. Tech stack (post-decisions)

- **Python**: 3.11+, managed with `uv` (always `uv run`, `uv add`)
- **Detection**: Ultralytics YOLO26 (n / s / m variants registered)
- **Pose**: Ultralytics YOLO26-pose (default); **RTMPose-x** for biomech-heavy tests (jump family, Medicine Ball Throw)
- **Tracking**: ByteTrack (Ultralytics native integration)
- **Calibration**: cone-based + known-marker
- **Video / math**: OpenCV, NumPy, SciPy, filterpy
- **Labelling**: Roboflow (custom-class fine-tuning datasets)
- **API**: FastAPI + RQ (worker queue choice deferred — see open Q1)
- **UI**: Streamlit
- **AI summary**: OpenAI `gpt-5-mini` via the `openai` SDK
- **Secrets**: `.env` loaded via `python-dotenv` at entrypoints only

## 4. Phase dependency graph

```
Phase 0   Foundations
   |
   +----> Phase 0.5  Object training setup  (data-only, runs alongside)
   |
   v
Phase 1   Scoring spine
   |
   v
Phase 2   Metric library
   |
   v
Phase 3   Test family base classes
   |
   v
Phase 4   21 test pipelines (one at a time)
   |
   v
Phase 5   Annotation polish
   |
   v
Phase 6   AI summary layer
   |
   v
Phase 7   API + UI surfaces
   |
   v
Phase 8   Hardening + retrain loop
```

Even where a phase could overlap with another, this plan is sequential.
Arrows are precedence, not parallelism.

---

## 5. Phase 0 — Foundations

**Goal.** Stand up the CV primitives every test depends on.

**Exit criteria.** A 5-second clip can be loaded, a player detected and
tracked across frames, pose extracted, calibration established from cones,
and a debug overlay rendered — all from one CLI invocation.

### Design notes

- `src/core/models/registry.py` is the single source of model paths. Sketch:
  ```python
  MODELS = {
      "detector_default":   ModelSpec("yolo26m.pt",      version="26.0.0", backend="ultralytics"),
      "pose_default":       ModelSpec("yolo26m-pose.pt", version="26.0.0", backend="ultralytics"),
      "pose_biomech":       ModelSpec("rtmpose-x.onnx",  version="1.0.0",  backend="onnx"),
      "detector_cones_v1":  ModelSpec("custom/cones_v1.pt", version="1.0.0", backend="ultralytics"),
  }
  ```
  Loading is lazy and cached per process. Hard rule #4: never hardcode
  model paths anywhere else.
- `src/core/utils/video_io.py` exposes a `frame_iter(path)` generator that
  yields `(frame_idx, frame_ndarray, timestamp_ms)`. Hard rule #3: no module
  re-opens the file.
- `src/core/calibration/camera_calibration.py` raises `CalibrationError` if
  pixel-to-metre ratio cannot be established. No silent fallback.

### Sessions

| # | Session | Spec to read | Deliverable |
|---|---|---|---|
| 0.1 | Repo housekeeping: bump model + AI provider in docs | this plan, section 2 | edits to `README.md`, `ARCHITECTURE.md`, `CLAUDE.md`, `docs/models/MODEL_REGISTRY.md`, `docs/ai_summary/AI_SUMMARY_SPEC.md`, `pyproject.toml`, `requirements.txt` |
| 0.2 | `uv` project + base deps | `pyproject.toml` | `uv add ultralytics opencv-python numpy scipy filterpy openai python-dotenv pyyaml onnxruntime`; dev: `uv add --dev ruff pytest pytest-cov`. RTMPose deploys via ONNX (no mmpose/mmcv) — avoids prebuilt-wheel issues on Apple Silicon and torch 2.11 |
| 0.3 | `scripts/download_models.py` | `docs/models/MODEL_REGISTRY.md` | downloads YOLO26, YOLO26-pose (Ultralytics auto-fetch), and `rtmpose-x.onnx` (direct URL from OpenMMLab releases) into `models/` |
| 0.4 | `src/core/models/registry.py` | `docs/models/MODEL_REGISTRY.md` | registry + lazy loaders + version pinning; `backend` field dispatches between Ultralytics and ONNX Runtime loaders |
| 0.5 | `src/core/utils/video_io.py` | `ARCHITECTURE.md` (single-pass rule) | `frame_iter`, FPS handling, codec sanity |
| 0.6 | `src/core/utils/geometry.py` | `CONVENTIONS.md` | px <-> world helpers, angle math |
| 0.7 | `src/core/utils/smoothing.py` | — | Savitzky-Golay, Kalman wrappers |
| 0.8 | `src/core/calibration/camera_calibration.py` | — | cone + known-marker calibration; raises `CalibrationError` |
| 0.9 | `src/core/detection/player_detector.py` | — | wraps registry detector with sane defaults |
| 0.10 | `src/core/tracking/bytetrack_tracker.py` | — | tracker with stable IDs |
| 0.11 | `src/core/pose/estimator.py` | — | pluggable pose estimator: factory selects YOLO-pose or RTMPose by registry key; uniform confidence-aware joint access |
| 0.12 | `src/core/annotation/overlays.py` | `docs/annotation/VIDEO_ANNOTATION_SPEC.md` | skeleton, bbox, gate line, HUD ticker primitives |
| 0.13 | Smoke test: end-to-end debug overlay on a sample clip | — | confirms exit criteria |

---

## 6. Phase 0.5 — Object training setup (data-only, runs alongside Phase 0)

**Goal.** Stand up the labelling + training loop so custom classes (cones,
hurdles, agility markers, medicine balls, target zones, foot-tap mats)
exist as detectors before they are needed by tests.

**Why now, not at Phase 8.** Several Phase 4 tests (every agility test,
every cone-based sprint) fail without a working cone detector. Setting
this up early lets labelling proceed in parallel while Phase 0 code is
being written.

### Recommended approach (three-tier source hierarchy)

For each custom class, try sources in order. Stop at the first one that
passes a held-out smoke test on a clip from `data/`.

**Tier 1 — Pretrained community model from Roboflow Universe.** Many
sports objects (traffic cones, hurdles, agility markers, medicine balls,
soccer balls) already have community-trained `.pt` weights. If one passes
a smoke test on your data, register it and skip training entirely.

**Tier 2 — Fine-tune YOLO26 on frames extracted from your own videos.**
Primary path when Tier 1 is unavailable or insufficient. Same camera,
same lighting, same domain — beats web-scraped data.

**Tier 3 — Supplementary reference images (optional).** Hand-collected or
web-scraped images held in `data/_labelling/reference_images/<class>/`.
Used only to *augment* a Tier 2 base when a class is rare in your videos.
**Must be labelled with bboxes in Roboflow before training** — folder name
alone is not a label (folder-per-class is the classification format, not
the detection format YOLO needs).

### Workflow

1. **Inventory custom classes.** List every non-COCO object across the 32
   tests. Build one multi-class model, not 32 single-class models.
2. **Roboflow Universe search.** For each class, search and evaluate any
   pretrained models. Record outcomes in `docs/models/COMMUNITY_MODELS.md`.
3. **Frame extraction.** 1 fps from `data/` into
   `data/_labelling/extracted_frames/`.
4. **Labelling in Roboflow.** Target 200–500 labelled instances per class
   for v1. Upload extracted frames + any Tier 3 reference images.
5. **Train.** Ultralytics CLI: `yolo detect train data=cones.yaml
   model=yolo26n.pt epochs=100 imgsz=640`. Start with `yolo26n` for
   iteration speed; bump to `s` / `m` only if mAP is short.
6. **Register.** Every fine-tuned `.pt` gets an entry in
   `src/core/models/registry.py`. Old versions stay forever.
7. **Active-learning loop.** Once Phase 4 tests run, the model surfaces
   low-confidence detections. Label only those, retrain. That becomes
   Phase 8's cadence.

### Sessions

| # | Session | Deliverable |
|---|---|---|
| 0.5.1 | Inventory custom classes from all 21 CV tests | `docs/models/CUSTOM_CLASSES.md` listing every non-COCO object |
| 0.5.2 | Roboflow Universe search per class | `docs/models/COMMUNITY_MODELS.md` recording per-class result; registry entries for any community model that passes a smoke test on a held-out clip from `data/` |
| 0.5.3 | `scripts/extract_frames.py` | samples 1 fps from `data/` into `data/_labelling/extracted_frames/` |
| 0.5.4 | Roboflow project setup (manual) for classes not covered by Tier 1 | one project per class group; export format = YOLO; ingest extracted frames + any Tier 3 reference images present in `data/_labelling/reference_images/<class>/` |
| 0.5.5 | `scripts/train_yolo.py` | wraps `yolo detect train`; reads config from YAML; writes outputs to `models/custom/<class>_v<N>/` |
| 0.5.6 | First fine-tune: cones (highest-value class, used by 11+ tests) — only if Tier 1 didn't yield a usable model | `cones_v1.pt`; registry entry; held-out mAP report |
| 0.5.7 | Smoke test: detectors on unseen clips | manual visual check per class |

This phase loops back in Phase 8.

---

## 7. Phase 1 — Scoring spine

**Goal.** Convert any raw metric value to a 0–100 score given a benchmark
and an `AthleteProfile`.

**Why first after Phase 0.** Pure logic, no detection dependency. Can be
implemented and tested in isolation, and unblocks unit testing of metrics
in Phase 2.

### Design notes

- Benchmarks are YAML only. The CSV the user supplies is converted by
  `scripts/csv_to_benchmark_yaml.py`, not loaded directly. Hard rule #5.
- Lookup key: `(test_id, metric_id, gender, age_band)` ->
  `BenchmarkLookupError` if missing.
- Four normalisation modes per `docs/scoring/NORMALIZATION.md` — that doc
  is authoritative.

### Sessions

| # | Session | Deliverable |
|---|---|---|
| 1.1 | `scripts/csv_to_benchmark_yaml.py` | takes user CSV, emits YAMLs into `benchmarks/<domain>/` |
| 1.2 | `src/scoring/benchmarks.py` | loader + cached lookup |
| 1.3 | `src/scoring/normalization.py` | the four scoring modes |
| 1.4 | `src/scoring/grade.py` | letter / band mapping |
| 1.5 | `tests/unit/scoring/` | full coverage; the layer that has to be bulletproof |
| 1.6 | Author 3–5 real benchmark YAMLs (linear sprint, CMJ, sit-and-reach) | populates `benchmarks/` enough for Phase 2 unit tests |

---

## 8. Phase 2 — Metric library

**Goal.** Pure functions for every metric the 21 CV tests need.

### Design notes

- One metric per file. File name = metric ID. No file I/O, no model
  loading, no logging side effects (hard rule #2). Numpydoc docstring with
  units. Unit test mirrors source path.
- Implement in dependency order so each batch unlocks tests.

### Sessions (one row = one session = one metric + its unit tests)

**Motion (unlocks all sprint, agility, endurance):**
2.1 `total_completion_time` · 2.2 `split_segment_times` · 2.3 `total_distance` · 2.4 `average_speed` · 2.5 `max_speed` · 2.6 `peak_acceleration` · 2.7 `peak_deceleration`

**Jump (unlocks CMJ, Drop Jump, Squat Jump, Standing Long Jump):**
2.8 `jump_height_flight_time` · 2.9 `ground_contact_time` · 2.10 `jump_height_rebound` · 2.11 `reactive_strength_index`

**Ball (unlocks dribbling, juggling, passing):**
2.12 `touches_per_metre` · 2.13 `ball_foot_distance` · 2.14 `max_consecutive_touches` · 2.15 `pass_velocity` · 2.16 `passing_accuracy`

**Throw, biomech, endurance, cognitive metrics**: implement just-in-time
when the corresponding family is reached in Phase 4. Not pre-built.

---

## 9. Phase 3 — Test family base classes

**Goal.** Centralise boilerplate shared across tests in the same family
so each Phase 4 session is small.

### Design notes

- `src/tests/base.py` defines `BaseTest` (ABC), `AthleteProfile` (dataclass:
  gender, age, optional fitness band — required, hard rule #8),
  `AnalysisResult` (metrics, scores, summary, annotated_video_path).
- Family base classes own: which detectors to load, the skeleton of
  `analyze()`, shared annotation overlays (gate lines for sprint, cone
  path for agility, etc.).

### Sessions

| # | Session | Deliverable |
|---|---|---|
| 3.1 | `src/tests/base.py` | `BaseTest`, `AthleteProfile`, `AnalysisResult` |
| 3.2 | `src/tests/families/sprint_family.py` | gates, splits, peak accel, max speed shared logic |
| 3.3 | `src/tests/families/agility_family.py` | cone layout, COD detection |
| 3.4 | `src/tests/families/jump_family.py` | flight/contact time, takeoff/landing pose; loads `pose_biomech` (RTMPose). Shared `detect_movement_onset(pose_series)` helper — tracks hip-midpoint y-velocity, returns the frame at which the test "really starts" (countermovement). All jump metrics and the HUD elapsed-time clock anchor to this event, not video frame 0. Per-test specs in §3 declare the exact onset signal. |
| 3.5 | `src/tests/families/dribbling_family.py` | touches, ball-foot distance, completion time |
| 3.6 | `src/tests/families/endurance_family.py` | distance, pacing, stage; audio-beep alignment + visual shuttle counting |
| 3.7 | `src/tests/families/throw_family.py` | release biomech, projectile trajectory, distance; loads `pose_biomech` |
| 3.8 | `src/tests/families/skill_family.py` | discrete action counting, ball control, accuracy zones |
| 3.9 | `src/tests/families/mobility_family.py` | LESS subset scoring (~8 of 17 items single-camera); loads `pose_biomech` |

---

## 10. Phase 4 — 21 test pipelines (one per session)

**Goal.** End-to-end implementation of each test.

### Per-session checklist (applies to every row below)

1. Author the test spec at `docs/tests/<domain>/<test>.md` from
   `TEST_SPEC_TEMPLATE.md`, using the corresponding sample video as ground
   truth.
2. Extend the family base class if a primitive is missing.
3. Implement `src/tests/<domain>/<test>.py`.
4. Add the benchmark YAML if not already present.
5. Write an integration test using the sample video in `data/`.
6. Verify the annotated `.mp4` is produced (hard rule #6).
7. Commit (`tests/<family>: implement <test>`).

### Sessions, in execution order

**Quick wins:** 4.1 Linear Sprint · 4.2 Counter Movement Jump · 4.3 Drop Jump · 4.4 Straight Line Dribbling · 4.5 Juggling · 4.6 Foot Tapping

**Agility family:** 4.7 T-Test · 4.8 Illinois Agility · 4.9 45-Second Agility Hurdle Jump · 4.10 5×10m Sprint with COD

**Jump family:** 4.11 Squat Jump · 4.12 Standing Long Jump

**Dribbling family:** 4.13 Zig-Zag Dribbling · 4.14 Figure of 8 Dribbling

**Endurance family:** 4.15 Yo-Yo Intermittent (IR2) · 4.16 Bangsbo Sprint (7×34.2m) · 4.17 Multistage Fitness · 4.18 Repeated Sprint Ability

**Skill:** 4.19 Wall Pass

**Throw:** 4.20 Medicine Ball Throw

**Mobility / Posture:** 4.21 LESS (subset score)

**Out of v1 scope** (deferred):
30-15 Intermittent, Cooper, DFB Agility, Hurdle Agility Run (replaced by 45-Second variant), Incremental Ramp, Single-Leg Hop, Sit-and-Reach, Stepwise Core Stability, DFB Shooting, Reaction Time, Pattern Recognition, Video-Based Decision-Making.

---

## 11. Phase 5 — Annotation polish

**Goal.** Annotated outputs look consistent, readable, colour-blind safe,
informative.

### Sessions

| # | Session | Deliverable |
|---|---|---|
| 5.1 | HUD ticker design pass | typography, palette, position, documented in `docs/annotation/VIDEO_ANNOTATION_SPEC.md` |
| 5.2 | End-card layout | scores grid + headline summary; `src/core/annotation/end_card.py` |
| 5.3 | Per-test overlay variations | gates for sprints, cone path for agility, ball trail for dribbling, range-of-motion arcs for sit-and-reach |

---

## 12. Phase 6 — AI summary layer (OpenAI)

**Goal.** Coach-facing natural-language summary for every test.

### Design notes

- `src/ai_summary/summarizer.py` wraps the `openai` SDK with retry +
  timeout + cost log line.
- One prompt template **per family**. v1 CV scope has **8 families**:
  Sprint, Agility, Jump, Dribbling, Endurance, Throw, Skill, Mobility/Posture
  (LESS only, subset score). Templates live at
  `src/ai_summary/templates/<family>.txt` with few-shot examples.
- Inputs: metrics JSON, scored JSON, athlete profile. Output: <= 300-word
  summary string.
- API key loaded from `.env` via `python-dotenv` at entrypoints only
  (`scripts/run_test.py`, `src/api/main.py`, `src/ui/streamlit_app.py`).
  Library code never reads `.env` directly.
- Mock the OpenAI client in unit tests; never hit the network in CI.

### Sessions

| # | Session | Deliverable |
|---|---|---|
| 6.1 | Update `docs/ai_summary/AI_SUMMARY_SPEC.md` for OpenAI | spec doc reflecting `gpt-5-mini` + family-template structure |
| 6.2 | `src/ai_summary/summarizer.py` | OpenAI client w/ retries; `gpt-5-mini` |
| 6.3 | Family prompt templates | one `.txt` per family in `src/ai_summary/templates/` |
| 6.4 | Wire summary call into every test pipeline | call sites in `src/tests/families/*` |
| 6.5 | Unit tests with mocked client | `tests/unit/ai_summary/` |

---

## 13. Phase 7 — Surfaces (API + UI)

**Goal.** Make the platform usable by non-developers.

### Sessions

| # | Session | Deliverable |
|---|---|---|
| 7.1 | `scripts/run_test.py` | CLI: `--test`, `--video`, `--athlete-gender`, `--athlete-age`, `--out` |
| 7.2 | `src/api/main.py` | FastAPI scaffold + health endpoint |
| 7.3 | `src/api/routes/analyze.py` | submit job, poll for result, schema per `docs/api/API_SPEC.md` |
| 7.4 | `src/api/workers.py` | RQ-based background worker (commit RQ-vs-Celery decision in this session) |
| 7.5 | `src/ui/streamlit_app.py` | tabbed app, one tab per test, embedded annotated video, scores grid, summary panel |

---

## 14. Phase 8 — Hardening + retrain loop

**Goal.** Inaccurate v1 results get fixed via a documented loop, not by
hacks.

### Failure-triage rule (the loop)

When a test produces a clearly wrong metric:

1. **Detection problem** (object missed / mislabelled). Label more frames
   in Roboflow, retrain the relevant detector, bump the registry version.
   No code change in `src/`.
2. **Logic problem** (metric formula wrong, gate trigger off-by-one). Fix
   in `src/metrics/` or `src/tests/`. Add a unit test that captures the
   failure case.
3. **Calibration problem.** Improve calibration robustness in
   `src/core/calibration/`. Never paper over with pixel-space heuristics.

Every retrain produces a new versioned weight in `models/custom/` and a
new registry entry. Old versions stay; we never overwrite.

### Sessions

| # | Session | Deliverable |
|---|---|---|
| 8.1 | Integration tests for every test using `data/` videos | `tests/integration/`; auto-skip if `data/` empty |
| 8.2 | Performance pass | profile a representative video; eliminate any double video reads (hard rule #3) |
| 8.3 | Per-test acceptance criteria | FPS minimum, max processing time, written into each `docs/tests/<domain>/<test>.md` |
| 8.4 | Documentation review across all 32 specs | consistency pass |
| 8.5 | Retrain runbook | `docs/models/RETRAIN_RUNBOOK.md` capturing the failure-triage rule |
| 8.6 | Retrain v2 of cones (most-used custom detector) using Phase 4 failures | `cones_v2.pt`; before/after mAP |

---

## 15. Cross-cutting setup (do once, in session 0.1)

- **`.env`** — gitignored; `OPENAI_API_KEY`, `ROBOFLOW_API_KEY`. Loaded via
  `python-dotenv` at entrypoints only.
- **`data/` layout** — matches the user's existing folder structure:
  ```
  data/
  |- 01. Physical Capabilities/<test_id>/*.mp4
  |- 02. Technical Skills/<test_id>/*.mp4
  +- 03. Psychological & Cognitive/<test_id>/*.mp4
  ```
  CLI accepts the path; tests look up the matching folder by test ID.
  Folder names contain spaces and a period — fine for storage, never
  imported as Python packages.
- **Per-test video convention.** A test folder may contain 1+ videos with
  any filenames. Two consumers, one rule:
  - **Integration test** (`tests/integration/`): glob `*.mp4` in the test
    folder, sort alphabetically, take the first. No `sample.mp4` rename
    required. To pin a specific clip as the integration input, prefix it
    `00_` or `a_`.
  - **Frame extraction** (`scripts/extract_frames.py`): walks the whole
    `data/` tree (excluding `_labelling/`) and ingests every `.mp4` for
    the training pool. Adding extra clips improves training without
    affecting integration results.
- **`data/_labelling/` layout** — derived data for object training.
  Underscore prefix sorts it separately and signals "not source video":
  ```
  data/_labelling/
  |- extracted_frames/                ← output of scripts/extract_frames.py
  +- reference_images/                ← Tier 3 supplement (optional)
     |- cones/*.jpg
     |- hurdles/*.jpg
     +- medicine_balls/*.jpg
  ```
  `reference_images/<class>/` is a bookkeeping convention for the human
  collecting images; the class-name subfolder is **not** a label. Every
  image in there must be uploaded to Roboflow and labelled with bboxes
  before it contributes to training.
- **`models/` layout**:
  ```
  models/
  |- yolo26m.pt
  |- yolo26m-pose.pt
  +- custom/
     |- cones_v1.pt
     +- cones_v2.pt
  ```
- **Logging** — per `CONVENTIONS.md`. Per-frame timings at DEBUG, milestones
  at INFO.
- **Commits** — `area: imperative`, e.g. `metrics: add reactive strength
  index`, `tests/agility: implement T-Test pipeline`.

---

## 16. Open questions / decisions still needed

1. **API worker queue** — RQ vs Celery. Plan currently assumes RQ (Redis-only,
   simpler). Confirm before session 7.4.
2. **Test spec authoring strategy** — currently planned just-in-time per
   Phase 4 session. If you want all 32 specs written upfront before any
   test code, insert a Phase 3.5 session block (cost: ~3–5 sessions of
   writing).
3. **Benchmark CSV schema** — needs to be agreed *before* session 1.1.
   Expected columns: `test_id, metric_id, gender, age_band, mean, sd,
   source, ...` — open this when you upload the CSV.
4. **Roboflow vs CVAT** — plan defaults to Roboflow. If you prefer
   self-hosted, swap at session 0.5.3 (no other knock-on effects).
5. **YOLO26 model size** — `n` (fastest, lower accuracy), `s`, `m`
   (balanced default), `x` (highest accuracy, slowest). Plan defaults
   to `m` for both detection and pose. Reconsider after first Phase 4
   test runs at production speed.
6. **AI summary model fallback** — `gpt-5-mini` is the default. If quality
   is insufficient, A/B against `gpt-5` proper as a Phase 6 follow-up.
