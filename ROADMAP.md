# Roadmap

Phases are strictly sequential (solo capacity model). Each item is one
Claude Code session. The detailed plan with locked decisions and per-session
checklists lives in `docs/plan/plan.md` — this file is the short index.

## Phase 0 — Foundations (do once, do not skip)

- [ ] `src/core/models/registry.py` — model registry: load YOLO26 detector, YOLO26-pose (default), RTMPose-x (`pose_biomech`), ByteTrack tracker; version pinning
- [ ] `src/core/utils/video_io.py` — frame iterator, FPS handling, codec sanity
- [ ] `src/core/utils/geometry.py` — pixel ↔ world helpers, angle math
- [ ] `src/core/utils/smoothing.py` — Savitzky-Golay, Kalman wrappers
- [ ] `src/core/calibration/camera_calibration.py` — cone-based and known-marker calibration
- [ ] `src/core/detection/player_detector.py` — single-class wrapper around YOLO with sensible defaults
- [ ] `src/core/tracking/bytetrack_tracker.py` — ByteTrack assoc + ID stability
- [ ] `src/core/pose/estimator.py` — pluggable pose estimator (factory by registry key); confidence-aware joint access
- [ ] `src/core/annotation/overlays.py` — primitives: skeleton, bbox, gate line, HUD ticker
- [ ] `scripts/download_models.py` — pulls registered model weights to `models/`

## Phase 1 — Scoring spine

These have no upstream dependencies on detection and unblock everything later.

- [ ] `src/scoring/benchmarks.py` — load YAML, look up by `(test_id, metric_id, gender, age_band)`
- [ ] `src/scoring/normalization.py` — implement the four scoring modes documented in `docs/scoring/NORMALIZATION.md`
- [ ] `src/scoring/grade.py` — letter / band mapping
- [ ] `tests/unit/scoring/` — full coverage; this layer must be bulletproof
- [ ] Author 3–5 real benchmark files (start with linear sprint, CMJ, sit-and-reach) so the scorer has something to chew on

## Phase 2 — Metric library

Pure functions. Implement one file, write its unit test, move on. Order chosen
so each new family unlocks a real test pipeline in Phase 3.

Motion (unlocks all sprint + agility + endurance):
- [ ] `total_completion_time` · `split_segment_times` · `total_distance` · `average_speed` · `max_speed` · `peak_acceleration` · `peak_deceleration`

Jump (unlocks CMJ, Drop Jump, Squat Jump, Standing Long Jump):
- [ ] `jump_height_flight_time` · `ground_contact_time` · `jump_height_rebound` · `reactive_strength_index`

Ball (unlocks dribbling, passing, juggling):
- [ ] `touches_per_metre` · `ball_foot_distance` · `max_consecutive_touches` · `pass_velocity` · `passing_accuracy`

Throw, Biomech, Endurance, Cognitive metrics: implement when their tests come up.

## Phase 3 — Test family base classes

- [ ] `src/tests/base.py` — `BaseTest` ABC, `AthleteProfile`, `AnalysisResult`
- [ ] `src/tests/families/sprint_family.py`
- [ ] `src/tests/families/agility_family.py`
- [ ] `src/tests/families/jump_family.py`
- [ ] `src/tests/families/dribbling_family.py`
- [ ] `src/tests/families/endurance_family.py` (audio + visual shuttle counting)
- [ ] `src/tests/families/throw_family.py`
- [ ] `src/tests/families/skill_family.py` (discrete action counting, ball control)

## Phase 4 — Test pipelines (21 of them, one at a time)

Recommended order — each row gets its own session:

**Quick wins:**
1. Linear Sprint (10/20/30/40m) → `src/tests/physical/linear_sprint.py`
2. Counter Movement Jump → `src/tests/physical/counter_movement_jump.py`
3. Drop Jump → `src/tests/physical/drop_jump.py`
4. Straight Line Dribbling → `src/tests/technical/straight_line_dribbling.py`
5. Juggling → `src/tests/technical/juggling.py`
6. Foot Tapping → `src/tests/physical/foot_tapping.py`

**Agility family:**
7. T-Test · 8. Illinois Agility · 9. 45-Second Agility Hurdle Jump · 10. 5×10m Sprint with COD

**Jump family:**
11. Squat Jump · 12. Standing Long Jump

**Dribbling family:**
13. Zig-Zag Dribbling · 14. Figure of 8 Dribbling

**Endurance family:**
15. Yo-Yo Intermittent (IR2) · 16. Bangsbo Sprint (7×34.2m) · 17. Multistage Fitness · 18. Repeated Sprint Ability

**Skill / passing:**
19. Wall Pass

**Throw:**
20. Medicine Ball Throw

**Mobility / Posture:**
21. LESS (subset score)

**Out of v1 scope** (deferred — awaiting data, ships in v1.1 or later):
30-15 Intermittent · Cooper · DFB Agility · Hurdle Agility Run (replaced by 45-Second variant) · Incremental Ramp · Single-Leg Hop · Sit-and-Reach · Stepwise Core Stability · DFB Shooting · Reaction Time · Pattern Recognition · Video-Based Decision-Making

## Phase 5 — Annotation polish

- [ ] HUD ticker design pass (consistent typography, colour-blind safe palette)
- [ ] End-card layout: scores + summary headline
- [ ] Per-test overlay variations (gates for sprints, cone path for agility, ball trail for dribbling)

## Phase 6 — AI summary layer

- [ ] `src/ai_summary/summarizer.py` — OpenAI client (`gpt-5-mini`) with retries
- [ ] Prompt template per test family (seven templates: sprint, agility, jump, dribbling, endurance, throw/skill, mobility)
- [ ] Few-shot examples in `src/ai_summary/templates/`
- [ ] Add the summary call as the final step of every test pipeline

## Phase 7 — Surfaces

- [ ] `src/api/main.py` — FastAPI scaffold
- [ ] `src/api/routes/analyze.py` — submit a job, poll for result
- [ ] `src/api/workers.py` — background worker (RQ or Celery, decide and document)
- [ ] `src/ui/streamlit_app.py` — tabbed app, one tab per test
- [ ] `scripts/run_test.py` — CLI for ad-hoc runs

## Phase 8 — Hardening

- [ ] Integration tests with sample videos in `data/` for every test
- [ ] Performance pass: profile a representative video, kill any double-passes over the file
- [ ] Add per-test acceptance criteria to docs (FPS minimum, max processing time)
- [ ] Documentation review across all 21 test specs
