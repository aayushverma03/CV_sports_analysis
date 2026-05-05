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

## Phase 4 — Test pipelines (20 of them, one at a time)

Recommended order — each row gets its own session:

**Quick wins:**
1. Linear Sprint (10/20/30/40m) → `src/tests/physical/linear_sprint.py`
2. Counter Movement Jump → `src/tests/physical/counter_movement_jump.py`
3. Drop Jump → `src/tests/physical/drop_jump.py`
4. Straight Line Dribbling → `src/tests/technical/straight_line_dribbling.py`
5. Juggling → `src/tests/technical/juggling.py`
6. Foot Tapping → `src/tests/physical/foot_tapping.py`

**Agility family:**
7. T-Test · 8. Illinois Agility · 9. 5×10m Sprint with COD

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
30-15 Intermittent · 45-Second Agility Hurdle Jump (no test-protocol video) · Cooper · DFB Agility · Hurdle Agility Run (replaced by 45-Second variant) · Incremental Ramp · Single-Leg Hop · Sit-and-Reach · Stepwise Core Stability · DFB Shooting · Reaction Time · Pattern Recognition · Video-Based Decision-Making

## Phase 5 — Annotation polish

- [ ] HUD ticker design pass (consistent typography, colour-blind safe palette)
- [ ] End-card layout: scores + summary headline
- [ ] Per-test overlay variations (gates for sprints, cone path for agility, ball trail for dribbling)
- [ ] Preprocessing video clipper (auto-trim to test-active segments).
  Detect sustained athletic motion across the whole video using the
  teleport-aware sustained-motion logic from `core/tracking/run_window.py`,
  applied per-frame globally rather than per-track. Cone/ball/marker
  presence near a moving athlete is a strong positive signal (YOLO-World
  already detects cones); a low-stride pose cue can distinguish
  test-posture (sprint/jump/weave) from standing still. Multi-attempt
  videos yield multiple clipped segments — caller picks which to score
  or scores each separately. Unblocks the T-Test demo video that
  currently has to fail-loud due to ID-swap contamination.

### Phase 4.14 follow-ups (Figure of 8 Dribbling)

- [ ] **Run window cuts short on slow dribbling motion.** The first
  smoke ended at 5.506 s while the athlete continued dribbling for
  ~30 s. `find_run_on_track`'s motion threshold (3% of bbox-h after
  smoothing) is tuned for sprint-style locomotion; figure-8 dribbling
  has the athlete largely in place, with circular motion that falls
  below threshold once the bbox-h smoothing is applied. Fix options:
  (a) lower motion threshold for dribbling tests, (b) add a
  dribbling-specific "ball-near-foot, ball-moving" alternative
  motion signal, (c) extend run window by gap-merge across longer
  pauses (>1 s) when a ball is being controlled.
- [ ] **Loop counter requires 2 detected cones.** Current smoke
  detected only 1 cone cluster on the indoor-turf video → winding
  computation skipped → loops_completed stuck at 0. Fix lands once
  the Roboflow-trained green/red dome detector replaces YOLO-World
  prompts; until then the loop counter is informational-only on this
  video. As a fallback, infer cone positions from athlete-trajectory
  extrema (the two outward turning points of the figure-8 are at
  the cones).
- [ ] **Re-run with Roboflow-trained markers** (same item as Phase
  4.13's). Will both fix the cone-pair calibration AND restore the
  loop counter.

### Phase 4.13 follow-ups (Zig-Zag Dribbling)

- [ ] **Re-run smoke once the Roboflow-trained cone detector lands.**
  User extracted 368 yellow_pole + 415 green_dome frames via
  `scripts/extract_roboflow_frames.py` and is training custom YOLO
  detectors in Roboflow. Once weights are available, swap them into
  `MarkerDetector` (or create `CustomMarkerDetector`) and re-run
  Phase 4.13 + 4.9 smokes — the cone-pair calibration path should
  start kicking in instead of the body-height-proxy fallback, which
  will tighten distance metrics and unlock cone-passage timestamps
  for ground-truth completion-time validation.
- [ ] **Cone detection finds 0 clusters with default YOLO-World
  prompts on follow-camera videos.** Documented for context;
  superseded by the Roboflow item above.
- [ ] **cone_miss_events + avg_cod_angle_deg.** Both deferred from
  v1. cone_miss_events needs slalom-side analysis (which side of
  the cone did the athlete pass on, vs the alternation pattern of
  the protocol). avg_cod_angle_deg needs trajectory inflection
  analysis at each cone passage.
- [ ] **Re-validate completion time** once cone detection works.
  Current 7.203 s is "elite" and extrapolated. Could be a real fast
  time for U17 or could indicate motion-onset trimming too
  aggressively. Easier to confirm with cone-passage timestamps for
  ground truth.

### Phase 4.11 / 4.12 follow-ups (Jump tests — input videos)

- [ ] **Squat Jump — re-validate on a clean test video.** The Phase 4.11
  smoke ran on a Polytan/Humotion promo video that contains hard scene
  cuts during the jump itself. The pipeline calibrated its standing
  ankle baseline during the pre-cut shot, then mis-interpreted the
  post-cut camera angle as "airborne" — yielding nonsensical metrics
  (jump_height_cm ≈ 1194 cm). Pipeline logic is correct; the video
  isn't a usable test recording. User will provide a single-shot
  side-on phone-recorded squat jump; re-run smoke then.
- [ ] **Standing Long Jump — re-validate on a clean test video.** The
  Phase 4.12 default video is a coach demonstrating the takeoff-line
  setup with a panning camera, not an athlete actually performing the
  jump. User will provide a single-shot side-on phone-recorded
  standing long jump (athlete in frame the whole time, takeoff line
  visible, athlete fully in shot at landing); re-run smoke then.
- [ ] **Optional: scene-cut detection.** Add a frame-to-frame mean
  absolute difference check; if it spikes (hard cut), reset the
  ankle baseline and either resume detection in the new shot or abort
  with ProtocolError. Protects all jump tests from edited footage.

### Phase 4.9 follow-ups (5×10m Sprint with COD)

The pipeline is end-to-end functional but has known precision gaps that
don't affect scored metrics on the demo video. To fix in a later session:

- [ ] **Peak acceleration / deceleration overestimate** (~100 m/s²
  reported vs ~5 m/s² physical max). Bbox-center jitter survives the
  current Savitzky-Golay smoothing on position. Fix options: track the
  pose mid-hip instead of bbox center; or apply a tighter outlier filter
  before differentiation (already capping speed at 15 m/s, but accel
  uses raw differentiation of capped speed).
- [ ] **Cone detection often finds < 4 clusters on yellow-pole + disk
  setups** because YOLO-World's class prompts split detections between
  pole-top and base-disk, or miss poles partially out of frame. Pipeline
  currently falls back to trajectory-extrema calibration (works), but a
  proper 4-cone detection would tighten calibration accuracy. Fix: tune
  the prompt set + add aspect-ratio post-filtering in MarkerDetector to
  prefer tall thin pole detections; or detect the disk base as a
  separate class and combine.
- [ ] **Camera-motion ORB anchor can fail when frame 0's view is
  completely panned away.** Currently falls back to LK chaining (which
  drifts). A multi-anchor scheme (re-anchor against the last successful
  anchor frame, not just frame 0) would be more robust on long videos.

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
- [ ] Documentation review across all 20 test specs
