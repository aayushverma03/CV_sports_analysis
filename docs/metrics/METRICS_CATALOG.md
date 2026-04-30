# Metrics Catalog

The single source of truth for every metric the v1 CV pipeline computes.
Each metric is implemented as a pure function in `src/metrics/<group>/<metric>.py`
(hard rule #2: no I/O, no model loading, no logging side effects).

Per-test specs at `docs/tests/<domain>/<test>.md` reference metric IDs from
this catalog — they do **not** redefine formulas.

## Conventions

- **Units in suffix**: `_s` (seconds), `_m` (metres), `_cm` (centimetres),
  `_ms` (m/s), `_ms2` (m/s²), `_deg` (degrees), `_pct` (percent), `_hz` (hertz).
- **Counts**: no suffix (e.g. `total_taps`, `successful_passes`, `hurdles_cleared`).
- **Lists**: plural metric name (e.g. `split_times_s`, `rep_times_s`, `loop_split_times_s`).
- **Status**: `active` (in v1) or `deferred` (test deferred, metric kept for
  v1.1+). Deferred metrics are not implemented in v1.
- All metrics return `float` scalars or small typed dicts; no DataFrames.
- All time-series inputs are `numpy.ndarray` plus `fps: float`.

## v1 test scope (20 tests)

For brevity, the "Applies to" column uses these short names:

**Sprint family**: `linear_sprint`, `5x10_cod`, `bangsbo` (7×34.2m), `rsa`
**Agility family**: `t_test`, `illinois`
**Jump family**: `cmj`, `drop_jump`, `squat_jump`, `slj` (Standing Long Jump)
**Endurance family**: `yo_yo` (IR2), `multistage`
**Throw family**: `med_ball`
**Dribbling family**: `straight_dribble`, `figure_8`, `zigzag`
**Skill family**: `wall_pass`, `juggling`, `foot_tapping`
**Mobility / Posture family**: `less` (subset score)

Deferred from v1: `45-second-agility-hurdle-jump` (no test-protocol video).
Metric `total_successful_jumps` and `failed_clearance_count` rows below stay
in the catalog for v1.1 reactivation.

---

## 1. Motion metrics — `src/metrics/motion/`

Movement primitives derived from athlete tracking.

| metric_id | definition | applies to |
|---|---|---|
| `total_completion_time_s` | Time from **test-start event** to **test-end event** (`(end_frame − start_frame) / fps`). The test-start event is per-test, defined in each spec's §3 (e.g. CMJ = countermovement onset; Linear Sprint = torso crosses start gate). The HUD elapsed-time clock anchors to the same event — never to video frame 0. | All timed tests except `yo_yo`, `multistage`, `wall_pass` (fixed-duration), `foot_tapping` (fixed-duration), `juggling` (open-ended) |
| `total_distance_m` | Cumulative path length of athlete COM in world coordinates: `Σ ||p[i+1] − p[i]||`. | All movement tests |
| `average_speed_ms` | `total_distance_m / total_completion_time_s`. | All movement tests |
| `max_speed_ms` | Max of Savitzky-Golay-smoothed speed series. | All movement tests |
| `peak_acceleration_ms2` | Max of `d(speed_smooth)/dt`. | Sprint, agility, dribbling, hurdle |
| `peak_deceleration_ms2` | Max magnitude of negative `d(speed_smooth)/dt`, returned as positive. | Agility, COD-heavy tests, drop_jump landing |
| `split_times_s` | List of cumulative times at each gate crossing, in order. | `linear_sprint`, `t_test`, `illinois`, `hurdle_agility`, dribbling tests |
| `time_at_distance_s` | Cumulative time when athlete first crosses a given distance from start. Specialized scalar metrics: `time_10m_s`, `time_20m_s`, `time_30m_s`, `time_40m_s`. | `linear_sprint` |
| `segment_completion_times_s` | Per-segment durations on a multi-segment course. | `t_test` (forward / lat-right / lat-left / backpedal) |

---

## 2. Sprint family metrics — `src/metrics/sprint/`

| metric_id | definition | applies to |
|---|---|---|
| `rep_times_s` | List of times per rep. | `5x10_cod`, `rsa`, `bangsbo` |
| `sprint_best_s` | `min(rep_times_s)`. | `rsa`, `bangsbo` |
| `sprint_worst_s` | `max(rep_times_s)`. | `rsa`, `bangsbo` |
| `sprint_mean_s` | `mean(rep_times_s)`. | `rsa`, `bangsbo` |
| `pct_sprint_decrement` | Glaister formula: `100 × (Σ rep_times / (n × sprint_best_s) − 1)`. | `rsa`, `bangsbo` |
| `fatigue_drop_off_pct` | Two-point: `(rep_last − rep_first) / rep_first × 100`. Distinct from `pct_sprint_decrement`; smaller-N formula. | `5x10_cod` |

---

## 3. Agility family metrics — `src/metrics/agility/`

| metric_id | definition | applies to |
|---|---|---|
| `cone_miss_events` | Count of cones the athlete failed to round on the prescribed side, OR cones the athlete physically contacted. Per-test definition documented in each spec. | `illinois`, `zigzag` |
| `total_successful_jumps` | Count of successful two-footed hurdle clearances within the fixed window (45 s). A clearance = both ankle keypoints rise above the hurdle's top y during airborne phase, no clipping of the hurdle bbox. | `45-second-agility-hurdle-jump` |
| `failed_clearance_count` | Number of jumps where ankle clipped the hurdle or athlete stepped around. Auxiliary; not benchmarked. | `45-second-agility-hurdle-jump` |
| `avg_cod_angle_deg` | Mean change-of-direction angle at each cone (athlete velocity vector before vs after cone). | `zigzag` |
| `hurdles_cleared` | Count of hurdles successfully cleared (no contact, no step-around). | `hurdle_agility` |
| `non_clearance_count` | `total_hurdles − hurdles_cleared`. | `hurdle_agility` |
| `disqualified` | `bool`: `non_clearance_count > 2`. | `hurdle_agility` |
| `lead_foot` | Enum `left` / `right`: leading foot at hurdle clearance (mode across hurdles). | `hurdle_agility` |
| `avg_hurdle_time_s` | Mean inter-hurdle time = `total_completion_time_s / num_hurdles`. | `hurdle_agility` |
| `per_hurdle_split_times_s` | List of times to clear each hurdle from start. | `hurdle_agility` |
| `stride_cadence_hz` | Steps per second from foot-strike detection. | `hurdle_agility` |
| `avg_ground_contact_time_s` | Mean foot-on-ground duration across detected strides. | `hurdle_agility` |
| `knee_symmetry_ratio` | `min(left_max_flex, right_max_flex) / max(left_max_flex, right_max_flex)`. 1.0 = perfect symmetry. | `hurdle_agility` |

**Hurdle non-clearance detection** (signal fusion): hurdle bbox displacement
> 5 cm frame-over-frame (knock) OR athlete ankle keypoint passing through
the hurdle bbox (clip/trip). Camera angle: side-on required.

---

## 4. Jump family metrics — `src/metrics/jump/`

Single attempt per test (v1 protocol). Use `jump_height_cm`, not `best_*`.

| metric_id | definition | applies to |
|---|---|---|
| `jump_height_cm` | Flight-time method: `h = g × t² / 8 × 100`, `g = 9.81`. | `cmj`, `squat_jump`, `drop_jump` (rebound), `slj` (peak height) |
| `flight_time_s` | `(landing_frame − takeoff_frame) / fps`. | `cmj`, `squat_jump`, `drop_jump`, `slj` |
| `ground_contact_time_s` | `(takeoff_frame − landing_frame) / fps` for the landing phase. | `drop_jump` |
| `rsi` | `jump_height_cm / 100 / ground_contact_time_s` (m/s, dimensionally). | `drop_jump` |
| `peak_takeoff_acceleration_ms2` | Peak vertical COM acceleration during ground contact phase. | `cmj`, `squat_jump`, `slj` |
| `peak_landing_deceleration_ms2` | Peak vertical COM deceleration on landing. Single-camera estimate is noisy (±20%); flag confidence. | `drop_jump` |
| `jump_distance_cm` | Horizontal displacement: takeoff toe → back-of-heels at landing, in world units. | `slj` |
| `peak_height_cm` | Max vertical position of COM during flight, relative to takeoff. | `slj` |
| `min_knee_angle_deg` | Minimum knee flexion angle reached (squat depth). Renamed from `average_knee_angle`. | `squat_jump` |

**Drop Jump validity**: drop height is a *setup parameter*, not a measured
metric. Pre-test config specifies the box height (typically 30/40/50 cm).

---

## 5. Throw family metrics — `src/metrics/throw/`

| metric_id | definition | applies to |
|---|---|---|
| `throw_distance_m` | Start line → first ball ground-contact in world units. | `med_ball` |
| `release_velocity_ms` | Projectile back-solve from `(throw_distance_m, release_angle_deg, release_height_m)`: `v = sqrt(g × d / sin(2θ))` adjusted for release height. | `med_ball` |
| `release_angle_deg` | Angle of ball velocity vector at release (from horizontal), measured over the 2–3 frames around release. | `med_ball` |
| `flight_time_s` | Release frame → first ground-contact frame. | `med_ball` |
| `max_height_m` | Peak ball height during flight, world units. | `med_ball` |
| `peak_arm_acceleration_ms2` | Peak wrist-keypoint acceleration in the throw window. Renamed from `peak_throw_acceleration` for clarity (athlete-arm, not ball). | `med_ball` |
| `trunk_rotation_deg` | Shoulder-line rotation amplitude relative to hip-line during throw. | `med_ball` |

---

## 6. Endurance family metrics — `src/metrics/endurance/`

Yo-Yo and Multistage drive stage progression from audio beeps. Two modes:
**audio mode** (primary) — beep detection on video soundtrack + visual
shuttle counting. **manual mode** (fallback) — operator-entered final stage,
with visual sanity check.

| metric_id | definition | applies to |
|---|---|---|
| `final_speed_level` | Last successfully completed speed level (e.g. Level 18 in Yo-Yo IR1). | `yo_yo`, `multistage` |
| `shuttles_at_final_level` | Number of shuttles completed at the final speed level. | `yo_yo` |
| `num_shuttles_completed` | Total successful shuttles across all levels. | `yo_yo` |
| `missed_beep_count` | Number of shuttles where athlete crossed the line *after* the beep. Test ends after 2 consecutive misses. | `yo_yo`, `multistage` |
| `total_distance_m` | `num_shuttles_completed × shuttle_length × 2` (out-and-back). | `yo_yo`, `multistage` |
| `total_completion_time_s` | First beep → end-of-test. | `yo_yo` |
| `vo2max_estimated` | Yo-Yo IR2 regression (Bangsbo et al. 2008): `VO2max = total_distance_m × 0.0136 + 45.3`. | `yo_yo` |
| `split_times_s` | Per-shuttle cumulative times. | `yo_yo` |
| `split_times_per_level_s` | Mean shuttle time at each completed level. | `multistage` |

---

## 7. Coordination metrics — `src/metrics/coordination/`

| metric_id | definition | applies to |
|---|---|---|
| `total_taps` | Count of foot-on-ball taps in the fixed test window (30 s). A tap = ankle keypoint within proximity threshold of ball centre, then leaves. | `foot_tapping` |
| `taps_per_second` | `total_taps / window_duration_s`. | `foot_tapping` |
| `left_taps` | Count restricted to left-foot (ankle 15) taps. **Informational, not benchmarked.** | `foot_tapping` |
| `right_taps` | Count restricted to right-foot (ankle 16) taps. **Informational, not benchmarked.** | `foot_tapping` |

---

## 8. Ball metrics — `src/metrics/ball/`

| metric_id | definition | applies to |
|---|---|---|
| `total_ball_touches` | Count of ball–foot contact events through the test. | `straight_dribble`, `figure_8`, `zigzag`, `juggling` |
| `touches_per_metre` | `total_ball_touches / total_distance_m`. | dribbling tests |
| `ball_foot_distance_m` | Returns `dict(mean_m, median_m, series)`. Distance from ball centre to nearest foot keypoint at each frame during ball-controlled segments. | dribbling tests, `wall_pass` (during reception) |
| `ball_athlete_distance_m` | Distance from ball centre to athlete COM. Distinct from `ball_foot_distance_m`. | dribbling tests |
| `control_loss_events` | Count of frames-runs where `ball_foot_distance_m > threshold` (default 1.0 m). | `straight_dribble` |
| `loop_split_times_s` | Per-loop times around each cone in the figure. | `figure_8` |
| `player_lane_deviation_m` | Max lateral deviation of athlete from the start-finish line (straight-line course only). | `straight_dribble` |
| `ball_lane_deviation_m` | Max lateral deviation of ball from the same line. | `straight_dribble` |
| `max_consecutive_touches` | Longest streak of consecutive ball touches with no drop. | `juggling` |
| `touches_per_second` | `total_ball_touches / juggling_duration_s`. Distinct from `taps_per_second` (foot-tapping); same idea, different domain. | `juggling` |

**Drop event definition (juggling)**: ball Y-coordinate reaches ground plane
(within calibration tolerance) AND ball is no longer within 0.5 m of any
foot/knee/head keypoint. Hand catches and double-bounces both end the streak.

---

## 9. Skill / passing metrics — `src/metrics/skill/`

| metric_id | definition | applies to |
|---|---|---|
| `successful_passes` | Count of complete pass-rebound-recontrol cycles within the 30 s window. A "cycle" = athlete strikes ball toward wall + ball returns + athlete recontrols (ball stays within their control radius). No target zone; no detection of where on the wall the ball landed. | `wall_pass` |
| `passing_accuracy_percent` | `successful_passes / total_pass_releases × 100`. An "attempt" is any pass release toward the wall; "successful" means the athlete recovered the rebound. | `wall_pass` |
| `average_decision_time_s` | Mean time from ball reception (foot stops ball) to next pass release. | `wall_pass` |
| `average_pass_velocity_ms` | Mean ball speed during the post-release window across all passes. | `wall_pass` |
| `max_pass_velocity_ms` | Max single-pass release velocity. | `wall_pass` |
| `left_leg_utilisation_pct` | `left_foot_touches / total_ball_touches × 100`. | dribbling tests, `wall_pass`, `juggling` |

**Wall Pass setup constraints** (test spec, not metric): wall distance and
target-zone polygon must be locked at config time. Without these, accuracy
is undefined.

---

## 10. Biomech / posture metrics — `src/metrics/biomech/`

Posture metrics need precise keypoints — these tests load `pose_biomech`
(RTMPose-x via ONNX) instead of the default YOLO26-pose.

| metric_id | definition | applies to |
|---|---|---|
| `trunk_lean_takeoff_deg` | Trunk forward lean (angle between shoulder-hip line and vertical) at takeoff frame. | `cmj`, `slj` |
| `trunk_lean_initial_contact_deg` | Trunk lean at first ground-contact frame after the drop. | `drop_jump` |
| `trunk_lean_over_hurdle_deg` | Mean trunk lean during airborne phase over each hurdle. | `hurdle_agility` |
| `body_approach_angle_deg` | Angle between athlete trunk-facing direction and incoming ball trajectory at reception. | `wall_pass` |

---

## 11. Universal validity flags (auto-applied to every test)

Computed by the family base class. Surface in the metrics JSON under a
`validity` block, never benchmarked, but consumed by the AI summary
("preliminary indication" softening when quality is marginal).

| metric_id | definition |
|---|---|
| `pose_confidence_low_pct` | % of frames where any required keypoint has confidence < 0.3. |
| `calibration_quality` | Enum: `good` / `marginal` / `failed`. Failed → `CalibrationError` raised before metrics run. |
| `tracking_id_drops` | Count of athlete-ID switches during the test (ByteTrack). |
| `frames_athlete_offscreen_pct` | % of test frames where the athlete bounding-box is partially or fully out of frame. |
| `audio_beep_alignment_quality` | (Yo-Yo, Multistage in audio mode only) Enum: `good` / `marginal` / `audio_missing`. Set to `manual` when manual log is used. |
| `manual_log_consistency_check` | (Yo-Yo, Multistage in manual mode only) Enum: `ok` / `inconsistent`. `inconsistent` when visual shuttle count implies a level off the manual entry by more than ±1. |

---

## Adding a new metric

1. Implement the pure function under the matching group folder.
2. Add a row to the appropriate table here.
3. Reference the `metric_id` in any test spec that uses it (catalog stays canonical).
4. Write unit tests at `tests/unit/metrics/<group>/test_<metric>.py`.

## Deprecated / renamed (decision log)

- `best_jump_height_cm` → **`jump_height_cm`** (single attempt protocol; drop the `best_` prefix).
- `average_knee_angle_deg` → **`min_knee_angle_deg`** (squat depth is the standard interpretation).
- `peak_throw_acceleration_ms2` → **`peak_arm_acceleration_ms2`** (clarify athlete-arm, not ball).
- `acceleration_peak_ms2`, `avg_speed_m_s`, `total_time_s` → unified to `peak_acceleration_ms2`, `average_speed_ms`, `total_completion_time_s`.
- `final_level` (Yo-Yo) → split into **`final_speed_level`** + **`shuttles_at_final_level`**.
- `segment_distances_m` (T-Test) → **`segment_completion_times_s`** (the prescribed distances are fixed; the times are what matters).
- `clearance_rate_pct` (Hurdle) → replaced with **`hurdles_cleared`** + **`non_clearance_count`** + **`disqualified`**.
- `total_time_s` (Wall Pass) → removed (test is fixed 30 s).
- `body_approach_angle_deg` removed from dribbling tests; kept only for `wall_pass`.
- Bangsbo (7×34.2m) had ball metrics in earlier drafts — removed, no ball in this variant.
- LESS, Sit-and-Reach, Stepwise Core Stability, Single-Leg Hop, 30-15 Intermittent, Cooper, DFB Agility, Incremental Ramp, DFB Shooting → all metrics for these tests deferred from v1 (kept on disk in earlier drafts under old names; not implemented in v1).
- **Foot Tapping setup updated** — earlier drafts assumed two ground markers / mats. Final protocol: athlete taps a stationary ball, alternating feet. No `foot_tap_mat` custom class; ball detected via COCO `sports_ball`.
- **Wall Pass `passing_accuracy_percent` redefined** — earlier drafts required a wall target zone polygon. Final protocol has **no target detection**; accuracy = rebound-recovery success rate. Drops the `wall_target_zone` custom detector entirely.
