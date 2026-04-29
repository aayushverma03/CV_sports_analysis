"""Generate per-test spec markdown files from a structured config.

This script is run once to scaffold spec files. It is not part of the runtime.
"""
from __future__ import annotations
from pathlib import Path
from textwrap import dedent

DOCS_ROOT = Path(__file__).parent.parent / "docs" / "tests"

# Each entry: (test_id, domain, family, display_name, purpose, equipment, protocol_steps,
#              cv_caps, metrics, hud, calibration_required, score_direction_hint, refs_or_legacy_port)
TESTS: list[dict] = [
    # ─────────────────── PHYSICAL — sprint family ───────────────────
    {
        "id": "5x10m-sprint-cod", "domain": "physical", "family": "sprint",
        "display": "5 × 10 m Sprint with Change of Direction",
        "purpose": "Repeated short sprints (5 shuttles of 10 m) with 180° turns. Measures repeated-effort acceleration, deceleration capacity, and re-acceleration after each turn.",
        "equipment": "Two cones 10 m apart on flat surface. Camera perpendicular to the lane, full course visible.",
        "protocol": [
            "Athlete starts at cone A in stationary stance",
            "Sprint to cone B, plant foot beyond B, 180° turn",
            "Sprint back to cone A, plant foot beyond A, 180° turn",
            "Repeat until 5 × 10 m = 50 m total covered",
            "Time stops when chest crosses the final line",
        ],
        "cv": {"player": True, "pose": True, "ball": False, "cone": True, "calib": "mandatory",
               "events": "start, each shuttle turnaround (foot plant past cone), final crossing"},
        "metrics": [
            ("total_completion_time", "motion", "s"),
            ("split_segment_times", "motion", "s (list × 5)"),
            ("max_speed", "motion", "m/s"),
            ("peak_acceleration", "motion", "m/s²"),
            ("peak_deceleration", "motion", "m/s²"),
        ],
        "hud": ["current_shuttle", "elapsed_time", "current_speed"],
        "score_dir": "lower_is_better (times); higher_is_better (peak metrics)",
        "refs": "Legacy: parts of `agility.py` and `tests/sprint_5x10_test.py`.",
    },
    {
        "id": "bangsbo-sprint", "domain": "physical", "family": "sprint",
        "display": "Bangsbo Sprint Test",
        "purpose": "Repeated 7×34.2 m sprints with fixed rest intervals. Football-specific protocol assessing ability to repeat near-maximal sprints.",
        "equipment": "Marked 34.2 m course with start and finish gates; cone at turnaround.",
        "protocol": [
            "Athlete completes 7 sprints of 34.2 m",
            "25 s rest between sprints (operator-cued or audio)",
            "Each sprint timed independently",
        ],
        "cv": {"player": True, "pose": False, "ball": False, "cone": True, "calib": "mandatory",
               "events": "sprint start × 7, sprint end × 7"},
        "metrics": [
            ("total_completion_time", "motion", "s (per sprint, list × 7)"),
            ("max_speed", "motion", "m/s"),
            ("fatigue_index", "endurance", "% drop best→last sprint"),
            ("average_speed", "motion", "m/s"),
        ],
        "hud": ["current_sprint", "sprint_time", "fatigue_index_running"],
        "score_dir": "lower_is_better (times); higher_is_better (avg & max speed)",
        "refs": "Bangsbo (1994). Implements own metric `fatigue_index` — add to `metrics/endurance/`.",
    },
    {
        "id": "repeated-sprint-ability", "domain": "physical", "family": "sprint",
        "display": "Repeated Sprint Ability (RSA)",
        "purpose": "Series of maximal sprints with short rests; quantifies anaerobic capacity and resistance to performance decrement.",
        "equipment": "30 m straight lane (or 6×40 m or 10×20 m depending on protocol variant declared in config).",
        "protocol": [
            "Athlete completes N sprints (default 6 × 30 m)",
            "Fixed rest interval between sprints (default 20 s)",
            "Each sprint at maximum effort",
        ],
        "cv": {"player": True, "pose": False, "ball": False, "cone": True, "calib": "mandatory",
               "events": "start × N, end × N"},
        "metrics": [
            ("best_sprint_time", "motion", "s"),
            ("mean_sprint_time", "motion", "s"),
            ("total_completion_time", "motion", "s (sum)"),
            ("fatigue_index", "endurance", "%"),
            ("max_speed", "motion", "m/s"),
        ],
        "hud": ["current_sprint", "best_so_far", "fatigue_index_running"],
        "score_dir": "lower_is_better for times; lower_is_better for fatigue_index",
        "refs": "Spencer et al. (2005). Variant config declared in test pipeline.",
    },

    # ─────────────────── PHYSICAL — agility family ───────────────────
    {
        "id": "t-test", "domain": "physical", "family": "agility",
        "display": "T-Test (Agility)",
        "purpose": "T-shaped course tests forward sprint, lateral shuffle, backpedal. Multi-directional agility.",
        "equipment": "4 cones in T-shape: A (start) → B (10 yd / 9.14 m forward) → C (5 yd / 4.57 m left of B) → D (5 yd right of B). Total path: A→B→C→B→D→B→A.",
        "protocol": [
            "Sprint forward A→B, touch base of cone B with right hand",
            "Side-shuffle left to C, touch C with left hand (no crossover)",
            "Side-shuffle right to D, touch D with right hand",
            "Side-shuffle back to B, touch B with left hand",
            "Backpedal B→A through finish line",
        ],
        "cv": {"player": True, "pose": True, "ball": False, "cone": True, "calib": "mandatory",
               "events": "start, each cone touch (5 touches), finish"},
        "metrics": [
            ("total_completion_time", "motion", "s"),
            ("split_segment_times", "motion", "s (per leg × 5)"),
            ("max_speed", "motion", "m/s"),
            ("peak_deceleration", "motion", "m/s²"),
        ],
        "hud": ["elapsed_time", "current_segment"],
        "score_dir": "lower_is_better",
        "refs": "Pauole et al. (2000). Cone-touch detection: hand keypoint within X cm of cone.",
    },
    {
        "id": "illinois-agility", "domain": "physical", "family": "agility",
        "display": "Illinois Agility Test",
        "purpose": "Course with straight sprints and zig-zag through cones. Classic agility assessment.",
        "equipment": "Rectangle 10 m × 5 m with 4 corner cones; 4 internal cones in a line (3.3 m apart) for the zig-zag.",
        "protocol": [
            "Athlete prone behind start cone (face down, hands by shoulders)",
            "On signal, rise and sprint forward 10 m",
            "Turn, sprint back 10 m",
            "Weave through 4 internal cones (forward then back)",
            "Sprint final 10 m to finish",
        ],
        "cv": {"player": True, "pose": True, "ball": False, "cone": True, "calib": "mandatory",
               "events": "rise from prone, each cone passage, finish"},
        "metrics": [
            ("total_completion_time", "motion", "s"),
            ("split_segment_times", "motion", "s (per leg)"),
            ("max_speed", "motion", "m/s"),
            ("peak_deceleration", "motion", "m/s²"),
        ],
        "hud": ["elapsed_time", "current_phase"],
        "score_dir": "lower_is_better",
        "refs": "Standard protocol. Pose required for prone-start detection.",
    },
    {
        "id": "dfb-agility", "domain": "physical", "family": "agility",
        "display": "DFB Agility Test",
        "purpose": "DFB (German FA) standardised agility course. Specific cone layout per DFB testing manual.",
        "equipment": "Cone layout per DFB protocol — see `configs/dfb_agility.yaml` (to be authored from federation manual).",
        "protocol": [
            "Athlete completes the DFB-defined course",
            "Specific COD pattern: 90° and 180° turns",
            "Time recorded from start gate to finish",
        ],
        "cv": {"player": True, "pose": True, "ball": False, "cone": True, "calib": "mandatory",
               "events": "start, each defined waypoint, finish"},
        "metrics": [
            ("total_completion_time", "motion", "s"),
            ("split_segment_times", "motion", "s"),
            ("max_speed", "motion", "m/s"),
            ("peak_acceleration", "motion", "m/s²"),
            ("peak_deceleration", "motion", "m/s²"),
        ],
        "hud": ["elapsed_time", "current_segment"],
        "score_dir": "lower_is_better",
        "refs": "DFB Talentförderprogramm testing manual. Layout file required before implementation.",
    },
    {
        "id": "hurdle-agility-run", "domain": "physical", "family": "agility",
        "display": "Hurdle Agility Run",
        "purpose": "Course combining straight sprints, COD, and low-hurdle clearances. Tests multi-component agility plus jump-and-go transitions.",
        "equipment": "Cones and 3–5 mini-hurdles (15–30 cm) in a defined sequence. Layout per protocol config.",
        "protocol": [
            "Athlete sprints through course",
            "Clears each hurdle without contact",
            "Performs cone-defined COD between hurdle sets",
        ],
        "cv": {"player": True, "pose": True, "ball": False, "cone": True, "calib": "mandatory",
               "events": "start, each hurdle clearance (foot above hurdle), finish; hurdle contact = penalty"},
        "metrics": [
            ("total_completion_time", "motion", "s"),
            ("hurdle_clearance_count", "motion", "int"),
            ("hurdle_contact_count", "motion", "int (penalty)"),
            ("max_speed", "motion", "m/s"),
        ],
        "hud": ["elapsed_time", "hurdles_cleared"],
        "score_dir": "lower_is_better with penalty per contact",
        "refs": "Custom DFB-style hurdle agility protocol. Hurdle contact via ankle keypoint trajectory + hurdle bbox.",
    },

    # ─────────────────── PHYSICAL — jump family ───────────────────
    {
        "id": "counter-movement-jump", "domain": "physical", "family": "jump",
        "display": "Counter Movement Jump (CMJ)",
        "purpose": "Standard vertical jump from upright stance with countermovement. Estimates lower-body explosive power.",
        "equipment": "Side-on camera at hip height, full body in frame, plain wall background preferred.",
        "protocol": [
            "Athlete stands upright, hands on hips (or arms free per variant)",
            "Drops into shallow squat",
            "Jumps vertically as high as possible",
            "Lands on same spot",
        ],
        "cv": {"player": True, "pose": True, "ball": False, "cone": False, "calib": "optional (height via flight time, not pixel measurement)",
               "events": "start of countermovement (hip drops), takeoff (toe leaves ground), peak, landing (toe contact)"},
        "metrics": [
            ("jump_height_flight_time", "jump", "m"),
            ("flight_time", "jump", "s"),
            ("countermovement_depth", "biomech", "m or relative"),
        ],
        "hud": ["jump_height_live", "flight_time"],
        "score_dir": "higher_is_better",
        "refs": "Bosco et al. (1983). Existing legacy `streamlit_app.py` jump pipeline — port.",
    },
    {
        "id": "drop-jump", "domain": "physical", "family": "jump",
        "display": "Drop Jump",
        "purpose": "Athlete drops from a box and immediately rebounds. Measures reactive strength (RSI) and short stretch-shortening cycle.",
        "equipment": "Plyometric box (height per protocol — typical 30 / 40 / 60 cm). Side-on camera.",
        "protocol": [
            "Athlete stands on box edge",
            "Steps off (does not jump down)",
            "On landing, immediately rebounds vertically",
            "Lands on same spot",
        ],
        "cv": {"player": True, "pose": True, "ball": False, "cone": False, "calib": "box height as reference",
               "events": "step-off, ground contact, rebound takeoff, peak, landing"},
        "metrics": [
            ("ground_contact_time", "jump", "s"),
            ("jump_height_rebound", "jump", "m"),
            ("reactive_strength_index", "jump", "m/s"),
            ("flight_time", "jump", "s"),
        ],
        "hud": ["contact_time", "rebound_height", "rsi"],
        "score_dir": "higher_is_better (RSI is the headline)",
        "refs": "Young (1995). Legacy `tests/analyzers/drop_jump.py`.",
    },
    {
        "id": "squat-jump", "domain": "physical", "family": "jump",
        "display": "Squat Jump",
        "purpose": "Vertical jump from a held squat (no countermovement). Isolates concentric force production.",
        "equipment": "Side-on camera. No box.",
        "protocol": [
            "Athlete drops into ~90° knee flexion",
            "Holds for 2–3 s",
            "Jumps vertically without further countermovement",
            "Lands on same spot",
        ],
        "cv": {"player": True, "pose": True, "ball": False, "cone": False, "calib": "none required",
               "events": "squat hold (knee angle stable for ≥1.5s), takeoff, landing"},
        "metrics": [
            ("jump_height_flight_time", "jump", "m"),
            ("flight_time", "jump", "s"),
            ("squat_hold_quality", "biomech", "0..1 score (knee-angle stability during hold)"),
        ],
        "hud": ["jump_height_live", "flight_time"],
        "score_dir": "higher_is_better",
        "refs": "Differs from CMJ by absence of countermovement — flag if pipeline detects hip drop before takeoff.",
    },
    {
        "id": "standing-long-jump", "domain": "physical", "family": "jump",
        "display": "Standing Long Jump",
        "purpose": "Horizontal jump distance from standstill. Lower-body horizontal power.",
        "equipment": "Marked landing zone with distance markings. Camera side-on, perpendicular to jump direction.",
        "protocol": [
            "Athlete stands behind take-off line",
            "Countermovement allowed (arm swing, knee bend)",
            "Jumps horizontally as far as possible",
            "Lands on both feet, distance measured to closest body part to line",
        ],
        "cv": {"player": True, "pose": True, "ball": False, "cone": False, "calib": "mandatory (distance markings)",
               "events": "takeoff, landing (closest point)"},
        "metrics": [
            ("jump_distance", "jump", "m"),
            ("flight_time", "jump", "s"),
            ("takeoff_angle", "biomech", "°"),
        ],
        "hud": ["jump_distance_live"],
        "score_dir": "higher_is_better",
        "refs": "ACSM standard. Calibration via floor markings is critical.",
    },
    {
        "id": "single-leg-hop", "domain": "physical", "family": "jump",
        "display": "Single-Leg Hop",
        "purpose": "Horizontal hop on one leg; tests unilateral power and detects left/right asymmetry.",
        "equipment": "Marked floor with distance scale. Side-on camera.",
        "protocol": [
            "Athlete stands on test leg behind start line",
            "Hops as far as possible, landing on same leg",
            "Hold landing for 2 s without losing balance",
            "Repeat on other leg",
            "3 trials per side; record best",
        ],
        "cv": {"player": True, "pose": True, "ball": False, "cone": False, "calib": "mandatory",
               "events": "takeoff, landing, balance hold (2 s stable)"},
        "metrics": [
            ("hop_distance_left", "jump", "m"),
            ("hop_distance_right", "jump", "m"),
            ("left_right_asymmetry", "biomech", "%"),
            ("balance_hold_quality", "biomech", "0..1"),
        ],
        "hud": ["current_leg", "hop_distance_live"],
        "score_dir": "higher_is_better (distance); target 0 (asymmetry)",
        "refs": "Hopper test (Noyes 1991). Asymmetry > 10% commonly used as a flag — but the AI summary must NOT make injury claims.",
    },

    # ─────────────────── PHYSICAL — endurance family ───────────────────
    {
        "id": "cooper", "domain": "physical", "family": "endurance",
        "display": "Cooper 12-Minute Run",
        "purpose": "Maximal distance covered in 12 minutes of continuous running. VO₂max estimator.",
        "equipment": "Standard 400 m track (or marked closed course). Camera not strictly required if external GPS used; for video-only mode, a fixed wide-angle camera covering the lap.",
        "protocol": [
            "Athlete runs continuously for 12 minutes",
            "Maximal pace, no walking unless necessary",
            "Distance covered at the 12-min mark is recorded",
        ],
        "cv": {"player": True, "pose": False, "ball": False, "cone": True, "calib": "mandatory (track marks)",
               "events": "start, lap crossings, 12:00 minute mark"},
        "metrics": [
            ("total_distance_completed", "endurance", "m"),
            ("average_speed", "motion", "m/s"),
            ("vo2_max_estimate", "endurance", "ml·kg⁻¹·min⁻¹"),
            ("pacing_variability", "endurance", "% lap-time SD"),
        ],
        "hud": ["distance_so_far", "elapsed_time", "current_lap_pace"],
        "score_dir": "higher_is_better",
        "refs": "Cooper (1968). VO₂max regression: VO₂max ≈ (distance_m − 504.9) / 44.73",
    },
    {
        "id": "yo-yo-intermittent", "domain": "physical", "family": "endurance",
        "display": "Yo-Yo Intermittent Recovery Test",
        "purpose": "Progressive shuttle test with recovery jogging between sprints; specific to football's intermittent demands.",
        "equipment": "20 m shuttle marked by cones, 5 m recovery zone behind start cone. Audio cues for pace.",
        "protocol": [
            "Athlete completes 2 × 20 m shuttle at audio-cued pace",
            "10 s active recovery jog through 5 m zone",
            "Pace increases at each level until athlete fails to reach the line on the cue (twice)",
        ],
        "cv": {"player": True, "pose": False, "ball": False, "cone": True, "calib": "mandatory",
               "events": "shuttle start, line crossing per cue, recovery zone entry/exit, fail event"},
        "metrics": [
            ("stage_reached", "endurance", "level"),
            ("total_distance_completed", "endurance", "m"),
            ("vo2_max_estimate", "endurance", "ml·kg⁻¹·min⁻¹"),
            ("max_speed", "motion", "m/s"),
        ],
        "hud": ["current_level", "shuttles_completed", "audio_cue_phase"],
        "score_dir": "higher_is_better",
        "refs": "Bangsbo et al. (2008). Two variants (YYIR1 / YYIR2) — declare which in pipeline config.",
    },
    {
        "id": "multistage-fitness", "domain": "physical", "family": "endurance",
        "display": "Multistage Fitness Test (Bleep / Beep Test)",
        "purpose": "Progressive 20 m shuttle test; classic VO₂max field estimator.",
        "equipment": "20 m shuttle, audio cues at increasing pace.",
        "protocol": [
            "Athlete shuttles 20 m back and forth in time with audio bleeps",
            "Pace increases each level (~1 min per level)",
            "Test ends when athlete fails to reach line on the bleep twice consecutively",
        ],
        "cv": {"player": True, "pose": False, "ball": False, "cone": True, "calib": "mandatory",
               "events": "shuttle starts/ends, bleep timing, fail event"},
        "metrics": [
            ("stage_reached", "endurance", "level.shuttle"),
            ("total_distance_completed", "endurance", "m"),
            ("vo2_max_estimate", "endurance", "ml·kg⁻¹·min⁻¹"),
        ],
        "hud": ["current_level", "shuttles_in_level", "bleep_compliance"],
        "score_dir": "higher_is_better",
        "refs": "Léger & Lambert (1982). VO₂max regression per published table.",
    },
    {
        "id": "30-15-intermittent", "domain": "physical", "family": "endurance",
        "display": "30-15 Intermittent Fitness Test",
        "purpose": "30 s shuttle running + 15 s passive recovery, progressive pace. Intermittent fitness benchmark.",
        "equipment": "40 m shuttle with 3 m tolerance zones at each end. Audio cues.",
        "protocol": [
            "30 s of shuttle running at cued pace",
            "15 s passive recovery walking",
            "Pace increases each stage by 0.5 km/h",
            "Test ends when athlete cannot reach tolerance zone three times",
        ],
        "cv": {"player": True, "pose": False, "ball": False, "cone": True, "calib": "mandatory",
               "events": "stage start/end, recovery transitions, fail events"},
        "metrics": [
            ("vifit_30_15", "endurance", "km/h (final stage velocity)"),
            ("stage_reached", "endurance", "stage"),
            ("vo2_max_estimate", "endurance", "ml·kg⁻¹·min⁻¹"),
            ("total_distance_completed", "endurance", "m"),
        ],
        "hud": ["current_stage", "phase (run / rest)", "distance_in_phase"],
        "score_dir": "higher_is_better",
        "refs": "Buchheit (2008). VIFT = velocity at last completed stage.",
    },
    {
        "id": "incremental-ramp", "domain": "physical", "family": "endurance",
        "display": "Incremental Ramp Test",
        "purpose": "Continuous incremental running pace until volitional exhaustion. Lab-style test executed on a track.",
        "equipment": "Track or treadmill (treadmill variant out of scope for video — track only). Audio pace cues.",
        "protocol": [
            "Start at low jogging pace",
            "Pace increases continuously (e.g. +0.5 km/h every minute)",
            "Continues until athlete cannot maintain pace",
        ],
        "cv": {"player": True, "pose": False, "ball": False, "cone": True, "calib": "mandatory",
               "events": "start, pace-increase boundaries (audio), failure"},
        "metrics": [
            ("max_aerobic_velocity", "endurance", "km/h"),
            ("total_distance_completed", "endurance", "m"),
            ("total_time", "motion", "s"),
            ("pacing_variability", "endurance", "%"),
        ],
        "hud": ["current_pace_target", "compliance", "elapsed_time"],
        "score_dir": "higher_is_better",
        "refs": "Lab-derived; field implementation here. Pace compliance is detection-driven.",
    },

    # ─────────────────── PHYSICAL — mobility / posture / throw ───────────────────
    {
        "id": "sit-and-reach", "domain": "physical", "family": "mobility",
        "display": "Sit-and-Reach",
        "purpose": "Trunk flexion and posterior chain flexibility. Distance reached forward from a seated position.",
        "equipment": "Sit-and-reach box with cm scale OR floor markings beyond athlete's feet. Side-on camera.",
        "protocol": [
            "Athlete sits with legs extended, feet flat against the box",
            "Reaches forward slowly with both hands stacked",
            "Holds the maximum reach for 2 s",
            "Distance recorded at fingertip relative to feet (zero at toes)",
        ],
        "cv": {"player": True, "pose": True, "ball": False, "cone": False, "calib": "mandatory (cm scale)",
               "events": "stable seated start, max reach, hold (2 s), release"},
        "metrics": [
            ("reach_distance", "biomech", "cm (signed; +ve past toes)"),
            ("trunk_flexion_angle", "biomech", "°"),
            ("hold_duration", "biomech", "s"),
        ],
        "hud": ["live_reach", "trunk_angle"],
        "score_dir": "higher_is_better",
        "refs": "ACSM. Pose must give reliable hip + shoulder + wrist keypoints.",
    },
    {
        "id": "stepwise-core-stability", "domain": "physical", "family": "mobility",
        "display": "Stepwise Core Stability",
        "purpose": "Graded core endurance test with progressive postural challenges. Continues until form breaks down.",
        "equipment": "Mat. Side-on camera capturing trunk, hip, and limb keypoints.",
        "protocol": [
            "Athlete progresses through a defined sequence of plank-style holds with increasing difficulty",
            "Each stage held for a target duration (e.g. 15 s)",
            "Test ends when athlete cannot maintain target posture (deviation beyond tolerance)",
        ],
        "cv": {"player": True, "pose": True, "ball": False, "cone": False, "calib": "optional",
               "events": "stage transitions, posture deviation events, fail event"},
        "metrics": [
            ("stage_reached", "endurance", "stage"),
            ("total_hold_duration", "biomech", "s"),
            ("posture_compliance", "biomech", "% time within tolerance"),
            ("deviation_count", "biomech", "int"),
        ],
        "hud": ["current_stage", "hold_time_remaining", "deviation_warning"],
        "score_dir": "higher_is_better (stage); higher_is_better (compliance)",
        "refs": "McGill core endurance principles, adapted to stepwise format. Tolerance bands per stage live in the test config.",
    },
    {
        "id": "landing-error-scoring-system", "domain": "physical", "family": "mobility",
        "display": "Landing Error Scoring System (LESS)",
        "purpose": "Validated 17-item biomechanical scoring of jump-landing technique. Identifies movement patterns associated with elevated injury risk in research literature.",
        "equipment": "30 cm box for drop, target line at 50% body height in front. Side-on AND frontal cameras strongly preferred (frontal-plane scoring items).",
        "protocol": [
            "Athlete drops from 30 cm box",
            "Lands and immediately jumps for max height",
            "Three trials, scored independently and averaged",
        ],
        "cv": {"player": True, "pose": True, "ball": False, "cone": False, "calib": "optional (relative angles only)",
               "events": "drop, ground contact, peak, second landing"},
        "metrics": [
            ("less_total_score", "biomech", "0–17 (lower = better technique)"),
            ("knee_valgus_left", "biomech", "° max"),
            ("knee_valgus_right", "biomech", "° max"),
            ("trunk_flexion_at_landing", "biomech", "°"),
            ("ankle_dorsiflexion_at_landing", "biomech", "°"),
            ("landing_symmetry", "biomech", "%"),
        ],
        "hud": ["less_running_score", "trial_number"],
        "score_dir": "lower_is_better (LESS total); target ranges for individual items",
        "refs": "Padua et al. (2009). 17-item rubric implemented item-by-item. AI summary MUST NOT diagnose injury risk — describe movement quality only.",
    },
    {
        "id": "medicine-ball-throw", "domain": "physical", "family": "throw",
        "display": "Medicine Ball Throw (Seated / Standing)",
        "purpose": "Upper-body / total-body power. Distance medicine ball is thrown.",
        "equipment": "Medicine ball (mass per protocol — typically 2–5 kg). Marked landing zone. Side-on camera.",
        "protocol": [
            "Athlete in seated or standing start position (declare in pipeline config)",
            "Throws ball forward from chest as far as possible",
            "Distance measured from start line to first landing point",
            "3 trials, best recorded",
        ],
        "cv": {"player": True, "pose": True, "ball": True, "cone": False, "calib": "mandatory (landing zone)",
               "events": "release, ball trajectory, landing"},
        "metrics": [
            ("throw_distance", "throw", "m"),
            ("release_velocity", "throw", "m/s"),
            ("release_angle", "throw", "°"),
        ],
        "hud": ["throw_distance_live", "release_velocity"],
        "score_dir": "higher_is_better",
        "refs": "Existing legacy `ball_throw.py` — port and clean up. Ball detection via the YOLO ball class.",
    },
    {
        "id": "foot-tapping", "domain": "physical", "family": "skill",
        "display": "Foot Tapping Test",
        "purpose": "Maximum number of alternating foot taps in a fixed time. Measures lower-limb cyclic speed.",
        "equipment": "Two markers / mats spaced by ~30 cm. Camera side-on or top-down.",
        "protocol": [
            "Athlete stands on one foot, taps alternately between two markers as fast as possible",
            "Fixed duration (typically 10 s)",
            "Total taps counted",
        ],
        "cv": {"player": True, "pose": True, "ball": False, "cone": False, "calib": "optional",
               "events": "tap events (foot contacts marker), test start, test end"},
        "metrics": [
            ("total_taps", "motion", "int"),
            ("taps_per_second", "motion", "1/s"),
            ("left_right_asymmetry", "biomech", "%"),
        ],
        "hud": ["live_tap_count", "taps_per_second_running"],
        "score_dir": "higher_is_better",
        "refs": "Existing legacy `tests/analyzers/tapping.py` — port. Tap detection via ankle keypoint Z-velocity threshold.",
    },

    # ─────────────────── TECHNICAL — dribbling family ───────────────────
    {
        "id": "straight-line-dribbling", "domain": "technical", "family": "dribbling",
        "display": "Straight Line Dribbling",
        "purpose": "Dribble a ball in a straight line at maximum speed. Combines locomotion with ball-control demands.",
        "equipment": "30 m straight lane with start/finish gates. Football. Side-on camera.",
        "protocol": [
            "Athlete starts behind line with ball at feet",
            "Dribbles maximally to finish line",
            "Ball must remain within lane",
        ],
        "cv": {"player": True, "pose": True, "ball": True, "cone": True, "calib": "mandatory",
               "events": "start, ball touches, finish; lane exit = penalty"},
        "metrics": [
            ("total_completion_time", "motion", "s"),
            ("max_speed", "motion", "m/s"),
            ("touches_per_metre", "ball", "1/m"),
            ("ball_foot_distance", "ball", "m (mean & median)"),
            ("lane_exit_count", "ball", "int (penalty)"),
        ],
        "hud": ["elapsed_time", "current_speed", "touches"],
        "score_dir": "lower_is_better (time); higher_is_better (speed); target_value (touches/m)",
        "refs": "Standard skill battery. Existing legacy dribbling pipeline — port.",
    },
    {
        "id": "zig-zag-dribbling", "domain": "technical", "family": "dribbling",
        "display": "Zig-Zag Dribbling",
        "purpose": "Dribble through a slalom of cones. Tight ball control under directional change.",
        "equipment": "5–7 cones at 2 m spacing in a slalom pattern. Football. Camera angled to see all cones.",
        "protocol": [
            "Athlete starts behind start gate with ball",
            "Weaves through every cone (alternating sides)",
            "Returns through finish gate",
            "Penalty for missed cones",
        ],
        "cv": {"player": True, "pose": True, "ball": True, "cone": True, "calib": "mandatory",
               "events": "start, cone passages, finish; missed cone = penalty"},
        "metrics": [
            ("total_completion_time", "motion", "s"),
            ("touches_per_metre", "ball", "1/m"),
            ("ball_foot_distance", "ball", "m (mean & median)"),
            ("cone_miss_count", "ball", "int (penalty)"),
            ("max_speed", "motion", "m/s"),
        ],
        "hud": ["elapsed_time", "cones_passed", "touches"],
        "score_dir": "lower_is_better (time, with penalty per missed cone)",
        "refs": "Standard skill assessment.",
    },
    {
        "id": "figure-of-8-dribbling", "domain": "technical", "family": "dribbling",
        "display": "Figure of 8 Dribbling",
        "purpose": "Dribble in a figure-of-8 pattern around two cones. Continuous tight COD with the ball.",
        "equipment": "2 cones spaced 3 m apart. Football.",
        "protocol": [
            "Athlete starts at midpoint between cones with ball",
            "Dribbles a full figure-of-8 around both cones (e.g. 2 complete loops)",
            "Returns to start",
        ],
        "cv": {"player": True, "pose": True, "ball": True, "cone": True, "calib": "mandatory",
               "events": "start, cone-pass orientations, loop completions, finish"},
        "metrics": [
            ("total_completion_time", "motion", "s"),
            ("touches_per_metre", "ball", "1/m"),
            ("ball_foot_distance", "ball", "m"),
            ("loop_consistency", "ball", "% trajectory match across loops"),
        ],
        "hud": ["elapsed_time", "loops_completed"],
        "score_dir": "lower_is_better",
        "refs": "Custom protocol; loop_consistency is a project-defined metric — implement in `metrics/ball/`.",
    },
    {
        "id": "juggling", "domain": "technical", "family": "skill",
        "display": "Juggling Test",
        "purpose": "Continuous ball juggling without ground contact. Measures ball-control quality and consistency.",
        "equipment": "Football. Open space. Camera framing the athlete from waist up.",
        "protocol": [
            "Athlete starts juggling on signal",
            "Counts each clean touch (foot, thigh, head per protocol)",
            "Ends when ball touches the ground OR fixed duration elapses",
        ],
        "cv": {"player": True, "pose": True, "ball": True, "cone": False, "calib": "optional",
               "events": "touch events with body-part labels, ball drops"},
        "metrics": [
            ("max_consecutive_touches", "ball", "int"),
            ("total_taps", "motion", "int"),
            ("touches_per_second", "motion", "1/s"),
            ("body_part_distribution", "ball", "% per part"),
        ],
        "hud": ["touch_count", "current_streak", "touches_per_second"],
        "score_dir": "higher_is_better",
        "refs": "Existing legacy `juggling_test.py` — port.",
    },

    # ─────────────────── TECHNICAL — passing / shooting ───────────────────
    {
        "id": "wall-pass", "domain": "technical", "family": "skill",
        "display": "Wall Pass Test",
        "purpose": "Repeated passes against a wall, receiving and passing again. Measures passing accuracy + first-touch control speed.",
        "equipment": "Wall, target zone marked on it. Athlete stands behind a marked line at fixed distance. Football.",
        "protocol": [
            "Athlete passes ball to wall target",
            "Receives rebound, controls, passes again",
            "Counts successful passes within fixed duration",
        ],
        "cv": {"player": True, "pose": True, "ball": True, "cone": False, "calib": "mandatory (wall distance)",
               "events": "pass releases, ball-wall contacts, receptions"},
        "metrics": [
            ("successful_passes", "ball", "int"),
            ("passing_accuracy", "ball", "%"),
            ("pass_velocity", "ball", "m/s (mean)"),
            ("decision_time", "ball", "s (reception → next pass)"),
        ],
        "hud": ["pass_count", "accuracy_running", "current_pass_velocity"],
        "score_dir": "higher_is_better (counts, accuracy, velocity); lower_is_better (decision time)",
        "refs": "Standard technical battery.",
    },
    {
        "id": "dfb-shooting", "domain": "technical", "family": "skill",
        "display": "DFB Shooting Test",
        "purpose": "DFB-protocol shooting accuracy from defined positions and into target zones in the goal.",
        "equipment": "Goal with target-zone overlays (typically corners and centre). Marked shooting positions. Football.",
        "protocol": [
            "Athlete shoots from each position in sequence",
            "Aims for declared target zones",
            "Score per shot based on zone hit (zones have different point values per DFB rules)",
        ],
        "cv": {"player": True, "pose": True, "ball": True, "cone": True, "calib": "mandatory (goal frame)",
               "events": "shot release, ball trajectory, goal-line crossing in target zone"},
        "metrics": [
            ("total_score", "ball", "int (DFB rubric)"),
            ("passing_accuracy", "ball", "%"),
            ("pass_velocity", "ball", "m/s (alias for shot velocity)"),
            ("zone_distribution", "ball", "% by zone"),
        ],
        "hud": ["current_position", "running_score", "last_shot_velocity"],
        "score_dir": "higher_is_better",
        "refs": "DFB shooting protocol manual. Goal target zones declared in test config.",
    },

    # ─────────────────── COGNITIVE ───────────────────
    {
        "id": "reaction-time", "domain": "cognitive", "family": "cognitive",
        "display": "Reaction Time Test",
        "purpose": "Simple stimulus-response latency. Measures pure reaction speed to a known stimulus.",
        "equipment": "Stimulus display (screen or LED). Athlete-side camera capturing response action (button press, foot tap, hand raise per protocol).",
        "protocol": [
            "Athlete in ready stance",
            "Random delay (1–4 s), then stimulus",
            "Athlete responds as fast as possible",
            "Multiple trials (e.g. 10), mean and best recorded",
        ],
        "cv": {"player": True, "pose": True, "ball": False, "cone": False, "calib": "n/a",
               "events": "stimulus onsets (timestamp from stimulus system), response actions detected via pose"},
        "metrics": [
            ("reaction_time", "cognitive", "s (mean)"),
            ("reaction_time", "cognitive", "s (best)"),
            ("response_accuracy", "cognitive", "% correct (for choice variants)"),
            ("trial_consistency", "cognitive", "% (1 - SD/mean)"),
        ],
        "hud": ["trial_number", "last_reaction_time", "running_mean"],
        "score_dir": "lower_is_better (time); higher_is_better (accuracy, consistency)",
        "refs": "Stimulus and response timestamps must be aligned to the same clock. The cognitive family base class (`src/tests/families/cognitive_family.py`) handles this.",
    },
    {
        "id": "pattern-recognition", "domain": "cognitive", "family": "cognitive",
        "display": "Pattern Recognition Test",
        "purpose": "Athlete identifies a pattern from a brief visual stimulus. Measures perceptual speed and pattern-matching accuracy.",
        "equipment": "Stimulus display (tactical board / image / short video clip). Response captured by selection input or verbal call detected by the operator.",
        "protocol": [
            "Stimulus shown for fixed duration (e.g. 2 s)",
            "Athlete selects from response options",
            "Response time and correctness recorded",
            "Multiple trials with varying difficulty",
        ],
        "cv": {"player": True, "pose": False, "ball": False, "cone": False, "calib": "n/a",
               "events": "stimulus onset, stimulus offset, response events"},
        "metrics": [
            ("response_accuracy", "cognitive", "% correct"),
            ("decision_latency", "cognitive", "s (mean)"),
            ("difficulty_progression", "cognitive", "ratio of accuracy at hardest vs easiest"),
        ],
        "hud": ["trial_number", "last_decision_time", "accuracy_running"],
        "score_dir": "higher_is_better (accuracy); lower_is_better (latency)",
        "refs": "No motion tracking required for the metric itself, only response capture. AI summary should describe perceptual speed, not 'intelligence'.",
    },
    {
        "id": "video-based-decision-making", "domain": "cognitive", "family": "cognitive",
        "display": "Video-Based Decision-Making",
        "purpose": "Athlete watches game-situation video clips and makes a tactical decision under time pressure. Measures contextual decision-making, not pure reaction.",
        "equipment": "Stimulus video clips (game situations). Response capture (selection or verbal).",
        "protocol": [
            "Clip plays up to a decision point (occluded at frame N)",
            "Athlete decides next action from given options (or describes it)",
            "Multiple clips covering varied scenarios",
        ],
        "cv": {"player": True, "pose": False, "ball": False, "cone": False, "calib": "n/a",
               "events": "clip start, occlusion frame, response event"},
        "metrics": [
            ("response_accuracy", "cognitive", "% correct (vs expert-rated answer)"),
            ("decision_latency", "cognitive", "s from occlusion to response"),
            ("scenario_difficulty_score", "cognitive", "weighted by clip difficulty rating"),
        ],
        "hud": ["clip_number", "last_decision_time", "accuracy_running"],
        "score_dir": "higher_is_better (accuracy, weighted score); lower_is_better (latency)",
        "refs": "Each clip has an expert-rated 'best decision' in the test config. Difficulty weighting per clip is part of the configuration data, not the code.",
    },
]


TEMPLATE = """# `{id}` — {display}

**Domain**: {domain}
**Family**: {family}
**Status**: spec

## 1. Purpose

{purpose}

## 2. Equipment & setup

{equipment}

## 3. Protocol

{protocol_block}

## 4. CV pipeline requirements

| Capability | Required? | Notes |
|---|---|---|
| Player detection + tracking | {player} | |
| Pose estimation | {pose} | |
| Ball detection + tracking | {ball} | |
| Cone detection | {cone} | |
| Calibration | {calib} | |
| Event detection | — | {events} |

## 5. Metrics produced

| Metric ID | Module | Unit |
|---|---|---|
{metric_rows}

## 6. Annotation requirements

- Skeleton: as appropriate to the test (see family default in `docs/annotation/VIDEO_ANNOTATION_SPEC.md`)
- Bounding box: yes
- Markers / gates / target zones: yes (test-specific elements above)
- HUD ticker fields: {hud}
- End-card: all metrics + test score

## 7. Benchmark schema

- File: `benchmarks/{domain}/{id}.yaml`
- Score direction: {score_dir}
- Aggregation: see benchmark file (`weighted_mean` default unless otherwise noted)

## 8. Failure modes

- Missing required entities (cones / ball / wall / box) → `DetectionError` or `CalibrationError`
- Protocol mismatches (duration, sequence) → `ProtocolError`
- Pose confidence below threshold on > 30% of frames → flag, do not abort

## 9. AI summary prompt notes

Use the family template `src/ai_summary/templates/{family}.md`. See its
cross-metric narrative section for patterns to surface for this test.

## 10. Acceptance criteria

- [ ] All §5 metrics computed on a sample video
- [ ] All §6 overlays render
- [ ] Score normalisation produces sane 0–100 outputs
- [ ] Unit tests cover metric calculations
- [ ] Integration test passes end-to-end

## 11. References

{refs}
"""


def render(t: dict) -> str:
    bool_str = lambda b: "yes" if b else "no"
    protocol_block = "\n".join(f"{i+1}. {step}" for i, step in enumerate(t["protocol"]))
    metric_rows = "\n".join(
        f"| `{mid}` | `metrics/{mod}/{mid}.py` | {unit} |"
        for (mid, mod, unit) in t["metrics"]
    )
    return TEMPLATE.format(
        id=t["id"], display=t["display"],
        domain=t["domain"], family=t["family"],
        purpose=t["purpose"],
        equipment=t["equipment"],
        protocol_block=protocol_block,
        player=bool_str(t["cv"]["player"]),
        pose=bool_str(t["cv"]["pose"]),
        ball=bool_str(t["cv"]["ball"]),
        cone=bool_str(t["cv"]["cone"]),
        calib=t["cv"]["calib"],
        events=t["cv"]["events"],
        metric_rows=metric_rows,
        hud=", ".join(f"`{h}`" for h in t["hud"]),
        score_dir=t["score_dir"],
        refs=t["refs"],
    )


if __name__ == "__main__":
    for t in TESTS:
        out = DOCS_ROOT / t["domain"] / f"{t['id']}.md"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(render(t))
        print(f"  wrote {out.relative_to(DOCS_ROOT.parent.parent)}")
    print(f"\n{len(TESTS)} test specs generated.")
