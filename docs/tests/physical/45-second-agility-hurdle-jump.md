# `45-second-agility-hurdle-jump` — 45-Second Agility Hurdle Jump Test

**Domain**: physical
**Family**: agility
**Status**: active
**Window**: fixed 45 s
**Replaces**: hurdle-agility-run (deferred to v1.1+)

## 1. Purpose

Lower-body power endurance + reactive jumping ability. Originally part of
the SPARQ assessment battery; widely used across football, basketball, and
combat sports. Highly correlated with repeated-sprint and change-of-direction
performance.

## 2. Equipment & setup

- **One mini-hurdle**, 12 in / 30 cm tall (SPARQ-standard height)
- Open flat surface
- Camera side-on, full body + hurdle in frame, 25-30 fps minimum

## 3. Protocol

1. Athlete stands beside the hurdle, both feet on the ground
2. On signal, jumps over the hurdle with both feet (two-footed take-off + landing)
3. Immediately jumps back over (still both feet)
4. Continues alternating sides as fast as possible for **45 seconds**
5. Successful jumps = both feet cleared the hurdle

The pipeline must auto-detect:

- **Test-start event**: athlete's first lift-off (ankle keypoints leave ground baseline)
- **Jump events**: each takeoff-airborne-landing cycle
- **Successful clearance**: both ankle keypoints rise above the hurdle's top y-coordinate during the airborne phase
- **Failed clearance**: ankle clips the hurdle bbox / hurdle bbox displaces / athlete steps around
- **Test-end event**: 45 s elapsed since test start

## 4. CV pipeline requirements

| Capability | Required? | Notes |
|---|---|---|
| Player detection + tracking | yes | Single athlete |
| Pose estimation | yes | `pose_biomech` (RTMPose) — ankle keypoints + clearance height precision |
| Ball detection + tracking | no | |
| Cone detection | no | |
| Hurdle detection | yes | `detector_cone_v1` does not cover hurdles — use `detector_hurdle_v1` |
| Calibration | optional | Hurdle height (30 cm) is a known reference for px-to-m if needed |
| Event detection | — | Each takeoff/landing pair, hurdle-clearance success/failure |

## 5. Metrics produced

See `docs/metrics/METRICS_CATALOG.md` for formulas and units.

**Primary (benchmarked):** `total_successful_jumps` — count of successful clearances in 45 s

**Validity (universal):** `pose_confidence_low_pct`, `calibration_quality`, `tracking_id_drops`, `frames_athlete_offscreen_pct`

**Auxiliary (not benchmarked v1):** `failed_clearance_count`, `mean_jump_height_cm`, `mean_ground_contact_time_s` — useful for technique narrative in the AI summary, won't be folded into the test score.

## 6. Annotation requirements

- Skeleton: yes (jump phase clarity)
- Bounding box: athlete + hurdle
- Markers / gates: hurdle bbox highlighted
- HUD ticker fields: `elapsed_time` (counts down from 45 s), `successful_jumps`, `failed_clearances`
- Event flashes: green pulse on each successful clearance, orange on failed
- End-card: total successful + score band

## 7. Benchmark schema

- File: `benchmarks/physical/45-second-agility-hurdle-jump.yaml`
- Score direction: higher_is_better on `total_successful_jumps`
- Thresholds (per user supply): male P=40 / E=50 / A=68 / L=75; female P=35 / E=42 / A=58 / L=65

## 8. Failure modes

- Hurdle not detected for > 20% of frames → `DetectionError`
- Athlete not detected for > 15% of test-window frames → `DetectionError`
- Test duration drift more than ±2 s from the prescribed 45 s window → `ProtocolError`
- Pose confidence < 0.3 on > 30% of jump frames → flag `low_pose_confidence` but continue

## 9. AI summary prompt notes

Use the family template `src/ai_summary/templates/agility.md`. Cross-metric narratives:

- High `total_successful_jumps` + high `mean_jump_height_cm` → "elite reactive power"
- High count + low mean_height → "fast cadence, may benefit from deeper drive on each jump"
- Many `failed_clearance_count` → "technique-limited; cadence is willing but clearance height is short"

## 10. Acceptance criteria

- [ ] Auto-detects start event within ±3 frames of true first takeoff on the sample video
- [ ] Successful-jump count within ±2 of manual ground truth
- [ ] Failed-clearance flagging consistent with visual review
- [ ] Annotated video shows the running counter + per-jump flashes
- [ ] Unit tests for clearance-height calculation
- [ ] Integration test passes end-to-end

## 11. References

- SPARQ rating system protocol (Nike, 2008)
- TopEnd Sports calculator: <https://www.topendsports.com/testing/tests/agility-jump.htm>
- Sample video (v1): `data/01. Physical Capabilities/45-Second Agility Hurdle Jump/nike_sparq_hurdle_drills.mp4` (Nike SPARQ Mini Hurdle Plyometric Training, 75 s — covers the protocol movement; user to replace with athlete-specific footage in v1.1)
