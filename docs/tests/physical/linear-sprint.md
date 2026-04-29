# `linear-sprint` — Linear Sprint (10 / 20 / 30 / 40 m)

**Domain**: physical
**Family**: sprint
**Status**: active

## 1. Purpose

Measures pure linear acceleration and top-end speed over short distances.
Splits at 10 m, 20 m, and 30 m separate first-step quickness, acceleration,
and speed maintenance phases. Standard testing battery item across DFB,
ACSM, and most football-specific protocols.

## 2. Equipment & setup

- **Course**: 30 m straight lane on flat turf or running track
- **Markers**: cones at 0 m, 10 m, 20 m, 30 m (4 cones in a single line)
- **Camera**: single fixed camera, perpendicular to the running direction,
  positioned to keep all four cones in frame for the entire run
- **Calibration**: cone spacings (10 m intervals) are the calibration
  reference; mandatory

## 3. Protocol

1. Athlete starts in a stationary upright stance behind the 0 m cone
2. Athlete initiates sprint on their own cue (no external start signal)
3. Run continues at maximum effort through the 30 m line
4. Recommended: 3 attempts with ≥ 3 min rest, score the best trial

The pipeline must auto-detect:
- Start event: torso/centre of mass crosses the start gate moving forward
- Gate crossings: 10, 20, 30 m
- End event: 30 m gate crossing

## 4. CV pipeline requirements

| Capability | Required? | Notes |
|---|---|---|
| Player detection + tracking | yes | Single athlete |
| Pose estimation | yes | Used for start-event detection (forward lean) and gait stability checks |
| Ball detection | no | |
| Cone detection | yes | All 4 cones required; failure → `CalibrationError` |
| Calibration | mandatory | 10 m cone intervals |
| Event detection | start motion, 10 m / 20 m / 30 m gate crossings |

## 5. Metrics produced

See `docs/metrics/METRICS_CATALOG.md` for formulas and units.

**Motion:** `total_completion_time_s`, `total_distance_m`, `average_speed_ms`, `max_speed_ms`, `peak_acceleration_ms2`, `peak_deceleration_ms2`, `split_times_s`

**Per-distance scalars (Linear Sprint specific):** `time_10m_s`, `time_20m_s`, `time_30m_s`, `time_40m_s`

**Validity (universal):** `pose_confidence_low_pct`, `calibration_quality`, `tracking_id_drops`, `frames_athlete_offscreen_pct`

## 6. Annotation requirements

- Skeleton: yes (subtle — emphasise stride mechanics)
- Bounding box: yes, white at 80% opacity
- Gate lines: vertical lines at 0/10/20/30 m, colour transitions to `gate-passed` as crossed
- Ball trail: no
- HUD ticker fields: `current_speed`, `elapsed_time`, `current_split`
- Event flashes: pulse on each gate as it is crossed
- End-card: scores for all 6 metrics + test score

## 7. Benchmark schema

- File: `benchmarks/physical/linear-sprint.yaml`
- Score direction: `lower_is_better` for time metrics, `higher_is_better` for `max_speed` and `peak_acceleration`
  (the YAML supports per-metric direction overrides — implement in scoring layer)
- Aggregation: weighted mean — total time 0.4, max speed 0.3, splits 0.3 combined

## 8. Failure modes

- < 4 cones detected → `CalibrationError`
- Athlete not detected for > 15% of frames between start and end → `DetectionError`
- Total run duration < 3 s or > 12 s → `ProtocolError` (likely wrong test or false start)
- Pose confidence < 0.3 on > 30% of frames → flag `low_pose_confidence` but continue

## 9. AI summary prompt notes

Use prompt template `src/ai_summary/templates/sprint.md`. Cross-metric narratives:
- Strong 10 m + weak max speed → "acceleration-dominant profile, develop top-end work"
- Weak 10 m + strong max speed → "first-step development opportunity"
- Strong 20 m, weak 30 m → "top-speed maintenance is the limiting factor"
- High peak deceleration recorded → ignore (not a deceleration test)

## 10. Acceptance criteria

- [ ] Auto-detects all 4 gate crossings on 5 different sample videos
- [ ] Time metrics within ±0.05 s of manual stopwatch ground truth
- [ ] Max speed within ±0.3 m/s of GPS / radar ground truth where available
- [ ] Annotated video shows all overlays and end-card
- [ ] Unit tests cover metric calculations
- [ ] Integration test passes end-to-end

## 11. References

- DFB testing battery U17–U19 documentation
- Existing implementation in `agility.py` (legacy) shares much of the gate-detection logic — port and clean up
