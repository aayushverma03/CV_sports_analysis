# `straight-line-dribbling` — Straight Line Dribbling

**Domain**: technical
**Family**: dribbling
**Status**: active

## 1. Purpose

Dribble a ball through a straight line of cones / dome markers at maximum
speed. Combines locomotion with ball-control demands.

## 2. Equipment & setup

A straight line of small cones / flat dome markers (typically 5–8 markers
spaced ~1 m apart). Football. Total course length declared by operator
(default 30 m).

Camera: either **side-on** (athlete moves laterally past the camera) or
**rear-view** (camera behind athlete, athlete dribbles away into depth).
Pipeline auto-classifies the view and adjusts the metric set accordingly
(see §5).

## 3. Protocol

1. Athlete stands behind the start marker with the ball at their feet
2. Initiates the run on their own cue (no external start signal)
3. Dribbles along the line of cones, weaving / following depending on
   course style, at maximum speed
4. Stops at the end of the course (or walks off)

The pipeline auto-detects:
- **Start event**: athlete pixel-displacement exceeds threshold after a
  stationary period (motion onset). Same gate used by `linear-sprint`.
- **End event**: athlete pixel-velocity drops below threshold for >= 1 s
  *after* the run started — heuristic stop detection. Cone-finish-line
  detection is not used here because the user's flat dome markers are not
  picked up by YOLO-World (cone_v2 in Phase 8.6 is the planned fix).

## 4. CV pipeline requirements

| Capability | Required? | Notes |
|---|---|---|
| Player detection + tracking | yes | ByteTrack — also tracks ball (COCO class 32) |
| Pose estimation | yes | left/right ankle keypoints for touch detection |
| Ball detection + tracking | yes | COCO `sports_ball` (class 32), tracked alongside player |
| Cone detection | no (v1) | flat dome markers not detected by YOLO-World; not currently used for calibration |
| Calibration | none | distance is declared by operator; pixel-to-metre is not required for the scored metrics |
| Event detection | — | start (motion onset), per-touch (ball–ankle proximity), end (athlete-stops-moving) |

## 5. Metrics produced

See `docs/metrics/METRICS_CATALOG.md` for formulas.

### Always computed (both views)

- `total_completion_time_s` — start frame to stop frame, divided by fps
- `touches_per_metre` — total touches / declared distance (m)
- `left_leg_utilisation_pct` — touches with the left foot as percentage of total touches (informational, no benchmark)

### Side-on only (deferred — needs working pixel-to-metre calibration)

- `ball_foot_distance_m` (mean / median)
- `control_loss_events`
- `total_distance_m`, `average_speed_ms`, `max_speed_ms`, `peak_acceleration_ms2`, `peak_deceleration_ms2`

These metrics depend on a reliable pixel-to-metre scale that the v1
pipeline can't establish (the user's flat dome markers aren't detected
by YOLO-World, and depth-axis motion in rear-view footage doesn't give a
2-D pixel scale either). They light up automatically once cone_v2 lands
or operator-declared athlete-height calibration is added.

### Validity (universal)

`pose_confidence_low_pct`, `tracking_id_drops`, `frames_athlete_offscreen_pct`. View classification result (`side_on` / `rear_view`) is logged in diagnostics.

## 6. Annotation requirements

- Skeleton: yes
- Bounding box on athlete: yes
- Bounding box on ball: yes (when detected)
- HUD ticker fields: `phase`, `elapsed`, `touches`
- End-card: scored metrics + final test score; informational metrics shown but unscored

## 7. Benchmark schema

- File: `benchmarks/technical/straight-line-dribbling.yaml`
- v1 scores 2 metrics out of the 4 in the file (time, touches/m). The
  other two (ball-foot distance, control loss) gate-keep on calibration
  and stay unscored until that lands.

## 8. Failure modes

- Athlete never stationary at start → `ProtocolError` ("athlete was never
  stationary at the start ...")
- Athlete never moves → `ProtocolError` ("no start motion detected")
- Athlete never stops moving (still in motion at end of video) →
  `ProtocolError` ("could not detect end of run")
- Ball never detected → still report time + touches=0, flag in diagnostics

## 9. AI summary prompt notes

Use the family template `src/ai_summary/templates/dribbling.md`. The
ball-foot-distance time series (when available) shows ball-lost moments
that should be highlighted in the coach narrative.

## 10. Acceptance criteria

- [ ] Auto-detects view (side-on / rear-view) on a sample of each
- [ ] Time within ±0.10 s of manual stopwatch ground truth
- [ ] Touch count within ±2 of manual count on a 10-touch run
- [ ] Annotated MP4 plays without artifacts
- [ ] Unit tests cover the state machine, view classifier, and touch
      detector independently of any model

## 11. References

Indicative ranges from soccer slalom/dribble research (Reilly 2008,
Ostojic 2006). Existing legacy dribbling pipeline: port and clean up.
