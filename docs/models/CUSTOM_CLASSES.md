# Custom Object Classes (Phase 0.5)

The non-COCO objects the v1 CV pipeline must detect. COCO already gives us
**person** (class 0) and **sports ball** (class 32) — used as-is for athletes
and footballs.

The Phase 0.5 plan: build **one multi-class detector** covering all custom
objects. No separate models. Per-class fine-tunes are a last resort, only if
the multi-class model can't hit usable accuracy.

## Class inventory

| Class | Used by | # tests | Priority | Tier 1 (Roboflow Universe) likelihood | Notes |
|---|---|---|---|---|---|
| **cones** | Linear Sprint, 5×10m COD, Bangsbo, RSA, T-Test, Illinois, Yo-Yo, Multistage, Straight Line Dribble, Figure of 8, Zig-Zag | 11 | **highest** | partial — see notes | User uses **4 distinct marker types**: orange traffic cones (tall), green flat dome markers, red flat dome markers, and yellow slalom poles. Trained as a single `cone` class on mixed data (community traffic-cone dataset + own-labelled disks + poles). Downstream test logic uses position only, not type. |
| **hurdles** | (none in v1 — 45-Second Agility Hurdle Jump deferred) | 0 | medium | medium (athletics hurdles datasets exist) | Mini-hurdles 15–30 cm (12-inch / SPARQ standard). Class kept in custom-detector pool for v1.1 reactivation when athlete-recorded test footage arrives. |
| **plyo_box** | Drop Jump | 1 | low | low (gym datasets sparse) | Wooden / plastic box, 30/40/50 cm typical |
| **medicine_ball** | Medicine Ball Throw | 1 | low | medium (gym fitness datasets) | Distinct from `sports_ball` (COCO) — larger, single-colour, textured. Train separately so YOLO doesn't conflate them |

**Total: 4 custom classes in 1 model.**

## Recommended structure

### `sports_objects_v1.pt` — single multi-class detector
4 classes: `cones`, `hurdles`, `plyo_box`, `medicine_ball`. One inference
call per frame. The 4 classes never co-occur (cones/hurdles/plyo_box/medicine_ball
all live in distinct test settings), so class imbalance during training is
acceptable.

### Use COCO as-is
- **person** — every test
- **sports_ball** — Juggling, all 3 dribbling tests, Wall Pass, **Foot
  Tapping**. Validate once Phase 4 ball-handling tests run on real footage;
  fine-tune `ball_v1` only if YOLO26's stock ball detection underperforms.

## Per-test mapping

| Test | Custom classes needed | COCO classes used |
|---|---|---|
| Linear Sprint | cones | person |
| 5×10m COD | cones | person |
| Bangsbo Sprint | cones | person |
| Repeated Sprint Ability | cones | person |
| T-Test | cones | person |
| Illinois Agility | cones | person |
| CMJ | (none — flight time, no calibration markers) | person |
| Drop Jump | plyo_box | person |
| Squat Jump | (none) | person |
| Standing Long Jump | (calibration markers via tape — see note below) | person |
| Yo-Yo Intermittent | cones | person |
| Multistage Fitness | cones | person |
| Medicine Ball Throw | medicine_ball | person |
| Foot Tapping | (none — uses ball + ankle pose) | person, sports_ball |
| Straight Line Dribble | cones | person, sports_ball |
| Figure of 8 | cones | person, sports_ball |
| Zig-Zag Dribble | cones | person, sports_ball |
| Wall Pass | (none — accuracy from rebound recovery, not target detection) | person, sports_ball |
| Juggling | (none) | person, sports_ball |

## Calibration vs detection

Some objects look like detection candidates but are actually calibration
inputs and live in the calibration pipeline, not the detector:

- **Standing Long Jump landing tape / distance markings** — known-marker
  calibration, set up at config time (athlete stands behind a known line at
  known distance). Pixel-to-metre via `calibrate_linear` from cone-style or
  tape-style markers.
- **Drop Jump box height** — setup parameter (operator declares 30/40/50 cm),
  not measured from video.
- **Wall distance (Wall Pass)** — setup parameter; the wall itself isn't
  detected, just the target zone on it.

## Things that look like objects but aren't

- **Gates** in sprints — derived geometrically from cone positions + athlete
  x-coordinate, not a separate class.
- **Ground contact / takeoff** — derived from foot-keypoint y-velocity, not
  a class.
- **Ball touches** in dribbling / juggling — derived from ball–foot bbox
  proximity, not a class.

## Resolved decisions

1. **Foot Tapping** — athlete taps on a stationary ball, alternating feet
   for the fixed test window. No mat / ladder / floor markers. A "tap" =
   ankle keypoint (left=15, right=16) within proximity threshold of the
   ball center. Left vs right is which ankle registered the tap. Tracked as
   `left_taps` / `right_taps` for FYI; benchmark uses `total_taps` and
   `taps_per_second` only.
2. **Wall Pass** — no target detection. A "successful pass" =
   - athlete strikes ball toward the wall,
   - ball returns,
   - athlete recontrols and strikes again.
   Pure athlete + ball + wall geometry; the wall is a setup distance, not a
   detected object. `passing_accuracy_percent` = recontrolled rebounds /
   total ball releases. Drives drop of the `wall_target_zone` class.

## Cross-references

- Phase 0.5 sessions and tier-source hierarchy → `docs/plan/plan.md` §6
- Where weights live + registry conventions → `docs/models/MODEL_REGISTRY.md`
- Per-test detector requirements → each test spec's §4 in `docs/tests/<domain>/<test>.md`
