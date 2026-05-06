# Test Rules — Quick Reference

One-page summary of every v1 test protocol. For full specs (CV
requirements, metrics, benchmarks, failure modes) see the per-test
docs under `docs/tests/{physical,technical,cognitive}/`.

This file is a derived view; the per-test specs are authoritative.

---

## Summary table

| Test | Family | Rules |
|---|---|---|
| Linear Sprint (10 / 20 / 30 / 40 m) | foundational · sprint | 30 m lane with cones at 0/10/20/30 m. Stand stationary behind 0 m, self-cued start, sprint max effort through 30 m line. Best of 3 trials, ≥3 min rest. |
| 5 × 10 m Sprint with COD | foundational · sprint | 2 cones 10 m apart. Stationary at A, sprint to B with foot beyond + 180° turn, sprint back to A with foot beyond + 180° turn, repeat for 5 × 10 m = 50 m. Time stops at chest crossing final line. |
| Bangsbo Sprint (7 × 34.2 m) | foundational · sprint | 7 sprints of 34.2 m, 25 s rest between (operator/audio-cued), each sprint timed independently. |
| Repeated Sprint Ability | foundational · sprint | N sprints (default 6 × 30 m), fixed rest between (default 20 s), max effort each rep. |
| T-Test | foundational · agility | 4 cones in T (A start; B 9.14 m forward; C 4.57 m left of B; D 4.57 m right of B). Sprint A→B touch B with right hand; side-shuffle to C touch with left (no crossover); side-shuffle to D touch with right; side-shuffle back to B touch with left; backpedal B→A through finish. |
| Illinois Agility | foundational · agility | 10 × 5 m rectangle, 4 corner cones + 4 internal cones 3.3 m apart. Prone behind start (face down, hands by shoulders); on signal sprint 10 m forward; turn, sprint 10 m back; weave through 4 internal cones (forward then back); sprint final 10 m to finish. |
| Counter Movement Jump (CMJ) | foundational · jump | Side-on camera, hip-height. Stand upright with hands on hips (or arms free), drop into shallow squat, jump vertically as high as possible, land on same spot. |
| Squat Jump | foundational · jump | Drop to ~90° knee flexion and **hold for 2–3 s** (no further countermovement), jump vertically, land on same spot. Pipeline flags hip-drop before takeoff as invalid (countermovement). |
| Drop Jump | foundational · jump | Plyo box (30 / 40 / 60 cm). Stand on box edge, **step off** (don't jump down), on landing immediately rebound vertically, land on same spot. |
| Standing Long Jump | foundational · jump | Marked landing zone, side-on camera. Stand behind take-off line, countermovement allowed (arm swing, knee bend), jump horizontally as far as possible, land on both feet. Distance measured to closest body part to line. |
| Yo-Yo Intermittent IR2 | foundational · endurance | 20 m shuttle + 5 m recovery zone, audio-cued pace (start 13 km/h). 2 × 20 m at pace, 10 s active recovery jog through 5 m zone, pace increases each level. Ends when athlete fails to reach line on cue twice. |
| Multistage Fitness (Beep Test) | foundational · endurance | 20 m shuttle, audio bleeps. Shuttle 20 m in time with bleeps; pace increases each level (~1 min per level). Ends after 2 consecutive failed-line-on-bleep events. |
| Medicine Ball Throw | foundational · throw | 2–5 kg ball, marked zone, side-on camera. Seated or standing start (config), throw forward from chest as far as possible, distance from start line to first landing. Best of 3. |
| Landing Error Scoring System (LESS, subset) | foundational · mobility | 30 cm box, target line at 50% body height ahead; side-on + frontal cameras preferred. Drop from 30 cm box, on landing immediately jump for max height. 3 trials, scored independently and averaged. |
| Foot Tapping | foundational · reflex | Football on ground, side-on or top-down camera. Stand over ball, tap ball alternately with each foot as fast as possible, 30 s fixed duration. Total taps counted. |
| Straight Line Dribbling | technical · dribbling | 5–8 markers ~1 m apart, total course length declared (default 30 m). Stand behind start with ball at feet, self-cued start, dribble at max speed, stop at end. Pipeline auto-classifies side-on vs rear-view. |
| Juggling | technical · skill | Ball + open space, waist-up camera. Start juggling on signal; each clean touch counts (foot / thigh / head per protocol); ends when ball touches ground OR fixed duration elapses. |
| Zig-Zag Dribbling | technical · dribbling | 5–7 cones at 2 m spacing in a slalom, ball. Start behind start gate, weave through every cone (alternating sides), return through finish gate. Penalty applied for missed cones. |
| Figure of 8 Dribbling | technical · dribbling | 2 cones 3 m apart, ball. Start at midpoint between cones, dribble a full figure-of-8 around both cones (2 complete loops), return to start. |
| Wall Pass | technical · skill | Wall + ball, marked line at fixed distance from wall. Pass to wall, receive rebound, control, pass again; cycle for fixed duration. Failed rebound recoveries (ball escapes) don't count toward `successful_passes`. |

---

## Foundational (Physical)

### Sprint family

#### Linear Sprint (10 / 20 / 30 / 40 m)
- **Setup**: 30 m lane, cones at 0 / 10 / 20 / 30 m, side-on camera
- 1. Stand stationary behind 0 m cone
- 2. Self-cued start, sprint at max effort through 30 m line
- 3. Best of 3 trials (≥ 3 min rest between)

#### 5 × 10 m Sprint with COD
- **Setup**: 2 cones 10 m apart, perpendicular camera
- 1. Stationary stance at cone A
- 2. Sprint to B, plant foot beyond, 180° turn
- 3. Sprint back to A, plant foot beyond, 180° turn
- 4. Repeat until 5 × 10 m = 50 m
- 5. Time stops when chest crosses the final line

#### Bangsbo Sprint (7 × 34.2 m)
- 1. 7 sprints of 34.2 m
- 2. 25 s rest between sprints (operator- or audio-cued)
- 3. Each sprint timed independently

#### Repeated Sprint Ability
- 1. N sprints (default 6 × 30 m)
- 2. Fixed rest interval between sprints (default 20 s)
- 3. Maximum effort each rep

### Agility family

#### T-Test
- **Setup**: 4 cones in T (A start; B 9.14 m forward; C 4.57 m left of B; D 4.57 m right of B)
- 1. Sprint A → B, touch B with right hand
- 2. Side-shuffle to C, touch with left (no crossover)
- 3. Side-shuffle to D, touch with right
- 4. Side-shuffle back to B, touch with left
- 5. Backpedal B → A through the finish line

#### Illinois Agility
- **Setup**: 10 × 5 m rectangle, 4 corner cones + 4 internal cones (3.3 m apart)
- 1. Prone behind start cone (face down, hands by shoulders)
- 2. On signal, rise and sprint 10 m forward
- 3. Turn, sprint 10 m back
- 4. Weave through 4 internal cones (forward then back)
- 5. Sprint final 10 m to finish

### Jump family

#### Counter Movement Jump (CMJ)
- **Setup**: side-on camera at hip height, full body in frame
- 1. Stand upright, hands on hips (or arms free per variant)
- 2. Drop into shallow squat (countermovement)
- 3. Jump vertically as high as possible
- 4. Land on same spot

#### Squat Jump
- 1. Drop to ~90° knee flexion
- 2. **Hold for 2 – 3 s** (no further countermovement)
- 3. Jump vertically
- 4. Land on same spot
- _Pipeline flags hip-drop before takeoff as invalid (countermovement detected)_

#### Drop Jump
- **Setup**: plyo box (30 / 40 / 60 cm)
- 1. Stand on box edge
- 2. **Step off** (do not jump down)
- 3. On landing, immediately rebound vertically
- 4. Land on same spot

#### Standing Long Jump
- **Setup**: marked landing zone with distance markings, side-on camera
- 1. Stand behind take-off line
- 2. Countermovement allowed (arm swing, knee bend)
- 3. Jump horizontally as far as possible
- 4. Land on both feet; distance measured to closest body part to line

### Endurance family

#### Yo-Yo Intermittent IR2
- **Setup**: 20 m shuttle, 5 m recovery zone, audio-cued pace (start 13 km/h)
- 1. 2 × 20 m shuttle at audio pace
- 2. 10 s active recovery jog through 5 m zone
- 3. Pace increases at each level
- 4. Ends when athlete fails to reach the line on the cue twice

#### Multistage Fitness (Beep Test)
- **Setup**: 20 m shuttle, audio bleeps at increasing pace
- 1. Shuttle 20 m back and forth in time with bleeps
- 2. Pace increases each level (~ 1 min per level)
- 3. Ends after 2 consecutive failed-line-on-bleep events

### Throw

#### Medicine Ball Throw
- **Setup**: 2 – 5 kg ball, marked zone, side-on camera
- 1. Seated or standing start (declared in pipeline config)
- 2. Throw forward from chest as far as possible
- 3. Distance from start line to first landing
- 4. Best of 3 trials

### Mobility

#### Landing Error Scoring System (LESS, subset)
- **Setup**: 30 cm box, target line at 50 % body height ahead; side-on + frontal cameras strongly preferred
- 1. Drop from 30 cm box
- 2. On landing, immediately jump for max height
- 3. 3 trials, scored independently and averaged

### Reflex / Lower-Limb Speed

#### Foot Tapping
- **Setup**: football on ground, side-on or top-down camera
- 1. Stand over ball
- 2. Tap ball alternately with each foot as fast as possible
- 3. Fixed 30 s duration
- 4. Total taps counted

---

## Technical

### Straight Line Dribbling
- **Setup**: 5 – 8 markers ~1 m apart; total course length declared (default 30 m); ball
- 1. Stand behind start with ball at feet
- 2. Self-cued start (no external signal)
- 3. Dribble at maximum speed along the line
- 4. Stop at end of course (or walk off)
- _Pipeline auto-classifies side-on vs rear-view from athlete trajectory and adjusts metric set._

### Juggling
- **Setup**: ball, open space, waist-up camera
- 1. Start juggling on signal
- 2. Each clean touch counts (foot / thigh / head per protocol)
- 3. Ends when ball touches the ground **OR** fixed duration elapses

### Zig-Zag Dribbling
- **Setup**: 5 – 7 cones at 2 m spacing in a slalom; ball
- 1. Start behind start gate with ball
- 2. Weave through every cone (alternating sides)
- 3. Return through finish gate
- 4. Penalty applied for missed cones

### Figure of 8 Dribbling
- **Setup**: 2 cones spaced 3 m apart; ball
- 1. Start at midpoint between cones with ball
- 2. Dribble a full figure-of-8 around both cones (2 complete loops)
- 3. Return to start

### Wall Pass
- **Setup**: wall + ball; marked line at fixed distance from wall (per protocol config)
- 1. Pass ball to the wall
- 2. Receive rebound, control, pass again
- 3. Cycles repeat for fixed duration
- 4. Failed rebound recoveries (ball escapes) don't count toward `successful_passes`
