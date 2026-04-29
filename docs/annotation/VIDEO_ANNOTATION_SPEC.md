# Video Annotation Spec

The annotated video is a deliverable, not a debug aid. It must look like
something a coach is comfortable showing the athlete. This document specifies
the visual language.

## Output

- Container: MP4 (H.264)
- Frame rate: match input
- Resolution: match input (no upscale, no downscale)
- Duration: input length + 4 s end-card

## Layers (back to front)

1. **Source frame** — original video, no filtering
2. **Field overlays** — gates, lanes, target zones, cone path. Drawn as
   semi-transparent fills (40% alpha) with crisp 2 px outlines. Use the
   palette below.
3. **Detection overlays** — tracked bounding box on the athlete (1 px outline,
   no fill), ball trail (fading 30-frame tail), other detections (cones)
   marked with small circles only when relevant
4. **Pose skeleton** — 17-point COCO skeleton, joints as 4 px circles, bones
   as 2 px lines. Colour by limb side (left = blue family, right = orange
   family) so left/right asymmetry is visually obvious
5. **HUD ticker** — top-left, transparent background, 24 px sans-serif.
   Live values for the metrics declared in the test spec's HUD list
6. **Event flashes** — when a key event fires (gate crossed, touch detected,
   takeoff), flash a 3-frame coloured pulse on the relevant overlay element

## End-card (last 4 seconds)

After the input video ends, append a 4-second end-card frame:

```
┌──────────────────────────────────────────────┐
│  Linear Sprint — 30 m                        │
│  Athlete: Profile 0421                       │
│  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━            │
│                                              │
│  Total time         4.42 s    [██████████░] 84  │
│  10 m split         1.84 s    [████████░░] 78   │
│  Max speed          7.2 m/s   [█████████░] 88   │
│  Peak acceleration  6.4 m/s²  [████████░░] 75   │
│  ─────────────────────────────────────────   │
│  Test score: 82 / 100   Band: above_average  │
└──────────────────────────────────────────────┘
```

Implementation: render to a `numpy` frame, append the same frame for `4 * fps`
duration.

## Palette (colour-blind safe)

| Use | Hex |
|---|---|
| Primary accent | `#2E86AB` (teal-blue) |
| Highlight / event flash | `#F4A261` (warm orange) |
| Left-side pose | `#1f77b4` |
| Right-side pose | `#ff7f0e` |
| Athlete bbox | `#FFFFFF` 80% opacity |
| Ball | `#E76F51` |
| Cone marker | `#FFB400` |
| Gate line (active) | `#2A9D8F` |
| Gate line (passed) | `#264653` |
| HUD background | `#000000` 50% opacity |
| HUD text | `#FFFFFF` |

## Typography

- HUD: Inter or Roboto, 24 px regular, 28 px bold for the active metric
- End-card title: 48 px bold
- End-card body: 28 px regular
- All text rendered at native resolution — no font-rasterisation artifacts

## Per-test variations

Each test spec's section §6 specifies which overlays and HUD fields apply.
Examples:

- **Sprint tests**: gate lines, split times in HUD, ball trail off
- **Agility tests**: cone path drawn as a polyline, current segment highlighted
- **Dribbling**: ball trail on, ball–foot distance live in HUD
- **Jumps**: pose skeleton emphasised, vertical reference line, flight-time clock starts on takeoff
- **Throws**: ball trajectory traced from release, distance ruler at landing
- **Cognitive**: stimulus shown in picture-in-picture corner (top-right), reaction-time clock in HUD

## Implementation primitives

`src/core/annotation/overlays.py` provides:

```python
def draw_skeleton(frame, keypoints, confidence, palette) -> frame
def draw_bbox(frame, box, label, colour) -> frame
def draw_gate(frame, line, state) -> frame  # state: 'active' | 'passed' | 'pending'
def draw_ball_trail(frame, history, max_age=30) -> frame
def draw_hud(frame, fields: dict[str, str], position='top-left') -> frame
def render_endcard(spec) -> np.ndarray
def event_flash(frame, region, colour, intensity) -> frame
```

`src/core/annotation/video_annotator.py` orchestrates the layers and writes
the output MP4.

## Performance budget

The annotator runs in the same pass as detection. Annotation rendering must
not exceed 8 ms per frame on a single CPU core (everything is OpenCV
primitives — no PIL, no Cairo). Profile if you suspect drift.

## What NOT to draw

- Per-frame metric values that update jitterily — smooth them or freeze
  between events
- Overlapping labels (label collision detection or just don't show secondary)
- Anything that blocks the athlete's body during the action phase of the test
