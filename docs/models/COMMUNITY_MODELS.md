# Community Models (Phase 0.5.2 — Tier 1 search)

Per-class search across Roboflow Universe for pretrained community models
that could replace fine-tuning. Recorded outcomes for each of the 4 v1
custom classes.

## Method

Web-searched Roboflow Universe + similar repositories per class. **Direct
WebFetch of universe.roboflow.com project pages returned 403** during this
search (anti-bot), so per-project mAP / image counts come from search-result
snippets, not from authoritative project pages. Before any training run on
community data, verify the project page directly in a browser.

## Decision criteria

A community model qualifies for Tier 1 (skip training) if and only if:

1. Trained weights are downloadable as `.pt` (not just dataset + train-yourself).
2. Smoke test on a held-out clip from `data/` produces visually reasonable
   detections (no obvious misses, no false positives flooding the frame).
3. Class label matches our convention (or trivially renamable).

If any of those fails but the **dataset** is high quality and large, we use
the **community dataset → train ourselves on yolo26n** path (call this
"Tier 1.5" — faster than labelling our own, gives us a local checkpoint).

If even the dataset isn't a good fit, fall back to Tier 2 (label our own).

---

## Per-class findings

### cones

**Verdict: hybrid — Tier 1.5 community traffic cones + Tier 2 own-labelled disks/poles, single unified `cone` class.**

User confirmed 4 distinct marker types in their videos: orange traffic cones,
green flat dome markers, red flat dome markers, and yellow slalom poles.
Community Universe coverage is **only strong for traffic cones**; the disks
and slalom poles need our own labels.

Strategy: train one `cone` class on mixed data. Downstream test logic only
uses cone *positions*, not types, so per-type subclassing is unnecessary.

Community traffic-cone candidates:

- [robotica-xftin / traffic-cones-4laxg](https://universe.roboflow.com/robotica-xftin/traffic-cones-4laxg) — augmented version, exports for YOLO v5–v26.
- [yolo-4qrlm / traffic-cones-hof8h](https://universe.roboflow.com/yolo-4qrlm/traffic-cones-hof8h) — ~1554 images.
- Roboflow class search: [`class:cone`](https://universe.roboflow.com/search?q=class:cone) — broader catalog.

Recommended next step (Phase 0.5.4–0.5.6):

1. Download a community traffic-cone dataset (~1.5k images) via the
   `roboflow` Python package using the key in `.env`.
2. Extract ~1 fps frames from user videos that contain **disks** (Yo-Yo,
   Multistage, T-Test) and **slalom poles** (dribbling tests). Label
   ~200–300 frames in Roboflow under the same `cone` class.
3. Merge the two datasets, train yolo26n at imgsz=640 for ~50 epochs.
4. Smoke test on held-out frames spanning all 4 marker types.
5. Register as `detector_cones_v1` in `src/core/models/registry.py`.

Total marker-type coverage budget: ~3 hours (frame extraction + labelling
of disks/poles only — community handles traffic cones).

### hurdles

**Verdict: Tier 2 (fine-tune our own from extracted frames).**

Search returned no athletics-mini-hurdle datasets. The "Show Jumping" hits
were equestrian (horse jumps), not relevant. The 45-Second Agility Hurdle
Jump (which replaces Hurdle Agility Run in v1) uses one 12-inch / 30 cm
SPARQ-standard mini-hurdle — specific enough that labelling our own clips
is faster than scouring Universe.

Recommended next step: Phase 0.5.4 — label hurdle instances from the
`45-Second Agility Hurdle Jump` sample video and any other clips we add.
Target 200–400 hurdle instances for v1.

### plyo_box

**Verdict: Tier 2 (fine-tune our own, only for Drop Jump).**

Some gym-equipment datasets exist:

- [Bangkit Academy / gym-equipment-object-detection](https://universe.roboflow.com/bangkit-academy-ognnb/gym-equipment-object-detection) — 6,620 images, but unclear if `plyo_box` is a labelled class (mostly machines, dumbbells, kettlebells).
- [FitForge / gym-equipment-detection-up9ts](https://universe.roboflow.com/fitforge/gym-equipment-detection-up9ts) — 118 images.

Plyo box is a very narrow class (Drop Jump only). Labelling effort to fine-tune
is small (~1 video, ~100 frames). Cheaper than vetting community datasets.

Recommended next step: Phase 0.5.4 — label `Drop-Jump` clips. Target 100–200
plyo-box instances.

### medicine_ball

**Verdict: try Tier 1.5 (use community dataset).**

Candidate identified by user:

- [karunakar-reddy-ruymd / medicine-balls-pwpff](https://universe.roboflow.com/karunakar-reddy-ruymd/medicine-balls-pwpff) — purpose-built medicine ball dataset.

WebFetch returns 403 on Universe (anti-bot) so dataset size + class structure
should be confirmed in browser before downloading. If usable, train yolo26n
on it directly and skip own-labelling.

Recommended next step: download via `roboflow` Python package, train at
imgsz=640 for ~50 epochs, smoke test on `Medicine Ball Throw` clip.

---

## Summary table

| Class | Tier | Action | Rough labelling effort |
|---|---|---|---|
| cones | 1.5 + 2 | Community traffic cones + our disks/poles, unified `cone` class | 200–300 frames (~1.5 hr) |
| hurdles | 2 | Label own footage | 200–400 instances, ~1–2 hrs |
| plyo_box | 2 | Label own footage | 100–200 instances, ~30–60 min |
| medicine_ball | 1.5 | Use [medicine-balls-pwpff](https://universe.roboflow.com/karunakar-reddy-ruymd/medicine-balls-pwpff) directly | 0 hrs |

**Total v1 labelling load: ~3–4 hours of manual work**. Plus model training,
GPU-bound and ~1–2 hours.

---

## Resolved decisions

1. **Cone style** — user uses 4 distinct marker types (traffic cones, green +
   red flat dome markers, yellow slalom poles). Strategy: single unified
   `cone` class trained on a mix of community traffic-cone data + our own
   labelled frames covering disks and poles.
2. **Medicine ball candidate** — [karunakar-reddy-ruymd / medicine-balls-pwpff](https://universe.roboflow.com/karunakar-reddy-ruymd/medicine-balls-pwpff)
   identified by user. Try Tier 1.5 directly.
3. **Dataset acquisition** — `roboflow` Python package using `ROBOFLOW_API_KEY`
   from `.env`. Added as a project dep.

## Cross-references

- Class inventory and protocol decisions → `docs/models/CUSTOM_CLASSES.md`
- Phase 0.5 sessions → `docs/plan/plan.md` §6
- Registry conventions for custom detectors → `docs/models/MODEL_REGISTRY.md` §"Custom fine-tuned detectors"
