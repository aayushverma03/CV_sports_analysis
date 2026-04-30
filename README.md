# Sports Performance Analysis

Computer-vision platform that ingests video of an athlete performing a standardised
test, extracts quantitative performance metrics, normalises each metric to a 0–100
score against gender- and age-banded benchmarks, returns an annotated video, and
generates a coach-facing AI summary.

## Scope (20 CV-pipeline tests across 2 domains)

**Physical Capabilities (15)**
5×10m Sprint with COD · Bangsbo Sprint (7×34.2m) · Counter Movement Jump · Drop Jump · Foot Tapping
Illinois Agility · Landing Error Scoring System (LESS, subset) · Linear Sprint (10/20/30/40m)
Medicine Ball Throw · Multistage Fitness · Repeated Sprint Ability · Squat Jump · Standing Long Jump
T-Test · Yo-Yo Intermittent (IR2)

**Technical Skills (5)**
Figure of 8 Dribbling · Juggling · Straight Line Dribbling · Wall Pass · Zig-Zag Dribbling

**Out of v1 scope (deferred — awaiting data or future ship):**
30-15 Intermittent, 45-Second Agility Hurdle Jump (no test-protocol video available),
Cooper, DFB Agility, Hurdle Agility Run (replaced by 45-Second variant), Incremental Ramp,
Single-Leg Hop, Sit-and-Reach, Stepwise Core Stability, DFB Shooting, and the 3
Psychological & Cognitive tests (Pattern Recognition, Reaction Time, Video-Based Decision-Making)
which ship later as in-app games.

## Outputs (per analysis)

1. **Annotated video** — overlaid pose skeleton, tracked bounding boxes, gates/markers, live metric ticker, end-card with scores
2. **Metrics JSON** — every measured value with units, frame ranges, and confidence
3. **Scored report** — each metric normalised to 0–100 against the gender/age benchmark
4. **AI summary** — natural-language coach report (strengths, weaknesses, recommendations)

## Tech stack

- **Detection**: Ultralytics YOLO26 (Sep 2025); upgrade path documented in `docs/models/MODEL_REGISTRY.md`
- **Tracking**: ByteTrack (Ultralytics native)
- **Pose**: YOLO26-pose by default; **RTMPose-x** for biomech-heavy tests (jump family, Medicine Ball Throw release biomechanics), selected via the registry
- **Calibration**: cone + known-marker (raises `CalibrationError` when pixel-to-metre ratio cannot be established)
- **Video / math**: OpenCV, NumPy, SciPy, filterpy
- **API**: FastAPI (background workers for long jobs)
- **UI**: Streamlit (analyst-facing) + the same FastAPI for production integrations
- **AI summary**: OpenAI `gpt-5-mini` via the `openai` SDK
- **Project tooling**: `uv` (always `uv run`, `uv add`); secrets via `.env` loaded with `python-dotenv` at entrypoints only

## Repo layout

```
sports-perf-analysis/
├── docs/             # Specs Claude Code reads to implement features
│   ├── tests/        # One spec per test (20 files)
│   ├── metrics/      # Metric formulas catalogue
│   ├── benchmarks/   # Benchmark schema + how lookups work
│   ├── scoring/      # 0–100 normalisation methodology
│   ├── annotation/   # Video overlay design
│   ├── ai_summary/   # Prompt design for coach summaries
│   ├── models/       # Model registry + upgrade path
│   └── api/          # REST API surface
├── src/
│   ├── core/         # CV primitives shared across all tests
│   ├── metrics/      # Reusable metric calculators
│   ├── tests/        # One pipeline per test, grouped by family
│   ├── scoring/      # Benchmark loading + normalisation
│   ├── ai_summary/   # OpenAI client + per-family prompt templates
│   ├── api/          # FastAPI app
│   └── ui/           # Streamlit app
├── benchmarks/       # YAML files: norms by gender × age band
├── models/           # Downloaded YOLO + pose weights (gitignored)
├── data/             # Sample videos (gitignored)
├── outputs/          # Annotated videos + JSON results (gitignored)
├── tests/            # pytest unit + integration suites
└── scripts/          # CLI entry points + utilities
```

## Quick start

See `ROADMAP.md` for the implementation order. Read `CLAUDE.md` first if you are
Claude Code — it tells you which docs to load before touching which files.

```bash
uv sync
uv run scripts/download_models.py
uv run scripts/run_test.py --test linear_sprint --video data/sample.mp4 --athlete-gender M --athlete-age 17
```

## Project documents

- `ARCHITECTURE.md` — module boundaries, data flow, why things live where they do
- `ROADMAP.md` — phased build plan (use this to drive Claude Code sessions)
- `CONVENTIONS.md` — code style, naming, error handling, logging
- `CLAUDE.md` — Claude Code rules of engagement for this repo
