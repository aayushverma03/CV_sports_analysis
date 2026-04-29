# CLAUDE.md — rules of engagement for Claude Code

You are working in a video-analysis sports performance project. Read this first
before touching any file.

## Before you write code, read the relevant docs

This repo is **spec-driven**. Every piece of feature work has a markdown spec in
`docs/`. Load it before editing the corresponding source file.

| Working on… | Read first |
|---|---|
| `src/tests/<family>/<test>.py` | `docs/tests/<domain>/<test>.md` AND `docs/tests/TEST_SPEC_TEMPLATE.md` |
| `src/metrics/<group>/<metric>.py` | `docs/metrics/METRICS_CATALOG.md` |
| `src/scoring/*` | `docs/scoring/NORMALIZATION.md` AND `docs/benchmarks/BENCHMARKS_GUIDE.md` |
| `src/core/annotation/*` | `docs/annotation/VIDEO_ANNOTATION_SPEC.md` |
| `src/ai_summary/*` | `docs/ai_summary/AI_SUMMARY_SPEC.md` |
| `src/core/models/*` | `docs/models/MODEL_REGISTRY.md` |
| `src/api/*` | `docs/api/API_SPEC.md` |
| Anything new | `ARCHITECTURE.md` AND `CONVENTIONS.md` |

## Hard rules

1. **Never put test-specific constants in `src/core/` or `src/metrics/`.** Cone
   layouts, sprint distances, attempt counts → those live in `src/tests/`. If
   you find yourself adding `if test_name == "illinois"` to a core module, stop
   and refactor.

2. **Metrics are pure functions.** No file I/O, no model loading, no logging
   side effects inside `src/metrics/`. Input arrays in, scalar (or small dict)
   out, with units in the docstring.

3. **Single video pass.** Do not loop over the same video file more than once
   per analysis. Detection, tracking, pose, and annotation rendering all share
   a single read of the frames.

4. **Never hardcode model paths.** Always go through
   `src/core/models/registry.py`. Model versions are upgraded by editing the
   registry, not by grepping the source tree.

5. **Benchmarks are data, not code.** Anything that looks like a normative
   value (sprint times, jump heights) belongs in a YAML file under
   `benchmarks/`, never in a Python literal.

6. **Annotated video output is mandatory.** Every test pipeline returns a path
   to an annotated `.mp4`. A test that produces metrics but no annotated video
   is broken.

7. **Fail loud on calibration.** If a test requires real-world distances and
   the calibration step could not establish a pixel-to-metre ratio, raise
   `CalibrationError` — do not silently proceed with pixel-space measurements.

8. **The AthleteProfile is required.** Scoring depends on gender and age band.
   A pipeline call without an `AthleteProfile` is a programming error, not
   "use defaults."

## How to run a Claude Code session

The recommended workflow: pick one row from `ROADMAP.md`, load the doc(s) it
references, implement, run the unit tests for that piece, commit, move on.

Avoid trying to implement multiple test pipelines in one session — each test
has enough setup-specific logic that batching loses focus.

## When in doubt

- Architecture question → `ARCHITECTURE.md`
- "How should I name this?" → `CONVENTIONS.md`
- "What's this metric supposed to compute?" → `docs/metrics/METRICS_CATALOG.md`
- "What does a benchmark file look like?" → `docs/benchmarks/BENCHMARKS_GUIDE.md`
  AND `benchmarks/physical/linear_sprint.yaml` (worked example)
- "How do I score 0–100?" → `docs/scoring/NORMALIZATION.md`

## Things that do not belong in this repo

- Personally identifiable athlete data
- Raw videos (use `data/` locally; it is gitignored)
- API keys (use `.env`, gitignored)
- Trained model weights (use `models/`, gitignored; download via
  `scripts/download_models.py`)
