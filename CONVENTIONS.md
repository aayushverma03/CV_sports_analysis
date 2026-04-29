# Conventions

## Python

- Python 3.11+ (use modern type hints: `list[int]`, `str | None`)
- `ruff` for lint + format (`pyproject.toml` configures both)
- Type-annotate all public functions; use `from __future__ import annotations` everywhere
- No bare `except:` — catch concrete exceptions
- Prefer `pathlib.Path` over `os.path`

## Naming

- Modules: `snake_case`
- Classes: `PascalCase`, suffix the role (`SprintFamily`, `LinearSprintTest`, `CalibrationError`)
- Constants: `SCREAMING_SNAKE_CASE`, top of module
- Test IDs (used in benchmarks YAML, API, filenames): `kebab-case` (e.g. `linear-sprint`, `5x10m-sprint-cod`)
- Metric IDs: `snake_case` matching the metric module filename

## Errors

Custom exception types live in the module that raises them. The four you will
see most:

- `CalibrationError` — pixel-to-metre ratio could not be established
- `DetectionError` — required entity (player, ball, cone set) not found
- `ProtocolError` — video does not match the test protocol (wrong duration, missing markers)
- `BenchmarkLookupError` — no benchmark for `(test, metric, gender, age_band)` triple

Never catch and silently continue. If you cannot proceed, raise. If you can
recover, log at WARNING and explain what assumption you made.

## Logging

Use the standard library `logging`. Get a logger via `logging.getLogger(__name__)`
at module top. Do not configure root logger inside library code — that's the
entrypoint's job.

Log levels:
- `DEBUG` — per-frame inference timings, internal state
- `INFO` — pipeline milestones ("calibrated 5.2 px/cm", "detected 8 cones")
- `WARNING` — recoverable anomalies ("low pose confidence on frame 1230")
- `ERROR` — pipeline-aborting failures

## Docstrings

Numpydoc style. Every public function gets:

```python
def jump_height_from_flight_time(flight_time_s: float) -> float:
    """Compute jump height from flight time using the projectile formula.

    Parameters
    ----------
    flight_time_s : float
        Time between toe-off and touch-down, in seconds.

    Returns
    -------
    float
        Jump height in metres.

    Notes
    -----
    Uses h = g * t² / 8 where g = 9.81 m/s². Assumes the athlete lands at the
    same height they took off from.
    """
```

## Testing

- Unit tests mirror source layout: `src/metrics/jump/jump_height_flight_time.py` ↔ `tests/unit/metrics/test_jump_height_flight_time.py`
- Use `pytest` parametrise for table-driven tests
- Integration tests need a real video — keep these in `tests/integration/` and skip if `data/` is empty
- Mock the Claude API for `src/ai_summary/` tests; do not call the real API in CI

## Commits

- One logical change per commit
- Commit message format: `area: short imperative` — e.g. `metrics: add reactive strength index`, `tests/agility: implement T-Test pipeline`, `scoring: fix age band lookup edge case`

## Do not

- Reach across modules to grab private state
- Add `if test == "X"` branching anywhere outside `src/tests/`
- Cache model objects at import time — load lazily through the registry
- Write to `outputs/` from anywhere but `src/api/` or `src/ui/` or `scripts/`
- Commit anything in `models/`, `data/`, or `outputs/`
