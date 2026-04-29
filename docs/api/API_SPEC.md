# API Spec

REST surface exposed by `src/api/main.py` (FastAPI). The Streamlit UI calls
these same endpoints — there is no parallel implementation.

## Conventions

- Base path: `/v1`
- All times: ISO 8601 UTC
- All file references: server-side paths (relative to `outputs/`) for the
  Streamlit UI; pre-signed URLs for external callers (configure in deployment)
- Errors: standard problem-detail JSON (`type`, `title`, `detail`, `status`)

## Endpoints

### `POST /v1/analyse`

Submit a video for analysis. Returns immediately with a job ID; the actual
work runs on a background worker.

**Request (multipart/form-data)**:
```
test_id:           string   (e.g. "linear-sprint")
video:             file     (mp4)
athlete_gender:    string   "M" | "F" | "X"
athlete_age:       integer  (years)
athlete_id:        string   (optional, for the consumer's tracking)
attempt_label:     string   (optional, e.g. "Attempt 2")
```

**Response (202 Accepted)**:
```json
{
  "job_id": "an_2025_04_28_abc123",
  "status": "queued",
  "test_id": "linear-sprint",
  "submitted_at": "2026-04-28T08:42:11Z"
}
```

### `GET /v1/jobs/{job_id}`

Poll for status.

**Response**:
```json
{
  "job_id": "an_2025_04_28_abc123",
  "status": "running",        // queued | running | done | failed
  "progress": 0.42,           // 0..1, only populated when running
  "message": "Computing metrics",
  "result": null              // populated when status == "done"
}
```

When `status == "done"`, `result` matches the `AnalysisResult` schema below.

### `GET /v1/results/{job_id}`

Same payload as the `result` field above, but as a standalone endpoint for
re-fetching after completion.

### `GET /v1/results/{job_id}/video`

Streams the annotated MP4. Supports `Range` requests for in-browser playback.

### `GET /v1/tests`

Lists all 20 CV-pipeline tests with their metadata.

**Response**:
```json
{
  "tests": [
    {
      "test_id": "linear-sprint",
      "display_name": "Linear Sprint (10/20/30 m)",
      "domain": "physical",
      "family": "sprint",
      "metrics": ["total_completion_time", "split_10m", "split_20m", "split_30m", "max_speed"]
    },
    ...
  ]
}
```

### `GET /v1/tests/{test_id}/benchmark`

Returns the benchmark file for a test as JSON (the same data as the YAML).
Useful for clients building their own visualisations.

## `AnalysisResult` schema

```json
{
  "job_id": "an_2025_04_28_abc123",
  "test_id": "linear-sprint",
  "athlete": {
    "gender": "M",
    "age": 17,
    "age_band": "U18",
    "athlete_id": null
  },
  "completed_at": "2026-04-28T08:43:55Z",

  "metrics": {
    "total_completion_time": { "raw": 4.42, "unit": "s" },
    "split_10m":              { "raw": 1.84, "unit": "s" },
    "split_20m":              { "raw": 3.10, "unit": "s" },
    "split_30m":              { "raw": 4.42, "unit": "s" },
    "max_speed":              { "raw": 7.2,  "unit": "m/s" }
  },

  "scores": {
    "total_completion_time": {
      "raw_value": 4.42, "raw_unit": "s",
      "score": 84, "band": "above_average",
      "percentile_estimate": 78,
      "extrapolated": false,
      "benchmark_confidence": "high"
    },
    ...
  },

  "test_score": { "score": 82, "band": "above_average" },

  "summary": {
    "headline": "Strong overall sprint, with acceleration the standout phase.",
    "strengths": [...],
    "areas_to_develop": [...],
    "training_suggestions": [...]
  },

  "annotated_video_path": "outputs/an_2025_04_28_abc123/annotated.mp4",

  "diagnostics": {
    "fps_input": 60,
    "duration_s": 5.4,
    "calibration_quality": "good",
    "low_pose_confidence_frames_pct": 4.2,
    "extrapolated_bands": []
  }
}
```

## Authentication

Out of scope for the initial build — implement when the API is deployed
externally. Use bearer tokens; rotate via a secrets manager.

## Background workers

`src/api/workers.py` — RQ-based worker that pulls jobs from Redis. The API
process enqueues; the worker process(es) execute. The worker imports
`src/tests/<domain>/<test>.py` directly and calls its `run()` function.

For local dev (no Redis), set `SPA_INLINE_WORKERS=1` to run the pipeline
synchronously inside the request — useful for testing, never for production.

## Pydantic schemas

`src/api/schemas.py` mirrors the JSON above as Pydantic v2 models. The same
models are reused in `src/tests/base.py` so the pipeline returns
`AnalysisResult` natively, no extra serialisation step.
