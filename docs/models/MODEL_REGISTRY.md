# Model Registry

The single source of truth for which models the system uses, where their
weights live, and how to upgrade them.

## Why a registry

YOLO and pose models are upgraded frequently. Hardcoding model paths or names
across the codebase makes upgrades painful. The registry is the single point
of change. **Hard rule #4: never hardcode model paths anywhere else.**

## Implementation

`src/core/models/registry.py` defines the registry as a dict keyed by role.
Entries carry a `backend` field so the loader can dispatch to the correct
inference library (Ultralytics for YOLO, ONNX Runtime for RTMPose).

```python
MODEL_REGISTRY = {
    "object_detector": {
        "name": "yolo26m",
        "weights": "yolo26m.pt",
        "backend": "ultralytics",
        "task": "detect",
        "version": "26.0.0",
        "device": "auto",
        "confidence_default": 0.35,
        "iou_default": 0.45,
    },
    "pose_default": {
        "name": "yolo26m-pose",
        "weights": "yolo26m-pose.pt",
        "backend": "ultralytics",
        "task": "pose",
        "version": "26.0.0",
        "keypoint_count": 17,
        "confidence_default": 0.30,
    },
    "pose_biomech": {
        "name": "rtmpose-x",
        "weights": "rtmpose-x.onnx",
        "backend": "onnx",
        "task": "pose",
        "version": "1.0.0",
        "keypoint_count": 17,
        "input_size": (288, 384),  # RTMPose-x default; (W, H)
        "confidence_default": 0.30,
    },
    "tracker": {
        "name": "bytetrack",
        "backend": "ultralytics",
        "config": "bytetrack.yaml",
    },
    # Custom fine-tuned detectors get added here as they are trained.
    # "detector_cones_v1": {"weights": "custom/cones_v1.pt", "version": "1.0.0", ...},
}
```

Models are loaded lazily — the first call to `get_model("object_detector")`
constructs and caches the instance. Subsequent calls return the cache.

## Pose: default vs biomech

Two pose backends, both first-class:

- **`pose_default` (YOLO26-pose)** — fast, runs alongside the detector with
  shared inference cost. Default for sprint, agility, dribbling, endurance,
  throw/skill families.
- **`pose_biomech` (RTMPose-x)** — higher keypoint AP than YOLO-pose,
  particularly under occlusion and at unusual joint configurations. Used by
  the **jump family** (CMJ, Drop Jump, Squat Jump, Standing Long Jump) and
  the **throw family** (Medicine Ball Throw — release biomechanics).

Selection happens in the family base class, not in test code:

```python
class JumpFamily(BaseTest):
    pose_model_key = "pose_biomech"

class SprintFamily(BaseTest):
    pose_model_key = "pose_default"
```

`src/core/pose/estimator.py` is a factory: it reads `pose_model_key`, looks
up the registry entry, and dispatches to the matching backend loader. The
joint-access API on top is uniform regardless of backend, so metric code
never branches on backend.

## Current pinned versions

| Role | Model | Weights file | Backend | Reason |
|---|---|---|---|---|
| Object detection | YOLO26m | `yolo26m.pt` | ultralytics | Faster CPU inference, NMS-free, better small-object detection (cones, distant balls) |
| Pose (default) | YOLO26m-pose | `yolo26m-pose.pt` | ultralytics | RLE head, drop-in with detector, single forward pass |
| Pose (biomech) | RTMPose-x | `rtmpose-x.onnx` | onnx | Higher keypoint AP for jump takeoff/landing and Medicine Ball Throw release biomechanics. Deployed via ONNX Runtime (no mmpose/mmcv install needed) |
| Tracking | ByteTrack | (config) | ultralytics | Mature, well-suited to single-athlete + ball |

Sizes default to `m`. Reconsider after first Phase 4 test runs at production
speed (open question in `docs/plan/plan.md` §16).

When a new model release ships:
1. Add the new entry to `MODEL_REGISTRY` with a fresh version string.
2. Run the integration test suite against the sample video set.
3. Verify metric outputs are within tolerance (≤ 2% drift on tracked tests).
4. Update this file's "Current pinned versions" table.
5. Bump the project version in `pyproject.toml`.

Old versions stay in the registry forever; we never overwrite weights.

## Where weights live

`models/` at the repo root, gitignored. Layout:

```
models/
|- yolo26m.pt
|- yolo26m-pose.pt
|- rtmpose-x.onnx
+- custom/
   |- cones_v1.pt
   +- cones_v2.pt
```

Populated by:

```bash
uv run scripts/download_models.py
```

The script reads the registry and downloads any missing weights via the
appropriate backend (Ultralytics for YOLO, direct URL from OpenMMLab releases
for the RTMPose ONNX export, direct URL for custom weights).

## Custom fine-tuned detectors

Cones, hurdles, agility markers, medicine balls, target zones, foot-tap mats
— anything outside COCO. Trained per Phase 0.5 of `docs/plan/plan.md`. Every
fine-tuned `.pt` gets a registry entry of the form:

```python
"detector_<class>_v<N>": {
    "weights": "custom/<class>_v<N>.pt",
    "backend": "ultralytics",
    "version": "<N>.0.0",
    "task": "detect",
    "trained_on": "data/_labelling/<class>/dataset.yaml",
}
```

## Device handling

`device: "auto"` resolves to:
- CUDA if `torch.cuda.is_available()`
- MPS on Apple Silicon if explicitly opted in (set `SPA_DEVICE=mps`)
- CPU otherwise

Per-call override via `get_model("object_detector", device="cpu")` for tests.

## Performance notes

YOLO26 and RTMPose-x throughput numbers will be measured in session 0.13
(the end-to-end smoke test) and recorded back here. Plan capacity in the
meantime around 2× video duration on commodity hardware, 0.3× on production
GPUs — refine once measured.
