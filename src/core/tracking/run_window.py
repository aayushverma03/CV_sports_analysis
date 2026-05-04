"""Find the longest sustained motion window on a single track.

Used by every multi-person agility pipeline after `player_picker`
chooses the player track. Teleport-aware: per-frame center jumps above
`_TELEPORT_FRAC * mean(bbox_h)` are zeroed (so smoothing doesn't smear
ID-swap spikes) and forced as hard run-breaks (so a contaminated track
can't read as one continuous run).
"""
from __future__ import annotations

import numpy as np

# Per-track history entry shape: (frame_idx, cx, cy, bbox_h, bbox_w)
TrackEntry = tuple[int, float, float, float, float]

# Defaults match the Illinois/T-Test tunables. Tests that need
# different smoothing or threshold can pass overrides.
_DEFAULT_SMOOTH_FRAMES = 60
_DEFAULT_MOTION_THRESHOLD_FRAC = 0.03
_DEFAULT_GAP_MERGE_FRAMES = 30
_DEFAULT_TELEPORT_FRAC = 0.5


def find_run_on_track(
    history: list[TrackEntry],
    *,
    min_run_frames: int,
    smooth_frames: int = _DEFAULT_SMOOTH_FRAMES,
    motion_threshold_frac: float = _DEFAULT_MOTION_THRESHOLD_FRAC,
    gap_merge_frames: int = _DEFAULT_GAP_MERGE_FRAMES,
    teleport_frac: float = _DEFAULT_TELEPORT_FRAC,
) -> tuple[int, int] | None:
    """Return (start_frame, stop_frame) of the longest motion window on
    a single track, or None if no qualifying segment >= min_run_frames.
    """
    run = longest_motion_run(
        history,
        smooth_frames=smooth_frames,
        motion_threshold_frac=motion_threshold_frac,
        gap_merge_frames=gap_merge_frames,
        teleport_frac=teleport_frac,
    )
    if run is None:
        return None
    start, stop = run
    if (stop - start) < min_run_frames:
        return None
    return run


def longest_motion_run(
    history: list[TrackEntry],
    *,
    smooth_frames: int = _DEFAULT_SMOOTH_FRAMES,
    motion_threshold_frac: float = _DEFAULT_MOTION_THRESHOLD_FRAC,
    gap_merge_frames: int = _DEFAULT_GAP_MERGE_FRAMES,
    teleport_frac: float = _DEFAULT_TELEPORT_FRAC,
) -> tuple[int, int] | None:
    if len(history) < smooth_frames + 2:
        return None
    frame_idxs = [h[0] for h in history]
    centers = np.array([(h[1], h[2]) for h in history], dtype=float)
    heights = np.array([h[3] for h in history], dtype=float)

    diffs = np.diff(centers, axis=0)
    motion = np.linalg.norm(diffs, axis=1)

    mean_h = float(np.mean(heights))
    teleports = motion > teleport_frac * mean_h
    motion = motion.copy()
    motion[teleports] = 0.0

    kernel = np.ones(smooth_frames, dtype=float) / smooth_frames
    smoothed = np.convolve(motion, kernel, mode="same")
    above = smoothed > motion_threshold_frac * mean_h

    merged = above.copy()
    i = 0
    while i < len(merged):
        if not merged[i]:
            j = i
            while j < len(merged) and not merged[j]:
                j += 1
            gap_len = j - i
            has_teleport = bool(teleports[i:j].any())
            if (i > 0 and j < len(merged)
                    and gap_len < gap_merge_frames
                    and not has_teleport):
                merged[i:j] = True
            i = j
        else:
            i += 1
    merged[teleports] = False

    best_start_i = -1
    best_len = 0
    cur_start = -1
    for i, a in enumerate(merged):
        if a:
            if cur_start < 0:
                cur_start = i
        else:
            if cur_start >= 0:
                cur_len = i - cur_start
                if cur_len > best_len:
                    best_start_i = cur_start
                    best_len = cur_len
                cur_start = -1
    if cur_start >= 0:
        cur_len = len(merged) - cur_start
        if cur_len > best_len:
            best_start_i = cur_start
            best_len = cur_len

    if best_start_i < 0 or best_len < 1:
        return None
    start_frame = frame_idxs[best_start_i]
    stop_frame = frame_idxs[min(best_start_i + best_len, len(frame_idxs) - 1)]
    return (start_frame, stop_frame)


def cluster_object_positions(
    detections: list[tuple[float, float]],
    *,
    radius_px: float,
    min_count: int,
) -> list[tuple[float, float]]:
    """Greedy spatial cluster of (cx, cy) detections into stable
    centroids. Used to fold many noisy cone/ball detections sampled
    across pass 1 into a small set of reliable object positions.
    """
    clusters: list[list[float]] = []  # [sum_x, sum_y, count]
    r2 = radius_px * radius_px
    for x, y in detections:
        attached = False
        for c in clusters:
            mx = c[0] / c[2]
            my = c[1] / c[2]
            if (mx - x) ** 2 + (my - y) ** 2 < r2:
                c[0] += x
                c[1] += y
                c[2] += 1
                attached = True
                break
        if not attached:
            clusters.append([x, y, 1])
    return [
        (c[0] / c[2], c[1] / c[2]) for c in clusters if c[2] >= min_count
    ]
