"""Pick the single 'main player' track from multi-track histories.

Per-video assumption: there is exactly one athlete performing the test;
everyone else (coach, bystanders, demonstrators) is incidental. This
module returns one track_id for the whole video.

Two-step algorithm:

1. **Pixel-area dominance** — the track that's the LARGEST bbox in the
   most frames. If the winner exceeds `min_dominance_frac` of all
   frames where any track is visible, that's the player.

2. **Object-proximity + motion fallback** — if step 1 doesn't yield a
   clear winner, score each track by `peak_motion /
   median_distance_to_nearest_object` (closer to balls/cones AND
   higher motion = higher score). Returns the highest-scoring track.

Returns None if neither step finds a winner — the caller must decide
how to error.
"""
from __future__ import annotations

import numpy as np

# Per-track history entry shape: (frame_idx, cx, cy, bbox_h, bbox_w)
TrackEntry = tuple[int, float, float, float, float]

_DEFAULT_MOTION_SMOOTH_FRAMES = 30
# Per-frame center jump above this fraction of mean bbox-h is a tracker
# ID-swap teleport — zeroed for smoothing AND treated as a hard run-break.
_TELEPORT_FRAC = 0.5
# A track frame is "in motion" if its smoothed motion exceeds this
# fraction of the track's mean bbox-h.
_MOTION_THRESHOLD_FRAC = 0.03


def pick_player(
    track_history: dict[int, list[TrackEntry]],
    object_positions: dict[int, list[tuple[float, float]]] | None = None,
    *,
    min_history_frames: int = 60,
    min_dominance_frac: float = 0.7,
    verbose: bool = False,
) -> int | None:
    """Top-level orchestrator.

    Parameters
    ----------
    track_history : dict[track_id -> list of (frame_idx, cx, cy, h, w)]
    object_positions : optional dict[frame_idx -> list of (x, y)]
        Reference object positions per frame (cones for agility,
        balls for ball tests). Used only by the fallback step.
    min_history_frames : int
        Skip tracks shorter than this — too brief to be the athlete.
    min_dominance_frac : float
        Step 1 wins only if the top track is the per-frame area
        winner in this fraction of frames.
    """
    eligible = {
        track_id: history
        for track_id, history in track_history.items()
        if len(history) >= min_history_frames
    }
    if verbose:
        print(
            f"[pick_player] {len(track_history)} total tracks, "
            f"{len(eligible)} eligible (>= {min_history_frames} frames)"
        )
        for track_id, history in eligible.items():
            print(f"  track {track_id}: {len(history)} frames")
    if not eligible:
        return None

    winner = pick_by_area_dominance(
        eligible, min_dominance_frac=min_dominance_frac, verbose=verbose
    )
    if winner is not None:
        if verbose:
            print(f"[pick_player] step 1 (area dominance) winner: track {winner}")
        return winner

    if verbose:
        print("[pick_player] step 1 inconclusive, trying step 2 (object proximity)")
    if object_positions:
        winner = pick_by_object_proximity(
            eligible, object_positions, verbose=verbose
        )
        if verbose:
            print(f"[pick_player] step 2 winner: track {winner}")
        return winner
    if verbose:
        print("[pick_player] no object_positions provided; step 2 skipped")
    return None


def pick_by_area_dominance(
    eligible: dict[int, list[TrackEntry]],
    *,
    min_dominance_frac: float,
    verbose: bool = False,
) -> int | None:
    """Per-frame area-winner counts; returns the track that won most often
    if its win-fraction >= `min_dominance_frac`, else None."""
    per_frame: dict[int, dict[int, float]] = {}
    for track_id, history in eligible.items():
        for fi, _, _, h, w in history:
            per_frame.setdefault(fi, {})[track_id] = float(h) * float(w)
    if not per_frame:
        return None
    win_counts: dict[int, int] = {}
    for tracks in per_frame.values():
        winner = max(tracks, key=lambda tid: tracks[tid])
        win_counts[winner] = win_counts.get(winner, 0) + 1
    if not win_counts:
        return None
    top_track, top_wins = max(win_counts.items(), key=lambda x: x[1])
    fraction = top_wins / len(per_frame)
    if verbose:
        print(f"  [area_dominance] win counts: {dict(win_counts)}")
        print(
            f"  [area_dominance] top track {top_track} won {top_wins}/"
            f"{len(per_frame)} = {fraction:.2%} (threshold {min_dominance_frac:.0%})"
        )
    if fraction >= min_dominance_frac:
        return top_track
    return None


def pick_by_object_proximity(
    eligible: dict[int, list[TrackEntry]],
    object_positions: dict[int, list[tuple[float, float]]],
    *,
    smooth_frames: int = _DEFAULT_MOTION_SMOOTH_FRAMES,
    verbose: bool = False,
) -> int | None:
    """Score = longest sustained motion run / median distance to nearest object.

    A track that genuinely runs the test has ONE long uninterrupted
    motion burst. A coach standing still or a tracker that ID-swaps
    between two stationary people produces only short bursts even if
    its smoothed-motion peak looks high. Counting the longest sustained
    run after teleport-cleaning rejects ID-swap noise that peak motion
    doesn't.
    """
    candidates: list[tuple[float, int, int, float]] = []
    for track_id, history in eligible.items():
        distances: list[float] = []
        for fi, cx, cy, _, _ in history:
            objs = object_positions.get(fi)
            if not objs:
                continue
            nearest = min(
                float(np.hypot(ox - cx, oy - cy)) for ox, oy in objs
            )
            distances.append(nearest)
        if not distances:
            continue
        median_dist = float(np.median(distances))

        if len(history) < smooth_frames + 2:
            continue
        run_frames = _longest_sustained_motion(history, smooth_frames)

        # Lower distance + longer sustained motion -> higher score.
        score = run_frames / (median_dist + 1.0)
        candidates.append((score, track_id, run_frames, median_dist))

    if not candidates:
        if verbose:
            print("  [proximity] no candidates with both motion and distance data")
        return None
    if verbose:
        for score, tid, rf, md in sorted(candidates, key=lambda c: -c[0]):
            print(
                f"  [proximity] track {tid}: longest_run={rf}f, "
                f"median_dist={md:.1f}, score={score:.3f}"
            )
    return max(candidates, key=lambda c: c[0])[1]


def _longest_sustained_motion(
    history: list[TrackEntry], smooth_frames: int
) -> int:
    """Frame count of the longest contiguous in-motion run on a track.

    Mirrors t_test._longest_motion_run but returns a length, not a
    window: zeroes out teleport spikes (ID-swap jumps), smooths with a
    boxcar, thresholds against bbox-height, and forces teleport frames
    to act as hard run-breaks so a track that bounces between two
    people can't accumulate one long fake run.
    """
    centers = np.array([(cx, cy) for _, cx, cy, _, _ in history], dtype=float)
    heights = np.array([h for _, _, _, h, _ in history], dtype=float)
    diffs = np.diff(centers, axis=0)
    motion = np.linalg.norm(diffs, axis=1)
    mean_h = float(np.mean(heights))
    teleports = motion > _TELEPORT_FRAC * mean_h
    motion = motion.copy()
    motion[teleports] = 0.0
    kernel = np.ones(smooth_frames, dtype=float) / smooth_frames
    smoothed = np.convolve(motion, kernel, mode="same")
    above = smoothed > _MOTION_THRESHOLD_FRAC * mean_h
    in_motion = above & ~teleports
    best = cur = 0
    for v in in_motion:
        if v:
            cur += 1
            if cur > best:
                best = cur
        else:
            cur = 0
    return int(best)
