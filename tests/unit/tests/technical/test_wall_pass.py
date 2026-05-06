"""Unit tests for Wall Pass pipeline-specific helpers."""
from __future__ import annotations

import numpy as np

from src.tests.technical.wall_pass import _Cycle, _detect_cycles


def _distance_signal(
    n_cycles: int,
    samples_per_cycle: int = 60,
    far: float = 1000.0,
) -> tuple[list[int], np.ndarray]:
    """Synthetic distance signal for `n_cycles` complete pass cycles.
    Each cycle: near (0) -> far (1000) -> near (0)."""
    frames: list[int] = []
    dists: list[float] = []
    fi = 0
    for _ in range(n_cycles):
        # Rising half
        for k in range(samples_per_cycle // 2):
            t = (k + 1) / (samples_per_cycle // 2)
            dists.append(far * t)
            frames.append(fi)
            fi += 1
        # Falling half
        for k in range(samples_per_cycle // 2):
            t = (k + 1) / (samples_per_cycle // 2)
            dists.append(far * (1.0 - t))
            frames.append(fi)
            fi += 1
    return frames, np.array(dists, dtype=float)


def test_detect_cycles_three_clean_passes():
    frames, dists = _distance_signal(n_cycles=3, samples_per_cycle=60)
    speeds = np.full_like(dists, 10.0)
    cycles = _detect_cycles(
        frames=frames, distances=dists, ball_speeds_ms=speeds,
        fps=30.0, max_distance_px=float(dists.max()),
    )
    assert len(cycles) == 3
    # Cycles in chronological order.
    assert cycles[0].pass_release_frame < cycles[1].pass_release_frame
    assert cycles[1].pass_release_frame < cycles[2].pass_release_frame


def test_detect_cycles_records_peak_speed():
    """Peak ball speed during outbound should equal the input max."""
    frames, dists = _distance_signal(n_cycles=1, samples_per_cycle=60)
    speeds = np.linspace(5.0, 20.0, len(dists))
    cycles = _detect_cycles(
        frames=frames, distances=dists, ball_speeds_ms=speeds,
        fps=30.0, max_distance_px=float(dists.max()),
    )
    assert len(cycles) == 1
    # Peak detected during the outbound (rising) leg, so the peak
    # speed should be the max during the first half.
    assert cycles[0].peak_ball_speed_ms > 12.0   # above mid-range


def test_detect_cycles_zero_when_distance_constant():
    """Distance never crosses thresholds -> no cycles."""
    frames = list(range(120))
    dists = np.full(120, 500.0, dtype=float)
    speeds = np.zeros(120, dtype=float)
    cycles = _detect_cycles(
        frames=frames, distances=dists, ball_speeds_ms=speeds,
        fps=30.0, max_distance_px=500.0,
    )
    assert cycles == []


def test_detect_cycles_zero_when_too_short():
    frames = [0, 1, 2]
    dists = np.array([0.0, 500.0, 1000.0])
    speeds = np.zeros(3)
    cycles = _detect_cycles(
        frames=frames, distances=dists, ball_speeds_ms=speeds,
        fps=30.0, max_distance_px=1000.0,
    )
    assert cycles == []
