"""Unit tests for the Juggling pipeline state + helper logic.

Full pipeline run is exercised by `scripts/smoke_juggling.py`.
"""
from __future__ import annotations

from src.tests.technical.juggling import _RunState


def test_runstate_starts_empty():
    state = _RunState()
    assert state.touches == []
    assert state.streaks == []
    assert state.current_streak == 0
    assert state.first_touch_frame is None


# Streak management is internal to JugglingTest.run() — but the streak
# rollup math is straightforward enough to test indirectly via the
# touches metric. Behavioural test below simulates the bookkeeping that
# run() does, exercising the same data shape.


def _simulate_streaks(touches_per_streak: list[int]) -> _RunState:
    state = _RunState()
    frame = 0
    for n in touches_per_streak:
        for _ in range(n):
            state.touches.append(_TouchPlaceholder(frame_idx=frame, side="L"))
            state.last_touch_frame = frame
            if state.first_touch_frame is None:
                state.first_touch_frame = frame
            state.current_streak += 1
            frame += 5  # 5-frame gap between touches
        # End streak (drop)
        state.streaks.append(state.current_streak)
        state.current_streak = 0
        frame += 30  # gap before next streak
    return state


# Mirror the dataclass shape with a minimal stand-in to avoid coupling
# the test to the private _Touch dataclass import.
class _TouchPlaceholder:
    def __init__(self, frame_idx: int, side: str):
        self.frame_idx = frame_idx
        self.side = side


def test_streak_rollup_picks_max():
    state = _simulate_streaks([5, 12, 3, 8])
    from src.metrics.ball.max_consecutive_touches import max_consecutive_touches
    assert max_consecutive_touches(state.streaks) == 12


def test_total_touches_sums_streaks():
    state = _simulate_streaks([5, 12, 3, 8])
    assert len(state.touches) == 28


def test_no_touches_yields_no_streaks():
    state = _RunState()
    from src.metrics.ball.max_consecutive_touches import max_consecutive_touches
    assert max_consecutive_touches(state.streaks) == 0
