"""T-Test (Agility) pipeline.

T-shape course: athlete sprints forward to centre cone, side-shuffles
left, side-shuffles across right, side-shuffles back to centre, then
backpedals to start. Total path A->B->C->B->D->B->A.

v1 ships only the scored metric `total_completion_time_s`. Cone
detection and segment_completion_times are spec-mandated for v1.x.

Inherits the standard 2-pass agility pipeline from
`src.tests.families.agility_family.AgilityFamilyTest` (player picker,
teleport-aware run window, cone-proximity fallback, bbox+skeleton+HUD
rendering on the chosen player only).

Known limitation: in close-proximity multi-attempt videos (e.g. coach +
two athletes, all crossing within a few bbox-widths of each other)
ByteTrack produces tracks contaminated by ID swaps. The teleport-aware
run-window finder rejects these and surfaces a ProtocolError rather
than silently report a duration spanning multiple attempts.
"""
from __future__ import annotations

from src.tests.families.agility_family import AgilityFamilyTest


class TTestTest(AgilityFamilyTest):
    """T-Test: 2-pass multi-track motion analysis -> total completion time."""

    test_id = "t-test"
    endcard_title = "T-Test (Agility)"
    # T-Test elite male is 8.8 s; below 6 s is implausible.
    min_run_frames = 180
    # Default 0.5 teleport_frac kept — T-Test demo videos are multi-
    # attempt and benefit from breaking runs at ID-swap teleports.
