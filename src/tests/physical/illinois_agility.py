"""Illinois Agility Test pipeline.

Course: 10x5 m rectangle, 4 corner cones + 4 internal cones in a line
3.3 m apart. Athlete starts prone behind start cone, sprints 10 m
forward, returns 10 m, weaves through 4 internal cones (forward then
back), sprints final 10 m to finish.

v1 ships only the scored metric `total_completion_time_s`. Cone
detection, route-violation tracking, and the prone-start event are
spec-mandated for later iteration.

Inherits the standard 2-pass agility pipeline from
`src.tests.families.agility_family.AgilityFamilyTest`. teleport_frac is
raised to 5.0 so brief tracker hiccups (occasional bbox jumps from
detector jitter or 1-frame ID flickers) don't fragment the athlete's
~14 s run; the picker's high area dominance is a strong signal that
the chosen track is genuinely the athlete.
"""
from __future__ import annotations

from src.tests.families.agility_family import AgilityFamilyTest


class IllinoisAgilityTest(AgilityFamilyTest):
    """Illinois Agility: 2-pass multi-track motion analysis -> total time."""

    test_id = "illinois-agility"
    endcard_title = "Illinois Agility"
    # Illinois elite male is ~15 s; below 12 s is implausible.
    min_run_frames = 360                 # 12 s @ 30 fps
    # Sprint-style athlete motion legitimately exceeds 50% of bbox-h
    # per frame near the camera; disable teleport-break logic for this
    # single-athlete test and trust the gap-merge bridging.
    teleport_frac = 5.0
