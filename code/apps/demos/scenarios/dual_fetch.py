"""dual_fetch — one order, two concurrent missions, two owners in parallel.

"Bring the red cube and the blue ball to the delivery pad" splits into two
independently-owned fetch missions (``submit_multi``): the allocator hands each a
distinct owner, and the two owners run the full search-and-fetch protocol at the
same time with non-overlapping searcher pools (need-to-know intact). The BEV shows
one accent-coloured, labelled ring per mission; the run finishes only when BOTH
objects are on the pad. Full learned stack (GROUND_NET + VLA), fixed four-room
layout, random spawns.
"""

from __future__ import annotations

from code.apps.demos.scenarios.core import Scenario

SCENARIO = Scenario(
    name="dual_fetch",
    title="Two Objects, Two Owners, In Parallel",
    description="One order splits into two concurrent missions - both delivered to the pad",
    layout_kind="rooms",
    spawn_seed=7,
    # Both targets are deep-occluded in DIFFERENT rooms — the red cube in the far
    # corner of storage A, the blue ball in the far corner of storage B — hidden
    # from every robot's start sightline. So neither owner can see its object nor
    # get a peer visibility report: BOTH must delegate a room-to-room search, and
    # the two searches run in parallel on opposite sides of the warehouse (the
    # whole point of the demo). The cross-owner searcher budget gives each owner
    # its own searcher (never both to the first owner), and each searcher is the
    # peer nearest its owner, so it sweeps the room that actually holds the target.
    planted={2: ("red", "cube"), 5: ("blue", "ball")},
    instruction="Bring the red cube and the blue ball to the delivery pad",
    concurrent=True,
    max_steps=5200,
    decimation=5,
    gif_ss=2.5,
    gif_dur=17.0,
    poster_t=10.0,
)
