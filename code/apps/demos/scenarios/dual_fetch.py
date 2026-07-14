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
    spawn_seed=2,
    planted={4: ("red", "cube"), 7: ("blue", "ball")},
    instruction="Bring the red cube and the blue ball to the delivery pad",
    concurrent=True,
    max_steps=3200,
    decimation=4,
    gif_ss=2.5,
    gif_dur=18.0,
    poster_t=5.0,
)
