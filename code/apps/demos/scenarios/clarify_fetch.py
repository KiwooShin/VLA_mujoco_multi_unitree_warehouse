"""clarify_fetch — ambiguous "bring me the cube" -> CLARIFY -> delegated fetch.

The warehouse holds three cubes (red, blue, yellow) plus non-cube clutter, so
"Alpha, bring me the cube" is manifest-ambiguous: Alpha asks the user which cube,
the scripted user answers "the red one", and Alpha delegates a room-to-room search
for the red cube, then fetches and delivers it. Full learned stack (GROUND_NET
perception + VLA locomotion) on the fixed four-room layout with random spawns.
"""

from __future__ import annotations

from code.comms.messages import ObjectQuery
from code.apps.demos.scenarios.core import Scenario

SCENARIO = Scenario(
    name="clarify_fetch",
    title="Ambiguous Order -> Clarify -> Fetch",
    description="Three cubes in the manifest: Alpha asks which one, then delegates the search",
    layout_kind="rooms",
    spawn_seed=1,
    planted={2: ("red", "cube"), 9: ("blue", "cube"), 6: ("yellow", "cube")},
    instruction="Alpha, bring me the cube",
    replies=("the red one",),
    ambiguity_query=ObjectQuery(None, "cube"),
    ambiguity_options=("red cube", "blue cube", "yellow cube"),
    max_steps=4800,
    decimation=4,
    gif_ss=3.0,
    gif_dur=20.0,
    poster_t=6.0,
)
