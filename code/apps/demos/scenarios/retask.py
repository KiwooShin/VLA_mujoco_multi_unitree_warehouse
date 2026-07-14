"""retask — mid-mission order change: the owner aborts and switches targets.

Alpha is fetching the red cube when the user changes their mind ("actually, bring
me the yellow ball instead"). Alpha stands down its helpers, drops the abandoned
approach, re-delegates a search for the new object and delivers THAT — the mission
result reflects the final task. On the BEV the old red-cube ring clears and a new
ring appears at the yellow ball once it is located (the unmistakable ring switch).
Full learned stack (GROUND_NET + VLA), fixed four-room layout, random spawns.

The two planted objects are the only cube and the only ball in the manifest (the
filler pool is cones/cylinders only), so each order is unambiguous — the demo
isolates the re-task beat cleanly.
"""

from __future__ import annotations

from code.apps.demos.scenarios.core import Scenario

# Cones/cylinders only -> the planted red cube and yellow ball are each unique.
_FILLERS = (("orange", "cylinder"), ("green", "cone"), ("purple", "cylinder"),
            ("cyan", "cone"), ("green", "cylinder"), ("orange", "cone"),
            ("purple", "cone"), ("cyan", "cylinder"))

SCENARIO = Scenario(
    name="retask",
    title="Mid-Mission Re-Task",
    description="User changes the order mid-fetch: Alpha aborts the cube and switches to the ball",
    layout_kind="rooms",
    spawn_seed=1,
    planted={3: ("red", "cube"), 7: ("yellow", "ball")},
    filler_pool=_FILLERS,
    instruction="Alpha, fetch the red cube to the delivery pad",
    retasks=((1900, "Alpha, actually bring me the yellow ball instead"),),
    max_steps=7500,
    decimation=7,
    gif_ss=4.0,
    gif_dur=20.0,
    poster_t=13.0,
)
