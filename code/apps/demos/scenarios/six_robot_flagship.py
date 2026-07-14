"""six_robot_flagship — six robots, big hall, ambiguous "a ball" -> clarify -> search.

The flagship: six named humanoids (Alpha..Foxtrot) in the 24x16 m six-room hall,
all starting at random poses. A fleet-addressed ambiguous order ("someone bring me
a ball") is CLARIFIED by the allocator — the manifest holds a red, a green and a
blue ball — and once the user answers ("the green one") the allocator assigns the
path-shortest owner, which delegates a search across the rooms. With more teammates
than rooms, one is held back as a reserve searcher. Full learned stack
(GROUND_NET + VLA).
"""

from __future__ import annotations

from code.comms.messages import ObjectQuery
from code.apps.demos.scenarios.core import Scenario

# Cubes/cones/cylinders only -> the three planted balls are the only balls, so
# "a ball" is ambiguous over exactly {red, green, blue} ball.
_FILLERS = (("orange", "cylinder"), ("purple", "cube"), ("cyan", "cone"),
            ("yellow", "cylinder"), ("orange", "cube"), ("purple", "cone"),
            ("cyan", "cylinder"), ("yellow", "cube"), ("green", "cone"),
            ("orange", "cone"), ("purple", "cylinder"))

SCENARIO = Scenario(
    name="six_robot_flagship",
    title="Six-Robot Flagship: Clarify, then Search",
    description="Fleet order 'bring me a ball' - allocator clarifies, assigns, and searches with a reserve",
    layout_kind="rooms6",
    spawn_seed=2,
    planted={2: ("red", "ball"), 10: ("green", "ball"), 6: ("blue", "ball")},
    filler_pool=_FILLERS,
    instruction="someone bring me a ball to the delivery pad",
    replies=("the green one",),
    ambiguity_query=ObjectQuery(None, "ball"),
    ambiguity_options=("red ball", "green ball", "blue ball"),
    # Six robots fire ~21 messages in the first <10 sim-steps; hide the ten
    # visibility "can you see it? / no" lines so the opening CLARIFY + the
    # delegation stay legible in the auto-scrolling panel (presentation only).
    hide_performatives=("QUERY_VISIBILITY", "REPORT_VISIBILITY"),
    max_steps=5200,
    decimation=6,
    gif_ss=2.5,
    gif_dur=22.0,
    poster_t=4.5,
)
