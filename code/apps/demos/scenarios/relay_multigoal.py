"""relay_multigoal — one owner fetches two objects in sequence, reusing helpers.

"Alpha, bring the red cube, then the blue ball to the delivery pad" is a sequential
multi-goal order: Alpha completes the red-cube leg (delegating a search, fetching,
delivering) and only THEN begins the blue-ball leg, re-delegating the same teammates
as searchers for the second object. TASK_COMPLETE is reported once, after the last
leg. Runs on a second never-seen sampled layout (a "rooms variant") with random
spawns and the full learned stack (GROUND_NET + VLA).
"""

from __future__ import annotations

from code.apps.demos.scenarios.core import Scenario, deep_spots


def _plant_two(layout):
    """Plant the red cube and blue ball in two different deep rooms."""
    a, b = deep_spots(layout, 2)
    return {a: ("red", "cube"), b: ("blue", "ball")}


SCENARIO = Scenario(
    name="relay_multigoal",
    title="Sequential Relay: Two Legs, One Owner",
    description="Fetch the red cube, then the blue ball - helpers re-used across both legs",
    layout_kind="sampled",
    sample_seed=4,
    spawn_seed=4,
    planted_fn=_plant_two,
    instruction="Alpha, bring the red cube, then the blue ball to the delivery pad",
    max_steps=5500,
    decimation=6,
    gif_ss=4.0,
    gif_dur=22.0,
    poster_t=14.0,
)
