"""unseen_map — delegated exploration on a NEVER-SEEN sampled rooms layout.

``sample_rooms_layout(seed)`` draws a fresh four-room warehouse the fleet has
never trained or been tuned on (randomized doorways, shelves, object spots and
delivery pad). Robots start at random reachability-checked poses; the red cube is
planted in a deep, shelf-occluded corner out of every initial sightline, so the
owner must delegate a room-to-room search to find it before fetching. Full learned
stack (GROUND_NET + VLA) — a clean generalization story.
"""

from __future__ import annotations

from code.apps.demos.scenarios.core import Scenario, deep_spots


def _plant_red_cube(layout):
    """Plant the single red cube in the deepest searchable-room corner."""
    spot = deep_spots(layout, 1)[0]
    return {spot: ("red", "cube")}


SCENARIO = Scenario(
    name="unseen_map",
    title="Exploration on a Never-Seen Map",
    description="A freshly sampled warehouse: delegate a room-to-room search for the hidden cube",
    layout_kind="sampled",
    sample_seed=1,
    spawn_seed=1,
    planted_fn=_plant_red_cube,
    instruction="Alpha, fetch the red cube to the delivery pad",
    max_steps=6500,
    decimation=8,
    gif_ss=4.0,
    gif_dur=22.0,
    poster_t=16.0,
)
