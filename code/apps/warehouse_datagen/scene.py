"""scene.py — Seeded warehouse episode-plan sampler for F5 DART data collection.

Produces, from a (seed, episode_index) pair, a fully-specified :class:`EpisodePlan`
describing ONE teacher-driven rollout in the warehouse arena:

  * a seeded warehouse scene (``hero_layout`` or a validity-preserving
    ``sample_layout`` variation) with per-object colour/shape variation
    (``warehouse_scene_cfg``);
  * a random FREE spawn pose (x, y, yaw) sampled from the inflated occupancy
    grid — anywhere clear of walls/shelves/objects, not only the home bays;
  * a driving MODE:
      - ``route``     : an A*-planned + line-of-sight-shortcut waypoint route to
                        a warehouse destination (object spot / delivery pad /
                        random free cell) — yields the realistic warehouse
                        command distribution (straights, aisle turns, crossover
                        weaves, alcove entries);
      - ``primitive`` : a single line-of-sight-clear target point steered to
                        directly — reproduces the BASELINE DART command
                        primitives (turn-in-place when out of FOV, decel arcs,
                        straight approach), matching the original recipe's
                        command distribution.

Everything here is pure Python / NumPy + the planner (no MuJoCo model compile),
so the command-distribution logic is unit-testable without a GPU/simulator.
``warehouse_scene_cfg`` only assembles a dict; the model is compiled later in
``rollout.py``.
"""

from __future__ import annotations

import dataclasses
import math
from typing import List, Optional, Tuple

import numpy as np

from code.scene import derive_rng  # reuse baseline SeedSequence derivation
from code.warehouse.layout import CALLSIGNS, hero_layout, sample_layout
from code.warehouse.arena import _default_objects, warehouse_scene_cfg
from code.apps.warehouse_demo.planning import build_inflated_grid
from code.planner.astar import PathNotFoundError, plan_path, shortcut_path, path_length
from code.planner.grid import OccupancyGrid

Point = Tuple[float, float]

# Planner / clearance tunables (mirrors code.apps.warehouse_demo.nav_core.NavParams
# — the validated warehouse-nav values; kept local so we never mutate them).
GRID_RES: float = 0.10
INFLATE_RADIUS: float = 0.40
SNAP_RADIUS: float = 0.60

# Episode geometry constraints.
MIN_ROUTE_LEN_M: float = 2.0     # skip trivially-short routes
MAX_PLAN_RETRIES: int = 8        # re-sample goal this many times before fallback
PRIMITIVE_DIST_RANGE: Tuple[float, float] = (2.0, 7.0)


@dataclasses.dataclass(frozen=True)
class EpisodePlan:
    """A fully-specified plan for one warehouse DART episode.

    Attributes:
        seed: Base dataset seed.
        ep_idx: Episode index (drives the per-episode RNG).
        scene_cfg: Universal warehouse scene_cfg (compiled by ``build_warehouse_arena``).
        spawn_xy: Random free spawn (x, y) in world metres.
        spawn_yaw: Random spawn yaw (rad).
        mode: 'route' or 'primitive'.
        path: Route waypoints [(x, y), ...] (route mode) else None.
        fixed_target: Single steer-to target (primitive mode) else None.
        goal_xy: Final navigation goal (both modes).
        route_len: Planned route arc length (m); straight-line dist for primitive.
        instruction: Natural-language task string.
        layout_name: Layout identifier ('hero' / 'sample_*').
    """

    seed: int
    ep_idx: int
    scene_cfg: dict
    spawn_xy: Point
    spawn_yaw: float
    mode: str
    path: Optional[List[Point]]
    fixed_target: Optional[Point]
    goal_xy: Point
    route_len: float
    instruction: str
    layout_name: str


# ---------------------------------------------------------------------------
# Free-space helpers
# ---------------------------------------------------------------------------
def free_cells_world(grid: OccupancyGrid) -> np.ndarray:
    """Return an (N, 2) array of world (x, y) centres of every FREE cell."""
    free = np.argwhere(~grid.grid)  # (N, 2) of (iy, ix)
    ox, oy = grid.origin_xy
    xs = ox + free[:, 1] * grid.resolution
    ys = oy + free[:, 0] * grid.resolution
    return np.stack([xs, ys], axis=1)


def sample_free_pose(grid: OccupancyGrid, rng: np.random.Generator) -> Tuple[Point, float]:
    """Sample a uniform random FREE (x, y) cell + a random yaw in (-pi, pi]."""
    cells = free_cells_world(grid)
    i = int(rng.integers(len(cells)))
    x, y = float(cells[i, 0]), float(cells[i, 1])
    yaw = float(rng.uniform(-math.pi, math.pi))
    return (x, y), yaw


def los_free(grid: OccupancyGrid, a: Point, b: Point, step: float = 0.05) -> bool:
    """True if every sampled point on segment a->b lies in a free grid cell."""
    d = math.hypot(b[0] - a[0], b[1] - a[1])
    n = max(2, int(d / step) + 1)
    for k in range(n + 1):
        t = k / n
        px = a[0] + (b[0] - a[0]) * t
        py = a[1] + (b[1] - a[1]) * t
        if not grid.is_free((px, py)):
            return False
    return True


# ---------------------------------------------------------------------------
# Goal / target sampling
# ---------------------------------------------------------------------------
def _candidate_goals(scene_cfg: dict, grid: OccupancyGrid,
                     rng: np.random.Generator) -> List[Point]:
    """Warehouse destinations: object spots, delivery pad, and random free cells."""
    goals: List[Point] = []
    # Delivery pad centre (a real warehouse destination).
    for z in scene_cfg.get("zones", []):
        if z.get("name") == "delivery":
            goals.append((float(z["cx"]), float(z["cy"])))
    # Object spots (the fetch targets).
    for obj in scene_cfg.get("objects", []):
        goals.append((float(obj["x"]), float(obj["y"])))
    # A few random free cells for open-lane diversity.
    cells = free_cells_world(grid)
    for _ in range(4):
        c = cells[int(rng.integers(len(cells)))]
        goals.append((float(c[0]), float(c[1])))
    rng.shuffle(goals)
    return goals


def _plan_route(scene_cfg: dict, spawn_xy: Point, goal_xy: Point) -> Optional[List[Point]]:
    """Inflated-grid A* + LoS shortcut to goal_xy; None if unreachable/too short."""
    grid = build_inflated_grid(scene_cfg, GRID_RES, INFLATE_RADIUS, goal_xy=goal_xy)
    try:
        raw = plan_path(grid, spawn_xy, goal_xy, snap_radius_m=SNAP_RADIUS)
    except PathNotFoundError:
        return None
    path = shortcut_path(grid, raw)
    if path_length(path) < MIN_ROUTE_LEN_M:
        return None
    return path


def _sample_primitive_target(grid: OccupancyGrid, spawn_xy: Point,
                             rng: np.random.Generator) -> Point:
    """A line-of-sight-clear target at a random bearing/distance (baseline-style).

    Random world bearing (independent of spawn yaw) so the target is frequently
    outside the initial FOV -> reproduces the baseline turn-in-place + straight
    command primitive. Falls back to the nearest clear short segment.
    """
    lo, hi = PRIMITIVE_DIST_RANGE
    for _ in range(40):
        d = float(rng.uniform(lo, hi))
        ang = float(rng.uniform(-math.pi, math.pi))
        tx = spawn_xy[0] + d * math.cos(ang)
        ty = spawn_xy[1] + d * math.sin(ang)
        if grid.is_free((tx, ty)) and los_free(grid, spawn_xy, (tx, ty)):
            return (tx, ty)
    # Fallback: short clear step in the least-cluttered cardinal direction.
    for d in (1.5, 1.0, 0.8):
        for ang in (0.0, math.pi / 2, math.pi, -math.pi / 2):
            tx = spawn_xy[0] + d * math.cos(ang)
            ty = spawn_xy[1] + d * math.sin(ang)
            if grid.is_free((tx, ty)) and los_free(grid, spawn_xy, (tx, ty)):
                return (tx, ty)
    return (spawn_xy[0] + 0.5, spawn_xy[1])


# ---------------------------------------------------------------------------
# Top-level episode-plan sampler
# ---------------------------------------------------------------------------
def sample_episode_plan(
    seed: int,
    ep_idx: int,
    *,
    primitive_frac: float = 0.3,
    layout_vary_frac: float = 0.85,
) -> EpisodePlan:
    """Deterministically sample the full plan for episode ``ep_idx``.

    The plan is a pure function of (seed, ep_idx): identical inputs always yield
    an identical plan (required for episode-level idempotency + determinism).

    Args:
        seed: Base dataset seed.
        ep_idx: Episode index.
        primitive_frac: Fraction of episodes driven as baseline-style
            direct-steer 'primitive' episodes (rest are A* 'route' episodes).
        layout_vary_frac: Fraction of episodes using a jittered ``sample_layout``
            (rest use the fixed ``hero_layout``).

    Returns:
        A fully-specified :class:`EpisodePlan`.
    """
    rng = derive_rng(seed, ep_idx)

    layout = sample_layout(rng) if rng.random() < layout_vary_frac else hero_layout()

    callsign = CALLSIGNS[int(rng.integers(len(CALLSIGNS)))]
    objects = _default_objects(layout, rng)
    target_index = int(rng.integers(len(objects))) if objects else 0
    scene_cfg = warehouse_scene_cfg(
        layout, robot=callsign, objects=objects, rng=rng, target_index=target_index,
    )

    # Spawn: random free pose anywhere in the hall (NOT just the bays).
    grid0 = build_inflated_grid(scene_cfg, GRID_RES, INFLATE_RADIUS)
    spawn_xy, spawn_yaw = sample_free_pose(grid0, rng)

    mode = "primitive" if rng.random() < primitive_frac else "route"

    if mode == "route":
        path = None
        goal_xy = spawn_xy
        for goal in _candidate_goals(scene_cfg, grid0, rng)[:MAX_PLAN_RETRIES]:
            if math.hypot(goal[0] - spawn_xy[0], goal[1] - spawn_xy[1]) < MIN_ROUTE_LEN_M:
                continue
            p = _plan_route(scene_cfg, spawn_xy, goal)
            if p is not None:
                path, goal_xy = p, goal
                break
        if path is None:
            # Fallback to a primitive episode (guarantees every index produces data).
            mode = "primitive"

    if mode == "primitive":
        target = _sample_primitive_target(grid0, spawn_xy, rng)
        return EpisodePlan(
            seed=seed, ep_idx=ep_idx, scene_cfg=scene_cfg,
            spawn_xy=spawn_xy, spawn_yaw=spawn_yaw, mode="primitive",
            path=None, fixed_target=target, goal_xy=target,
            route_len=float(math.hypot(target[0] - spawn_xy[0], target[1] - spawn_xy[1])),
            instruction=scene_cfg["instruction"], layout_name=scene_cfg["layout_name"],
        )

    return EpisodePlan(
        seed=seed, ep_idx=ep_idx, scene_cfg=scene_cfg,
        spawn_xy=spawn_xy, spawn_yaw=spawn_yaw, mode="route",
        path=[(float(x), float(y)) for x, y in path], fixed_target=None,
        goal_xy=(float(goal_xy[0]), float(goal_xy[1])),
        route_len=float(path_length(path)),
        instruction=scene_cfg["instruction"], layout_name=scene_cfg["layout_name"],
    )
