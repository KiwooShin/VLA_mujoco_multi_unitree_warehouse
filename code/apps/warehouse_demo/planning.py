"""planning.py — Occupancy-grid assembly + clearance geometry for nav.

Pure (simulator-free) helpers shared by ``nav_rollout``: reconstruct the
planner world from a warehouse ``scene_cfg`` (single source of truth), add the
non-goal object geoms as obstacles the walls-only grid omits, inflate for robot
clearance, and measure pelvis-to-wall clearance. Kept separate from the rollout
loop so both stay well under the 500-line budget and the math is unit-testable
without stepping physics.
"""

from __future__ import annotations

import math
from typing import Dict, Optional, Sequence, Tuple

from code.planner.grid import OccupancyGrid, inflate
from code.warehouse.layout import WallSpec, WarehouseLayout
from code.warehouse.occupancy import occupancy_grid

Point = Tuple[float, float]


def point_obb_distance(px: float, py: float, wall: Dict[str, float]) -> float:
    """Shortest distance from a point to a (possibly yawed) wall footprint.

    Args:
        px: Point x (m).
        py: Point y (m).
        wall: Wall dict with ``cx``, ``cy``, ``half_x``, ``half_y``, ``yaw``.

    Returns:
        Distance in meters; 0.0 if the point is inside the rectangle.
    """
    dx = px - float(wall["cx"])
    dy = py - float(wall["cy"])
    yaw = float(wall.get("yaw", 0.0))
    c, s = math.cos(yaw), math.sin(yaw)
    lx = dx * c + dy * s
    ly = -dx * s + dy * c
    ox = max(abs(lx) - float(wall["half_x"]), 0.0)
    oy = max(abs(ly) - float(wall["half_y"]), 0.0)
    return math.hypot(ox, oy)


def min_wall_clearance(px: float, py: float,
                       walls: Sequence[Dict[str, float]]) -> float:
    """Min distance from (px, py) to any wall footprint (inf if no walls)."""
    best = float("inf")
    for w in walls:
        d = point_obb_distance(px, py, w)
        if d < best:
            best = d
    return best


def layout_from_scene_cfg(scene_cfg: dict) -> WarehouseLayout:
    """Reconstruct a WarehouseLayout (walls + hall dims) from a scene_cfg.

    Hall extents come from the ``hall_x``/``hall_y`` scene_cfg keys when
    present (exact); older cfgs fall back to reconstruction from the extreme
    wall centers (the perimeter walls). Either way the occupancy grid derives
    from the SAME wall list the MJCF was built from (single source of truth).
    """
    walls_d = scene_cfg.get("walls", [])
    walls = [WallSpec(cx=w["cx"], cy=w["cy"], half_x=w["half_x"],
                      half_y=w["half_y"], yaw=w.get("yaw", 0.0),
                      height=w.get("height", 2.5), name=w.get("name", ""))
             for w in walls_d]
    hall_x = float(scene_cfg.get(
        "hall_x", 2.0 * max((abs(w.cx) for w in walls), default=8.0)))
    hall_y = float(scene_cfg.get(
        "hall_y", 2.0 * max((abs(w.cy) for w in walls), default=6.0)))
    return WarehouseLayout(hall_x=hall_x, hall_y=hall_y, walls=walls)


def add_object_obstacles(
    og: OccupancyGrid, objects: Sequence[dict], goal_xy: Optional[Point],
    exclude_radius: float,
) -> OccupancyGrid:
    """Return a copy of ``og`` with every non-goal object marked occupied.

    The occupancy grid rasterizes walls only, but the object geoms placed at the
    other ``object_spots`` are real physical obstacles: a path that ignores them
    walks the robot straight into a cube/cone and destabilizes the walk policy.
    Each object is stamped as a small disk of its physical radius; the object
    within ``exclude_radius`` of ``goal_xy`` (the fetch target) is left free so
    the robot can still approach it.

    Args:
        og: Walls-only occupancy grid.
        objects: Scene objects (each with ``x``, ``y``, ``size``).
        goal_xy: The navigation goal; the object here stays approachable.
        exclude_radius: Objects within this distance of ``goal_xy`` are skipped.

    Returns:
        A new :class:`OccupancyGrid` with object cells added.
    """
    grid = og.grid.copy()
    ny, nx = grid.shape
    res = og.resolution
    ox0, oy0 = og.origin_xy
    for obj in objects:
        ox, oy = float(obj["x"]), float(obj["y"])
        if goal_xy is not None and math.hypot(ox - goal_xy[0],
                                              oy - goal_xy[1]) <= exclude_radius:
            continue
        r_cells = max(1, int(math.ceil((float(obj.get("size", 0.2)) / 2.0) / res)))
        cix = int(round((ox - ox0) / res))
        ciy = int(round((oy - oy0) / res))
        for dy in range(-r_cells, r_cells + 1):
            iy = ciy + dy
            if iy < 0 or iy >= ny:
                continue
            for dx in range(-r_cells, r_cells + 1):
                ix = cix + dx
                if 0 <= ix < nx and dx * dx + dy * dy <= r_cells * r_cells:
                    grid[iy, ix] = True
    return OccupancyGrid(grid, res, og.origin_xy)


def build_inflated_grid(
    scene_cfg: dict, resolution: float, inflate_radius: float, *,
    goal_xy: Optional[Point] = None, exclude_radius: float = 0.5,
) -> OccupancyGrid:
    """Rasterize the walls (+ non-goal objects) and dilate by robot clearance.

    Args:
        scene_cfg: Warehouse scene_cfg (``walls`` + ``objects``).
        resolution: Grid cell size (m).
        inflate_radius: Robot-clearance dilation radius (m).
        goal_xy: Navigation goal; the object there is kept approachable.
        exclude_radius: Skip objects within this distance of ``goal_xy``.

    Returns:
        The inflated :class:`OccupancyGrid` A* plans over.
    """
    layout = layout_from_scene_cfg(scene_cfg)
    og = occupancy_grid(layout, resolution=resolution)
    objects = scene_cfg.get("objects", [])
    if objects:
        og = add_object_obstacles(og, objects, goal_xy, exclude_radius)
    return inflate(og, inflate_radius)
