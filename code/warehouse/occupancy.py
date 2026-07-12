"""occupancy.py — Rasterize a warehouse layout into a planner occupancy grid.

The warehouse wall list (``code.warehouse.layout``) and the A* planner
(``code.planner``) share the single :class:`code.planner.grid.OccupancyGrid`
contract, so the simulated MJCF geometry and the planning world derive from the
same source of truth and can never skew (docs/multi_plan.md sec 2 / sec 4).

Public API
----------
occupancy_grid(layout, resolution=0.1) -> OccupancyGrid
"""

from __future__ import annotations

import numpy as np

from code.planner.grid import OccupancyGrid
from code.warehouse.layout import WarehouseLayout


def occupancy_grid(layout: WarehouseLayout,
                   resolution: float = 0.1) -> OccupancyGrid:
    """Rasterize every wall of ``layout`` onto a hall-covering occupancy grid.

    The grid tiles the full hall footprint (x in [-hall_x/2, hall_x/2], y in
    [-hall_y/2, hall_y/2]). A cell is occupied if its centre lies inside any
    wall footprint, including yawed (rotated) rectangles. Zones are visual-only
    and are never rasterized. Perimeter walls, being centred on the hall edge,
    contribute a thin occupied border.

    Args:
        layout: Source layout whose ``walls`` are rasterized.
        resolution: Cell edge length in metres (> 0).

    Returns:
        An :class:`OccupancyGrid` with ``grid[iy, ix]`` True where occupied,
        ``origin_xy`` at the centre of cell (0, 0).

    Raises:
        ValueError: If ``resolution`` is not positive.
    """
    if resolution <= 0.0:
        raise ValueError(f"resolution must be > 0, got {resolution}")

    half_x = layout.hall_x / 2.0
    half_y = layout.hall_y / 2.0
    nx = int(round(layout.hall_x / resolution))
    ny = int(round(layout.hall_y / resolution))
    origin_x = -half_x + resolution / 2.0
    origin_y = -half_y + resolution / 2.0

    xs = origin_x + np.arange(nx) * resolution  # (nx,)
    ys = origin_y + np.arange(ny) * resolution  # (ny,)
    grid_x, grid_y = np.meshgrid(xs, ys)  # (ny, nx) each

    occ = np.zeros((ny, nx), dtype=np.bool_)
    for wall in layout.walls:
        dx = grid_x - wall.cx
        dy = grid_y - wall.cy
        c, s = np.cos(wall.yaw), np.sin(wall.yaw)
        lx = dx * c + dy * s
        ly = -dx * s + dy * c
        inside = (np.abs(lx) <= wall.half_x) & (np.abs(ly) <= wall.half_y)
        occ |= inside

    return OccupancyGrid(occ, float(resolution), (float(origin_x), float(origin_y)))
