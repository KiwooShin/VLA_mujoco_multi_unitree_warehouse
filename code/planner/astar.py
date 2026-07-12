"""Grid A* path planning over an :class:`~code.planner.grid.OccupancyGrid`.

Provides shortest collision-free routes for warehouse navigation:

* :func:`plan_path` — 8-connected grid A* with no corner cutting, an octile
  heuristic and deterministic tie-breaking. Endpoints that land on an occupied
  cell are snapped to the nearest free cell within a small radius before the
  search gives up.
* :func:`shortcut_path` — greedy line-of-sight smoothing that uses a *supercover*
  grid traversal (every cell a segment touches is checked), producing a minimal
  waypoint path that stays entirely in free cells.
* :func:`path_length` — polyline arc length.

Frame/index conventions follow :mod:`code.planner.grid`: ``grid[iy, ix]`` is
True where occupied, cells are indexed ``(iy, ix)`` and world points are
``(x, y)`` in meters. Returned waypoints are the world centers of the planned
cells (from snapped start to snapped goal, inclusive).
"""

from __future__ import annotations

import heapq
import math
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from code.planner.grid import OccupancyGrid

_SQRT2: float = math.sqrt(2.0)

# 8-connected moves as (d_iy, d_ix, step_cost). Cardinals first (deterministic,
# and preferred on ties because heapq compares the trailing cell index anyway).
_MOVES: Tuple[Tuple[int, int, float], ...] = (
    (-1, 0, 1.0),
    (1, 0, 1.0),
    (0, -1, 1.0),
    (0, 1, 1.0),
    (-1, -1, _SQRT2),
    (-1, 1, _SQRT2),
    (1, -1, _SQRT2),
    (1, 1, _SQRT2),
)

Cell = Tuple[int, int]
Point = Tuple[float, float]


class PathNotFoundError(RuntimeError):
    """Raised when A* cannot connect start to goal.

    The message reports the (snapped) start/goal cells and free-space
    statistics to make dead grids and enclosed goals easy to diagnose.
    """


def _free_mask(og: OccupancyGrid) -> np.ndarray:
    """Returns the boolean free-space mask (True where traversable)."""
    return ~og.grid


def _raw_cell(og: OccupancyGrid, xy: Point) -> Cell:
    """Rounds a world point to a cell index, clamped into grid bounds."""
    ix = int(round((xy[0] - og.origin_xy[0]) / og.resolution))
    iy = int(round((xy[1] - og.origin_xy[1]) / og.resolution))
    ny, nx = og.grid.shape
    ix = min(max(ix, 0), nx - 1)
    iy = min(max(iy, 0), ny - 1)
    return iy, ix


def _nearest_free_cell(
    og: OccupancyGrid,
    free: np.ndarray,
    xy: Point,
    snap_radius_m: float,
) -> Optional[Cell]:
    """Finds the free cell nearest to ``xy`` within ``snap_radius_m``.

    Args:
        og: The occupancy grid.
        free: Boolean free-space mask (``~og.grid``).
        xy: World point in meters.
        snap_radius_m: Maximum snap distance in meters (>= 0).

    Returns:
        The nearest free ``(iy, ix)`` cell, or None if none lies within the
        radius (including when ``xy`` itself is already free — returned then).
    """
    iy0, ix0 = _raw_cell(og, xy)
    if free[iy0, ix0]:
        return iy0, ix0
    ny, nx = og.grid.shape
    r_cells = int(math.ceil(snap_radius_m / og.resolution))
    best: Optional[Cell] = None
    best_d2 = float("inf")
    r2 = snap_radius_m * snap_radius_m + 1e-9
    for dy in range(-r_cells, r_cells + 1):
        iy = iy0 + dy
        if iy < 0 or iy >= ny:
            continue
        for dx in range(-r_cells, r_cells + 1):
            ix = ix0 + dx
            if ix < 0 or ix >= nx or not free[iy, ix]:
                continue
            cx = og.origin_xy[0] + ix * og.resolution
            cy = og.origin_xy[1] + iy * og.resolution
            d2 = (cx - xy[0]) ** 2 + (cy - xy[1]) ** 2
            if d2 <= r2 and d2 < best_d2:
                best_d2 = d2
                best = (iy, ix)
    return best


def _octile(a: Cell, b: Cell) -> float:
    """Octile distance heuristic between two cells (admissible for 8-conn)."""
    dy = abs(a[0] - b[0])
    dx = abs(a[1] - b[1])
    lo, hi = (dy, dx) if dy < dx else (dx, dy)
    return (hi - lo) + _SQRT2 * lo


def plan_path(
    og: OccupancyGrid,
    start_xy: Point,
    goal_xy: Point,
    *,
    snap_radius_m: float = 0.3,
) -> List[Point]:
    """Plans a shortest collision-free path with grid A*.

    Uses 8-connected moves with strict no-corner-cutting (a diagonal step is
    only allowed when both shared-edge cardinal cells are also free), an octile
    heuristic, and fully deterministic tie-breaking (ties resolve on heuristic
    then on cell index, so the same inputs always yield the same path).

    If the start or goal cell is occupied it is snapped to the nearest free cell
    within ``snap_radius_m`` before the search runs.

    Args:
        og: Occupancy grid to plan over (inflate it beforehand for clearance).
        start_xy: World start point (x, y) in meters.
        goal_xy: World goal point (x, y) in meters.
        snap_radius_m: Snap radius for occupied endpoints in meters.

    Returns:
        World-coordinate waypoints (cell centers) from the snapped start to the
        snapped goal, inclusive. A start/goal in the same cell yields a single
        waypoint.

    Raises:
        PathNotFoundError: If an endpoint cannot be snapped to free space, or no
            collision-free route connects them.
    """
    free = _free_mask(og)
    ny, nx = og.grid.shape

    start = _nearest_free_cell(og, free, start_xy, snap_radius_m)
    goal = _nearest_free_cell(og, free, goal_xy, snap_radius_m)
    n_free = int(free.sum())
    if start is None or goal is None:
        raise PathNotFoundError(
            f"cannot snap endpoint to free space within {snap_radius_m} m: "
            f"start_xy={start_xy} -> {start}, goal_xy={goal_xy} -> {goal}; "
            f"free cells={n_free}/{ny * nx}"
        )
    if start == goal:
        return [og.cell_to_world(start)]

    came_from: Dict[Cell, Cell] = {}
    g_score: Dict[Cell, float] = {start: 0.0}
    closed: set = set()
    # Heap entry: (f, h, iy, ix) — deterministic total order (cells are unique).
    open_heap: List[Tuple[float, float, int, int]] = [
        (_octile(start, goal), _octile(start, goal), start[0], start[1])
    ]

    while open_heap:
        _, _, ciy, cix = heapq.heappop(open_heap)
        current = (ciy, cix)
        if current in closed:
            continue
        if current == goal:
            return _reconstruct(og, came_from, current)
        closed.add(current)
        gc = g_score[current]
        for diy, dix, cost in _MOVES:
            niy = ciy + diy
            nix = cix + dix
            if niy < 0 or niy >= ny or nix < 0 or nix >= nx:
                continue
            if not free[niy, nix]:
                continue
            if diy != 0 and dix != 0:
                # No corner cutting: both shared cardinal cells must be free.
                if not free[ciy + diy, cix] or not free[ciy, cix + dix]:
                    continue
            neighbor = (niy, nix)
            if neighbor in closed:
                continue
            tentative = gc + cost
            if tentative < g_score.get(neighbor, float("inf")):
                came_from[neighbor] = current
                g_score[neighbor] = tentative
                h = _octile(neighbor, goal)
                heapq.heappush(open_heap, (tentative + h, h, niy, nix))

    raise PathNotFoundError(
        f"no collision-free route from cell {start} to {goal}; "
        f"expanded {len(closed)} cells, free cells={n_free}/{ny * nx}"
    )


def _reconstruct(
    og: OccupancyGrid,
    came_from: Dict[Cell, Cell],
    goal: Cell,
) -> List[Point]:
    """Rebuilds the world-coordinate path from the came-from map."""
    cells: List[Cell] = [goal]
    node = goal
    while node in came_from:
        node = came_from[node]
        cells.append(node)
    cells.reverse()
    return [og.cell_to_world(c) for c in cells]


def _supercover_cells(cell0: Cell, cell1: Cell) -> List[Cell]:
    """Returns every grid cell the segment ``cell0``->``cell1`` touches.

    Implements Eugen Dedu's supercover variant of Bresenham: at exact corner
    crossings BOTH straddled cells are emitted, so the returned set is a
    conservative superset of the cells the continuous segment overlaps. Cells
    are indexed ``(iy, ix)``.
    """
    iy0, ix0 = cell0
    iy1, ix1 = cell1
    y, x = iy0, ix0
    cells: List[Cell] = [(y, x)]
    dy = iy1 - iy0
    dx = ix1 - ix0
    ystep = 1 if dy >= 0 else -1
    xstep = 1 if dx >= 0 else -1
    dy = abs(dy)
    dx = abs(dx)
    ddy = 2 * dy
    ddx = 2 * dx
    if ddx >= ddy:  # x is the driving axis (|slope| <= 1)
        error = dx
        errorprev = dx
        for _ in range(dx):
            x += xstep
            error += ddy
            if error > ddx:
                y += ystep
                error -= ddx
                total = error + errorprev
                if total < ddx:
                    cells.append((y - ystep, x))
                elif total > ddx:
                    cells.append((y, x - xstep))
                else:  # exact corner: both straddled cells
                    cells.append((y - ystep, x))
                    cells.append((y, x - xstep))
            cells.append((y, x))
            errorprev = error
    else:  # y is the driving axis (|slope| > 1)
        error = dy
        errorprev = dy
        for _ in range(dy):
            y += ystep
            error += ddx
            if error > ddy:
                x += xstep
                error -= ddy
                total = error + errorprev
                if total < ddy:
                    cells.append((y, x - xstep))
                elif total > ddy:
                    cells.append((y - ystep, x))
                else:  # exact corner: both straddled cells
                    cells.append((y, x - xstep))
                    cells.append((y - ystep, x))
            cells.append((y, x))
            errorprev = error
    return cells


def _segment_clear(free: np.ndarray, cell0: Cell, cell1: Cell) -> bool:
    """True if every supercover cell of the segment is in-bounds and free."""
    ny, nx = free.shape
    for iy, ix in _supercover_cells(cell0, cell1):
        if iy < 0 or iy >= ny or ix < 0 or ix >= nx or not free[iy, ix]:
            return False
    return True


def shortcut_path(og: OccupancyGrid, path: Sequence[Point]) -> List[Point]:
    """Greedily smooths a path via supercover line-of-sight checks.

    Walks the path keeping only waypoints needed to preserve a collision-free
    route: from each kept waypoint it advances to the farthest later waypoint
    reachable by a clear straight segment. Because each retained segment is a
    straight chord of the sub-path it replaces, the smoothed path is never
    longer than the input (triangle inequality) and stays entirely in free
    cells.

    Args:
        og: Occupancy grid the path lives in.
        path: World-coordinate waypoints (e.g. from :func:`plan_path`).

    Returns:
        A minimal-waypoint world path with the same endpoints.
    """
    if len(path) <= 2:
        return [tuple(p) for p in path]

    free = _free_mask(og)
    cells: List[Cell] = [_raw_cell(og, p) for p in path]
    n = len(path)
    result: List[Point] = [tuple(path[0])]
    i = 0
    while i < n - 1:
        j = n - 1
        while j > i + 1 and not _segment_clear(free, cells[i], cells[j]):
            j -= 1
        result.append(tuple(path[j]))
        i = j
    return result


def path_length(path: Sequence[Point]) -> float:
    """Returns the total Euclidean arc length of a polyline path (meters)."""
    total = 0.0
    for a, b in zip(path, path[1:]):
        total += math.hypot(b[0] - a[0], b[1] - a[1])
    return total
