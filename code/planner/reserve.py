"""reserve.py — Space-time reservation planning for proactive multi-robot routing.

The baseline fleet resolves robot-robot conflicts *reactively*: the mutual-proximity
pause (``code.fleet.fleet``) freezes the lower-priority robot inside 1.0 m and
resumes it at 1.2 m. That is safe (0 falls) but wastes makespan and reads as a
heuristic. This module adds a *proactive* layer on TOP of the same occupancy grid
the A* planner already uses (``code.planner.grid`` / ``code.planner.astar`` stay
untouched):

* :class:`ReservationTable` — a time-expanded occupancy map. Each robot books the
  ``(cell, time)`` space-time cells its planned route sweeps, inflated ``+/- t_pad``
  control steps and to the cell neighbours inside its footprint radius. A later
  robot queries the table so its own plan can steer clear.
* :func:`plan_path_st` — A* over ``(cell, time)`` with a *wait* action that routes a
  robot around another robot's booked space-time (or waits for it to clear),
  falling back to plain :func:`code.planner.astar.plan_path` when the search
  exceeds a node budget (the caller is told via :attr:`STResult.fell_back`).

The reservation layer NEVER replaces the proximity pause; it only reshapes routes
before stepping. The pause stays armed underneath as the safety net. Everything
here is pure Python/NumPy — deterministic and unit-testable without a simulator.

Time base
---------
All times are integer **control steps** (50 Hz; ``code.sim.teacher.CONTROL_DT`` =
0.02 s). A conservative speed model (m/s) converts a cell edge into a step count:
``steps_per_cell = round(cell_edge_m / (speed_mps / 50))``. A slower model speed
means longer (more conservative) reservation windows.
"""

from __future__ import annotations

import dataclasses
import heapq
from typing import Dict, List, Optional, Set, Tuple

from code.planner.astar import (
    _MOVES,
    _SQRT2,
    _free_mask,
    _nearest_free_cell,
    _raw_cell,
    PathNotFoundError,
    plan_path,
)
from code.planner.grid import OccupancyGrid

Cell = Tuple[int, int]
Point = Tuple[float, float]

CONTROL_HZ: float = 50.0  # control steps per second (code.sim.teacher.CONTROL_DT)
DEFAULT_SPEED_MPS: float = 0.5  # conservative fleet walking speed (see fleet_eval calib)


def steps_per_cell(resolution: float, speed_mps: float) -> Tuple[int, int]:
    """Return (cardinal, diagonal) control steps to cross one grid cell.

    Args:
        resolution: Grid cell edge length (m).
        speed_mps: Conservative model walking speed (m/s); >0.

    Returns:
        (card, diag) step counts (each >= 1). ``diag`` covers a sqrt(2)-longer edge.
    """
    m_per_step = max(1e-6, float(speed_mps) / CONTROL_HZ)
    card = max(1, int(round(resolution / m_per_step)))
    diag = max(1, int(round(resolution * _SQRT2 / m_per_step)))
    return card, diag


def cell_times_for_path(
    cells: List[Cell], resolution: float, speed_mps: float,
) -> List[int]:
    """Cumulative step offsets (from 0) at which each cell of ``cells`` is reached.

    Cardinal edges cost ``card`` steps, diagonal edges ``diag`` (see
    :func:`steps_per_cell`). ``cell_times[0] == 0``.
    """
    card, diag = steps_per_cell(resolution, speed_mps)
    times: List[int] = [0]
    for a, b in zip(cells, cells[1:]):
        step = diag if (a[0] != b[0] and a[1] != b[1]) else card
        times.append(times[-1] + step)
    return times


class ReservationTable:
    """Time-expanded occupancy: which robot owns which ``(cell, time-window)``.

    A booking stamps, for every cell of a robot's route, a time window
    ``[t - t_pad, t + t_pad]`` (t = the step it occupies that cell) onto that cell
    AND onto the cell neighbours within ``footprint_radius_m`` (so a second robot
    keeps its centre roughly a footprint away in space-time). Bookings are keyed by
    ``robot_id`` so a robot can atomically :meth:`release` and rebook on every
    replan.
    """

    def __init__(
        self,
        resolution: float,
        *,
        footprint_radius_m: float = 0.30,
        t_pad: int = 6,
    ) -> None:
        """Build an empty table.

        Args:
            resolution: Grid cell edge (m) — must match the grid routes plan over.
            footprint_radius_m: Spatial inflation of each booking (m). Every path
                cell also reserves the disk of neighbours within this radius, so a
                second robot's centre stays ~a footprint clear of the booked path.
            t_pad: Temporal inflation (+/- steps) around each cell's occupancy time.
        """
        self.resolution = float(resolution)
        self.footprint_radius_m = float(footprint_radius_m)
        self.footprint_cells = int(round(footprint_radius_m / self.resolution))
        self.t_pad = int(t_pad)
        # cell -> list of (t_lo, t_hi, robot_id) closed intervals.
        self._cell_resv: Dict[Cell, List[Tuple[int, int, str]]] = {}
        # robot_id -> set of cells it has stamped (for O(touched) release).
        self._robot_cells: Dict[str, Set[Cell]] = {}
        # Precompute the footprint disk offsets once.
        r = self.footprint_cells
        self._disk: List[Cell] = [
            (dy, dx)
            for dy in range(-r, r + 1)
            for dx in range(-r, r + 1)
            if dy * dy + dx * dx <= r * r
        ]

    # ---- Mutation ----
    def reserve(
        self, cells: List[Cell], t_start: int, cell_times: List[int], robot_id: str,
    ) -> None:
        """Book a route's space-time cells for ``robot_id``.

        Args:
            cells: Grid cells (iy, ix) the route sweeps, start to goal.
            t_start: Absolute control step at which the route begins.
            cell_times: Step offsets (from ``t_start``) at each cell; parallel to
                ``cells``.
            robot_id: Owner id (a callsign); prior bookings for it are NOT cleared
                here — call :meth:`release` first to rebook.

        Raises:
            ValueError: If ``cells`` and ``cell_times`` differ in length.
        """
        if len(cells) != len(cell_times):
            raise ValueError(
                f"cells ({len(cells)}) and cell_times ({len(cell_times)}) "
                "must be the same length")
        touched = self._robot_cells.setdefault(robot_id, set())
        t0 = int(t_start)
        for (iy, ix), ct in zip(cells, cell_times):
            tc = t0 + int(ct)
            t_lo, t_hi = tc - self.t_pad, tc + self.t_pad
            for dy, dx in self._disk:
                c = (iy + dy, ix + dx)
                self._cell_resv.setdefault(c, []).append((t_lo, t_hi, robot_id))
                touched.add(c)

    def release(self, robot_id: str) -> None:
        """Drop every booking owned by ``robot_id`` (idempotent)."""
        cells = self._robot_cells.pop(robot_id, None)
        if not cells:
            return
        for c in cells:
            lst = self._cell_resv.get(c)
            if not lst:
                continue
            kept = [iv for iv in lst if iv[2] != robot_id]
            if kept:
                self._cell_resv[c] = kept
            else:
                del self._cell_resv[c]

    # ---- Query ----
    def is_reserved(
        self, cell: Cell, t: int, ignore_id: Optional[str] = None,
    ) -> bool:
        """True if any robot other than ``ignore_id`` books ``cell`` at step ``t``."""
        lst = self._cell_resv.get(cell)
        if not lst:
            return False
        for t_lo, t_hi, rid in lst:
            if rid == ignore_id:
                continue
            if t_lo <= t <= t_hi:
                return True
        return False

    def first_conflict(
        self, cells: List[Cell], t0: int, speed: float,
        ignore_id: Optional[str] = None,
    ) -> Optional[Tuple[int, Cell, int]]:
        """First space-time clash if a robot walks ``cells`` from step ``t0``.

        Times along ``cells`` come from the ``speed`` model (:func:`cell_times_for_path`).

        Returns:
            ``(index, cell, step)`` of the first reserved cell, or None if the whole
            route is clear of other robots' bookings.
        """
        times = cell_times_for_path(cells, self.resolution, speed)
        for i, (c, ct) in enumerate(zip(cells, times)):
            if self.is_reserved(c, t0 + ct, ignore_id):
                return (i, c, t0 + ct)
        return None

    def active_robots(self) -> Set[str]:
        """Set of robot ids that currently hold at least one booking."""
        return {rid for rid, cells in self._robot_cells.items() if cells}


@dataclasses.dataclass
class ReservationContext:
    """What a robot needs to plan-and-book against a shared :class:`ReservationTable`.

    Passed into :meth:`code.apps.warehouse_demo.nav_core.StepwiseNav.plan`; the
    caller books :attr:`STResult` afterwards.
    """

    table: ReservationTable
    t0: int
    speed: float
    robot_id: str


@dataclasses.dataclass
class STResult:
    """Outcome of :func:`plan_path_st`.

    Attributes:
        path: World waypoints (cell centres), snapped start to snapped goal.
        cells: Raw cell sequence (includes repeated cells for wait actions) — the
            sequence to :meth:`ReservationTable.reserve`.
        cell_times: Step offsets from ``t0`` at each entry of ``cells``.
        fell_back: True if the ST search hit the node budget and this is a plain
            :func:`plan_path` result instead.
        conflict_free: True if the returned route is clear of other robots'
            bookings (always True for a real ST solution; may be False for a
            fallback that could not dodge a conflict).
        expanded: Number of ST states expanded (0 for a trivial/fallback path).
    """

    path: List[Point]
    cells: List[Cell]
    cell_times: List[int]
    fell_back: bool
    conflict_free: bool
    expanded: int


def _reconstruct_st(
    came: Dict[Tuple[int, int, int], Tuple[int, int, int]],
    goal_state: Tuple[int, int, int],
    t0: int,
) -> Tuple[List[Cell], List[int]]:
    """Rebuild (cells, cell_times) from the came-from map of the ST search."""
    chain = [goal_state]
    node = goal_state
    while node in came:
        node = came[node]
        chain.append(node)
    chain.reverse()
    cells = [(s[0], s[1]) for s in chain]
    times = [s[2] - t0 for s in chain]
    return cells, times


def _dedupe_world(og: OccupancyGrid, cells: List[Cell]) -> List[Point]:
    """World centres of ``cells`` with consecutive duplicates (waits) collapsed."""
    out: List[Point] = []
    prev: Optional[Cell] = None
    for c in cells:
        if c != prev:
            out.append(og.cell_to_world(c))
            prev = c
    return out


def plan_path_st(
    og: OccupancyGrid,
    table: Optional[ReservationTable],
    start_xy: Point,
    goal_xy: Point,
    t0: int,
    speed: float,
    *,
    ignore_id: Optional[str] = None,
    snap_radius_m: float = 0.30,
    node_budget: int = 150000,
    horizon_steps: int = 6000,
) -> STResult:
    """Plan a space-time route that dodges other robots' reservations.

    Runs A* over ``(iy, ix, t)`` states with the same 8-connected, no-corner-cut
    moves as :func:`plan_path`, plus a *wait* action (stay in the cell for one
    cell-duration). A move/wait is only taken into a ``(cell, t')`` that
    ``table`` does not reserve for another robot at ``t'``. Costs are in control
    steps (``card``/``diag`` per edge, one cell-duration per wait) and the octile
    step-count heuristic keeps the search admissible and deterministic (heap ties
    break on ``(h, iy, ix, t)``).

    If the search exceeds ``node_budget`` expansions, it returns a plain
    :func:`plan_path` result with ``fell_back=True`` so the caller can still book
    *something* and log the fallback.

    Args:
        og: Occupancy grid (already inflated for wall clearance).
        table: Shared reservation table, or None (then this is plain A* over time).
        start_xy: World start (x, y) in meters.
        goal_xy: World goal (x, y) in meters.
        t0: Absolute control step at which the robot starts this route.
        speed: Conservative model walking speed (m/s).
        ignore_id: Robot id whose own bookings are ignored (its stale plan).
        snap_radius_m: Snap radius for occupied endpoints (matches plan_path).
        node_budget: Max ST states to expand before falling back.
        horizon_steps: Cap on how far into the future (from t0) the search looks.

    Returns:
        An :class:`STResult`.

    Raises:
        PathNotFoundError: If an endpoint cannot snap to free space.
    """
    free = _free_mask(og)
    ny, nx = og.grid.shape
    start = _nearest_free_cell(og, free, start_xy, snap_radius_m)
    goal = _nearest_free_cell(og, free, goal_xy, snap_radius_m)
    if start is None or goal is None:
        raise PathNotFoundError(
            f"cannot snap endpoint to free space within {snap_radius_m} m: "
            f"start_xy={start_xy} -> {start}, goal_xy={goal_xy} -> {goal}")

    card, diag = steps_per_cell(og.resolution, speed)
    wait_cost = card

    def heur(iy: int, ix: int) -> int:
        dy = abs(iy - goal[0])
        dx = abs(ix - goal[1])
        lo, hi = (dy, dx) if dy < dx else (dx, dy)
        return (hi - lo) * card + lo * diag

    if start == goal:
        return STResult([og.cell_to_world(start)], [start], [0], False, True, 0)

    def reserved(cell: Cell, t: int) -> bool:
        return table is not None and table.is_reserved(cell, t, ignore_id)

    start_state = (start[0], start[1], int(t0))
    came: Dict[Tuple[int, int, int], Tuple[int, int, int]] = {}
    g_score: Dict[Tuple[int, int, int], int] = {start_state: 0}
    closed: Set[Tuple[int, int, int]] = set()
    h0 = heur(start[0], start[1])
    open_heap: List[Tuple[int, int, int, int, int]] = [
        (h0, h0, start[0], start[1], int(t0))]
    expanded = 0
    goal_state: Optional[Tuple[int, int, int]] = None

    while open_heap:
        _, _, iy, ix, t = heapq.heappop(open_heap)
        state = (iy, ix, t)
        if state in closed:
            continue
        closed.add(state)
        if (iy, ix) == goal:
            goal_state = state
            break
        expanded += 1
        if expanded > node_budget:
            break
        g = g_score[state]
        for diy, dix, _cost in _MOVES:
            niy, nix = iy + diy, ix + dix
            if niy < 0 or niy >= ny or nix < 0 or nix >= nx:
                continue
            if not free[niy, nix]:
                continue
            if diy != 0 and dix != 0:
                if not free[iy + diy, ix] or not free[iy, ix + dix]:
                    continue
            step = diag if (diy != 0 and dix != 0) else card
            nt = t + step
            if nt - t0 > horizon_steps:
                continue
            if reserved((niy, nix), nt):
                continue
            ns = (niy, nix, nt)
            if ns in closed:
                continue
            tentative = g + step
            if tentative < g_score.get(ns, 1 << 62):
                came[ns] = state
                g_score[ns] = tentative
                hh = heur(niy, nix)
                heapq.heappush(open_heap, (tentative + hh, hh, niy, nix, nt))
        # Wait action: hold the cell for one cell-duration (only if it stays clear).
        wt = t + wait_cost
        if wt - t0 <= horizon_steps and not reserved((iy, ix), wt):
            ns = (iy, ix, wt)
            if ns not in closed:
                tentative = g + wait_cost
                if tentative < g_score.get(ns, 1 << 62):
                    came[ns] = state
                    g_score[ns] = tentative
                    hh = heur(iy, ix)
                    heapq.heappush(open_heap, (tentative + hh, hh, iy, ix, wt))

    if goal_state is not None:
        cells, times = _reconstruct_st(came, goal_state, t0)
        world = _dedupe_world(og, cells)
        return STResult(world, cells, times, False, True, expanded)

    # ---- Fallback: plain A*, still booked so others route around it. ----
    world = plan_path(og, start_xy, goal_xy, snap_radius_m=snap_radius_m)
    cells = [_raw_cell(og, p) for p in world]
    times = cell_times_for_path(cells, og.resolution, speed)
    conflict = (table is not None
                and table.first_conflict(cells, t0, speed, ignore_id) is not None)
    return STResult(world, cells, times, True, not conflict, expanded)
