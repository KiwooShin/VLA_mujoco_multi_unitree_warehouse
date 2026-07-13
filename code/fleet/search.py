"""search.py — Region partition + serpentine patrol for delegated search.

When the task owner cannot locate an object and no peer can see it, it delegates
a divide-and-conquer search: each idle peer is handed a named region of the hall
(:data:`SEARCH_REGIONS`) and patrols it while polling the visibility oracle. This
module owns the *where to walk* half of that behaviour:

* :func:`region_bounds` / :func:`region_centroid` — split the hall into
  ``north``/``middle``/``south`` bands by thirds of ``hall_y`` (matching the
  region strings the comms protocol passes to searchers).
* :func:`patrol_waypoints` — a boustrophedon (lawnmower) sweep of a region,
  sampled at ~1.5 m spacing from the FREE cells of the inflated planner grid so
  every waypoint is planner-reachable and clear of walls/shelves.
* :class:`SearchController` — drives one :class:`~code.fleet.robot_unit.RobotUnit`
  along its region's patrol, advancing to the next waypoint each time the robot
  arrives and looping the sweep until the search is aborted.

Everything here is deterministic given the layout (no RNG), so a seeded mission
replays identically.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

from code.apps.warehouse_demo.planning import build_inflated_grid
from code.fleet.robot_unit import RobotUnit

XY = Tuple[float, float]

# Region labels handed to searchers (partition the hall along y into thirds).
SEARCH_REGIONS: Tuple[str, ...] = ("north", "middle", "south")

# Patrol geometry. Waypoints are spaced coarser than the ~6 m visibility range so
# a walking searcher sweeps the region quickly (the head-cam cone does the fine
# coverage between vantage points); a nearest-neighbour tour from the robot's
# entry point avoids the long "walk to the far corner first" detour a fixed
# west->east lawnmower incurs.
_SPACING_M: float = 2.5      # nominal vantage-point spacing
_EDGE_MARGIN_M: float = 1.0  # keep waypoints this far inside the hall perimeter


def region_bounds(region: str, hall_y: float) -> Tuple[float, float]:
    """Return the ``(y_lo, y_hi)`` band of a named region.

    The hall (y in ``[-hall_y/2, hall_y/2]``) is split into equal thirds:
    ``south`` (most negative y), ``middle``, ``north`` (most positive y).

    Args:
        region: One of :data:`SEARCH_REGIONS`.
        hall_y: Full hall extent along y (m).

    Returns:
        The half-open band ``(y_lo, y_hi)`` in metres.

    Raises:
        ValueError: If ``region`` is not a known region.
    """
    half = hall_y / 2.0
    third = hall_y / 3.0
    if region == "south":
        return (-half, -half + third)
    if region == "middle":
        return (-half + third, -half + 2.0 * third)
    if region == "north":
        return (half - third, half)
    raise ValueError(f"unknown region {region!r}; expected one of {SEARCH_REGIONS}")


def region_centroid(region: str, hall_x: float, hall_y: float) -> XY:
    """Return the geometric centre of a region band (x=0, mid-band y)."""
    y_lo, y_hi = region_bounds(region, hall_y)
    return (0.0, (y_lo + y_hi) / 2.0)


def _axis_samples(lo: float, hi: float, spacing: float) -> List[float]:
    """Evenly spaced samples spanning ``[lo, hi]`` inclusive (>= 2 points)."""
    if hi <= lo:
        return [(lo + hi) / 2.0]
    n = max(1, int(round((hi - lo) / spacing)))
    step = (hi - lo) / n
    return [lo + i * step for i in range(n + 1)]


def _nearest_neighbour_order(points: List[XY], start: XY) -> List[XY]:
    """Greedily order ``points`` by nearest-neighbour from ``start`` (deterministic)."""
    remaining = list(points)
    ordered: List[XY] = []
    cur = start
    while remaining:
        j = min(range(len(remaining)),
                key=lambda i: ((remaining[i][0] - cur[0]) ** 2
                               + (remaining[i][1] - cur[1]) ** 2, remaining[i]))
        cur = remaining.pop(j)
        ordered.append(cur)
    return ordered


def patrol_waypoints(
    scene_cfg: dict, region: str, *, start_xy: Optional[XY] = None,
    resolution: float = 0.10, inflate_radius: float = 0.40,
    spacing: float = _SPACING_M, margin: float = _EDGE_MARGIN_M,
) -> List[XY]:
    """Return a region patrol over free planner cells (vantage points).

    A coarse grid of candidate cells (spacing ~:data:`_SPACING_M`) is sampled
    across the region; only cells FREE in the inflated occupancy grid are kept
    (so each is reachable and clear of walls/shelves at the deployed inflation
    radius). When ``start_xy`` is given the vantage points are ordered by a
    nearest-neighbour tour from it (short, entry-relative sweep); otherwise a
    boustrophedon row order is used. Deterministic for a fixed layout + start.

    Args:
        scene_cfg: Warehouse scene_cfg (``walls``/``objects``/``hall_x/y``).
        region: Region to cover (:data:`SEARCH_REGIONS`).
        start_xy: Robot entry point for nearest-neighbour ordering (optional).
        resolution: Occupancy grid cell size (m).
        inflate_radius: Robot-clearance dilation (m).
        spacing: Vantage-point spacing (m).
        margin: Keep-out from the hall perimeter (m).

    Returns:
        The ordered patrol waypoints (world x, y); possibly empty if the whole
        region is occupied.
    """
    grid = build_inflated_grid(scene_cfg, resolution, inflate_radius)
    hall_x = float(scene_cfg["hall_x"])
    hall_y = float(scene_cfg["hall_y"])
    y_lo, y_hi = region_bounds(region, hall_y)
    x_lo, x_hi = -hall_x / 2.0 + margin, hall_x / 2.0 - margin
    rows = _axis_samples(y_lo + margin / 2.0, y_hi - margin / 2.0, spacing)
    cols = _axis_samples(x_lo, x_hi, spacing)

    waypoints: List[XY] = []
    for r, y in enumerate(rows):
        xs = cols if r % 2 == 0 else list(reversed(cols))
        for x in xs:
            if grid.is_free((x, y)):
                waypoints.append((round(x, 3), round(y, 3)))
    if start_xy is not None:
        waypoints = _nearest_neighbour_order(waypoints, start_xy)
    return waypoints


def free_centroid(scene_cfg: dict, region: str, **kw) -> XY:
    """Centroid of a region's free patrol cells (falls back to the geometric centre)."""
    pts = patrol_waypoints(scene_cfg, region, **kw)
    if not pts:
        return region_centroid(region, float(scene_cfg["hall_x"]),
                               float(scene_cfg["hall_y"]))
    return (sum(p[0] for p in pts) / len(pts), sum(p[1] for p in pts) / len(pts))


class SearchController:
    """Drives one robot along a region patrol, looping until aborted.

    The controller owns no perception: the coordination protocol polls the
    visibility oracle each step and calls :meth:`stop` (via ``abort_search``)
    when the object is found. Until then :meth:`tick` keeps the robot walking the
    lawnmower path, wrapping back to the first waypoint after the last so a slow
    object sighting is still eventually covered.
    """

    def __init__(self, unit: RobotUnit) -> None:
        """Bind the controller to a robot unit (no motion yet)."""
        self._unit = unit
        self._waypoints: List[XY] = []
        self._idx = 0
        self.active = False
        self.region: Optional[str] = None

    def start(self, scene_cfg: dict, region: str, **kw) -> bool:
        """Plan the region patrol and start walking its first waypoint.

        Args:
            scene_cfg: Warehouse scene_cfg.
            region: Region to patrol.
            **kw: Forwarded to :func:`patrol_waypoints`.

        Returns:
            True if at least one reachable waypoint was assigned.
        """
        kw.setdefault("start_xy", self._unit.xy)
        self._waypoints = patrol_waypoints(scene_cfg, region, **kw)
        self._idx = 0
        self.region = region
        self.active = bool(self._waypoints)
        if self.active:
            self._assign_reachable()
        return self.active

    def _assign_reachable(self) -> None:
        """Assign the next waypoint that the unit can actually plan to."""
        n = len(self._waypoints)
        for _ in range(n):
            if self._unit.assign_goal(self._waypoints[self._idx]):
                return
            self._idx = (self._idx + 1) % n  # skip an unplannable waypoint
        self.active = False  # nothing reachable

    def tick(self) -> None:
        """Advance the patrol one control step's worth of bookkeeping.

        Call once per control step *after* the fleet has stepped. When the robot
        has reached (or stalled at) its current waypoint, advance to the next
        (wrapping around) and re-plan.
        """
        if not self.active or not self._waypoints:
            return
        if self._unit.done or not self._unit.active:
            self._idx = (self._idx + 1) % len(self._waypoints)
            self._assign_reachable()

    def stop(self) -> None:
        """Abort the search and hold the robot in place."""
        self.active = False
        self._unit.halt()

    @property
    def waypoints(self) -> List[XY]:
        """The planned patrol waypoints (read-only copy)."""
        return list(self._waypoints)
