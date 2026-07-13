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

from typing import List, Optional, Sequence, Tuple

from code.apps.warehouse_demo.planning import build_inflated_grid
from code.fleet.robot_unit import RobotUnit
from code.warehouse.layout import Room, WarehouseLayout, room_of

XY = Tuple[float, float]

# Region labels handed to searchers on the single-hall (hero) layout: partition
# the hall along y into thirds. On a multi-room layout the room NAMES are the
# regions instead (see :func:`search_regions_for_layout`).
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


def _room_by_name(region: str, rooms: Sequence[Room]) -> Optional[Room]:
    """Return the room whose name equals ``region``, or ``None``."""
    for r in rooms:
        if r.name == region:
            return r
    return None


def region_box(region: str, hall_x: float, hall_y: float,
               rooms: Sequence[Room] = ()) -> Tuple[float, float, float, float]:
    """Return the ``(x_lo, x_hi, y_lo, y_hi)`` bounding box of a search region.

    On a multi-room layout (``rooms`` given and ``region`` names one) the box is
    that room's footprint; otherwise the region is a hero-layout y-third band
    spanning the full hall width.

    Args:
        region: Region label — a room name (rooms mode) or a hero third.
        hall_x: Full hall extent along x (m).
        hall_y: Full hall extent along y (m).
        rooms: The layout's rooms (empty for the single-hall hero layout).

    Returns:
        The region box ``(x_lo, x_hi, y_lo, y_hi)`` in metres.
    """
    room = _room_by_name(region, rooms)
    if room is not None:
        return (room.cx - room.half_x, room.cx + room.half_x,
                room.cy - room.half_y, room.cy + room.half_y)
    y_lo, y_hi = region_bounds(region, hall_y)
    return (-hall_x / 2.0, hall_x / 2.0, y_lo, y_hi)


def region_centroid(region: str, hall_x: float, hall_y: float,
                    rooms: Sequence[Room] = ()) -> XY:
    """Return the geometric centre of a region (room box or hero y-third band)."""
    x_lo, x_hi, y_lo, y_hi = region_box(region, hall_x, hall_y, rooms)
    return ((x_lo + x_hi) / 2.0, (y_lo + y_hi) / 2.0)


def search_regions_for_layout(layout: WarehouseLayout) -> Tuple[str, ...]:
    """Return the search-partition region labels for a layout (F6 unification).

    On the multi-room layout the regions are the room NAMES — minus the room(s)
    the robots spawn in, which the fleet already covers from its home bays via
    the owner's own view and peer visibility queries (leaving one searchable
    room per peer). On the single-hall hero layout the regions are the fixed
    north/middle/south thirds.

    Args:
        layout: The active warehouse layout.

    Returns:
        The ordered region labels to hand to searchers.
    """
    if not layout.rooms:
        return SEARCH_REGIONS
    spawn_rooms = {room_of(layout, (x, y))
                   for (x, y, _yaw) in layout.spawn_poses.values()}
    return tuple(r.name for r in layout.rooms if r.name not in spawn_rooms)


def region_name_for_xy(layout: WarehouseLayout, xy: XY) -> str:
    """Return the F3 room/region name a point lies in.

    On a rooms layout this is ``room_of`` (the true named room); on the hero
    single hall it falls back to the "north/middle/south area" third the point
    occupies (docs/final_demo_spec.md F3).

    Args:
        layout: The active warehouse layout.
        xy: World point (x, y) in metres.

    Returns:
        The region/room name (usable verbatim in the F3 report sentence).
    """
    if layout.rooms:
        return room_of(layout, xy)
    y = float(xy[1])
    third = layout.hall_y / 3.0
    half = layout.hall_y / 2.0
    if y >= half - third:
        return "north area"
    if y <= -half + third:
        return "south area"
    return "middle area"


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
    rooms: Sequence[Room] = (),
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
        rooms: The layout's rooms (rooms mode); when ``region`` names one, the
            patrol covers that room's footprint instead of a hero y-third band.

    Returns:
        The ordered patrol waypoints (world x, y); possibly empty if the whole
        region is occupied.
    """
    grid = build_inflated_grid(scene_cfg, resolution, inflate_radius)
    hall_x = float(scene_cfg["hall_x"])
    hall_y = float(scene_cfg["hall_y"])
    bx_lo, bx_hi, y_lo, y_hi = region_box(region, hall_x, hall_y, rooms)
    x_lo, x_hi = bx_lo + margin, bx_hi - margin
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


def free_centroid(scene_cfg: dict, region: str, *,
                  rooms: Sequence[Room] = (), **kw) -> XY:
    """Centroid of a region's free patrol cells (falls back to the geometric centre)."""
    pts = patrol_waypoints(scene_cfg, region, rooms=rooms, **kw)
    if not pts:
        return region_centroid(region, float(scene_cfg["hall_x"]),
                               float(scene_cfg["hall_y"]), rooms)
    return (sum(p[0] for p in pts) / len(pts), sum(p[1] for p in pts) / len(pts))


class SearchController:
    """Drives one robot along a region patrol, looping until aborted.

    The controller owns no perception: the coordination protocol polls the
    visibility oracle each step and calls :meth:`stop` (via ``abort_search``)
    when the object is found. Until then :meth:`tick` keeps the robot walking the
    lawnmower path, wrapping back to the first waypoint after the last so a slow
    object sighting is still eventually covered.
    """

    def __init__(self, unit: RobotUnit, *, rooms: Sequence[Room] = ()) -> None:
        """Bind the controller to a robot unit (no motion yet).

        Args:
            unit: The robot to drive along the patrol.
            rooms: The layout's rooms (rooms mode) so a room-named region patrols
                that room's footprint; empty for the hero thirds.
        """
        self._unit = unit
        self._rooms: Tuple[Room, ...] = tuple(rooms)
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
        kw.setdefault("rooms", self._rooms)
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
