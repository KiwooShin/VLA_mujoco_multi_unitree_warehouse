"""layout.py — Pure-Python warehouse layout spec (single source of truth).

This module holds the geometric description of the warehouse hall: perimeter
walls, double-sided shelf rows, partitions, floor zones, robot spawn bays and
object spots. It has **no** MuJoCo dependency — ``code.warehouse.arena`` turns a
:class:`WarehouseLayout` into MJCF and ``code.warehouse.occupancy`` rasterizes
the same wall list into a planner occupancy grid, so the simulated geometry and
the planning world can never skew (docs/multi_plan.md sec 2).

Frame convention: hall-centered world coordinates (x right, y up in a top-down
view). The hero hall is 16 m x 12 m, so x in [-8, 8] and y in [-6, 6].

Public API
----------
WallSpec, Zone, WarehouseLayout — frozen dataclasses.
hero_layout() -> WarehouseLayout — the fixed, video-tuned demo layout.
sample_layout(rng) -> WarehouseLayout — seeded validity-preserving variation.
"""

from __future__ import annotations

import dataclasses
import math
from typing import Dict, List, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Robot callsigns and their demo accent colours (helmet/pad tint per robot).
# ---------------------------------------------------------------------------
CALLSIGNS: Tuple[str, ...] = ("Alpha", "Bravo", "Charlie", "Delta")
# Home-bay floor pads: each robot's identity colour, slightly desaturated (moved
# toward mid-gray) so the pads read as calm industrial floor markings rather than
# neon on video. Robot torso accents (code.fleet.viz.ACCENT_RGBA) stay vivid so
# the robots themselves remain easy to track.
_BAY_RGBA: Dict[str, Tuple[float, float, float, float]] = {
    "Alpha": (0.78, 0.30, 0.30, 0.50),    # red
    "Bravo": (0.34, 0.44, 0.78, 0.50),    # blue
    "Charlie": (0.80, 0.72, 0.34, 0.50),  # yellow
    "Delta": (0.56, 0.38, 0.70, 0.50),    # purple
}

_PERIM_RGBA: Tuple[float, float, float, float] = (0.74, 0.75, 0.78, 1.0)
_SHELF_RGBA: Tuple[float, float, float, float] = (0.55, 0.40, 0.24, 1.0)
_PART_RGBA: Tuple[float, float, float, float] = (0.62, 0.64, 0.68, 1.0)
# Delivery pad: a clear, readable green — clearly the destination, not neon.
_DELIVERY_RGBA: Tuple[float, float, float, float] = (0.30, 0.60, 0.40, 0.52)

# Validity thresholds (metres).
_MIN_AISLE_M: float = 2.2
_MIN_CLEAR_M: float = 0.5
_MIN_SPOT_SPACING_M: float = 0.8
_MIN_SPAWN_SPACING_M: float = 1.5
_WALL_T: float = 0.1  # perimeter/partition half-thickness


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------
@dataclasses.dataclass(frozen=True)
class WallSpec:
    """A single axis-or-yawed rectangular wall/shelf/partition block.

    Attributes:
        cx: World x of the block centre (m).
        cy: World y of the block centre (m).
        half_x: Half-extent along the block's local x axis (m).
        half_y: Half-extent along the block's local y axis (m).
        yaw: Rotation about +z (rad); 0 means axis-aligned.
        height: Full block height (m); 2.5 perimeter, ~1.8 shelf, ~2.0 partition.
        rgba: (r, g, b, a) colour, each channel in [0, 1].
        name: Unique geom name.
    """

    cx: float
    cy: float
    half_x: float
    half_y: float
    yaw: float = 0.0
    height: float = 2.5
    rgba: Tuple[float, float, float, float] = _PERIM_RGBA
    name: str = ""


@dataclasses.dataclass(frozen=True)
class Zone:
    """A visual-only, non-colliding floor pad (delivery pad / home bay).

    Attributes:
        name: Unique zone name.
        cx: World x of the pad centre (m).
        cy: World y of the pad centre (m).
        half_x: Half-extent along x (m).
        half_y: Half-extent along y (m).
        rgba: (r, g, b, a) colour, each channel in [0, 1] (alpha < 1).
    """

    name: str
    cx: float
    cy: float
    half_x: float
    half_y: float
    rgba: Tuple[float, float, float, float] = (0.5, 0.5, 0.5, 0.5)


@dataclasses.dataclass(frozen=True)
class Room:
    """An axis-aligned room bounding box tiling the warehouse interior.

    Rooms are the single source of truth for named regions: the search
    partition, the F3 "currently in {room}" position reports and the
    COMMAND_SEARCH region strings all consume ``layout.rooms``. The boxes tile
    the interior with shared edges (no gaps, no interior overlap); open
    doorways cut into the dividing walls connect adjacent rooms.

    Attributes:
        name: Human-readable region name, usable verbatim in speech
            (e.g. "loading room", "storage A", "back room").
        cx: World x of the room-box centre (m).
        cy: World y of the room-box centre (m).
        half_x: Half-extent along x (m).
        half_y: Half-extent along y (m).
    """

    name: str
    cx: float
    cy: float
    half_x: float
    half_y: float


@dataclasses.dataclass(frozen=True)
class WarehouseLayout:
    """A complete warehouse layout (walls, zones, spawns, object spots).

    Attributes:
        hall_x: Full hall extent along x (m); hall spans [-hall_x/2, hall_x/2].
        hall_y: Full hall extent along y (m); hall spans [-hall_y/2, hall_y/2].
        walls: Perimeter + shelf + partition blocks.
        zones: Visual floor pads (delivery pad + home bays).
        spawn_poses: callsign -> (x, y, yaw) robot start pose.
        object_spots: (x, y) positions where objects may be placed.
        rooms: Named room bounding boxes tiling the interior (empty for the
            single-hall hero/sample layouts; populated by ``rooms_layout``).
        name: Short layout identifier (e.g. "hero", "sample_7", "rooms").
    """

    hall_x: float = 16.0
    hall_y: float = 12.0
    walls: List[WallSpec] = dataclasses.field(default_factory=list)
    zones: List[Zone] = dataclasses.field(default_factory=list)
    spawn_poses: Dict[str, Tuple[float, float, float]] = dataclasses.field(
        default_factory=dict
    )
    object_spots: List[Tuple[float, float]] = dataclasses.field(default_factory=list)
    rooms: Tuple[Room, ...] = ()
    name: str = "hero"


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------
def _point_rect_distance(px: float, py: float, wall: WallSpec) -> float:
    """Return the shortest distance from a point to a (possibly yawed) rect.

    Args:
        px: Point x (m).
        py: Point y (m).
        wall: Rectangle whose footprint is tested (height ignored).

    Returns:
        Distance in metres; 0.0 if the point lies inside the rectangle.
    """
    dx = px - wall.cx
    dy = py - wall.cy
    c, s = math.cos(wall.yaw), math.sin(wall.yaw)
    lx = dx * c + dy * s
    ly = -dx * s + dy * c
    ox = max(abs(lx) - wall.half_x, 0.0)
    oy = max(abs(ly) - wall.half_y, 0.0)
    return math.hypot(ox, oy)


def _obb_axes(cx: float, cy: float, hx: float, hy: float,
              yaw: float) -> Tuple[np.ndarray, List[np.ndarray]]:
    """Return the centre, half-axis vectors of an oriented box footprint."""
    c, s = math.cos(yaw), math.sin(yaw)
    ax = np.array([c, s]) * hx
    ay = np.array([-s, c]) * hy
    return np.array([cx, cy]), [ax, ay]


def _obb_overlap(a: WallSpec, b_cx: float, b_cy: float, b_hx: float,
                 b_hy: float, b_yaw: float = 0.0) -> bool:
    """Separating-axis test between wall ``a`` and a second oriented box."""
    ca, axes_a = _obb_axes(a.cx, a.cy, a.half_x, a.half_y, a.yaw)
    cb, axes_b = _obb_axes(b_cx, b_cy, b_hx, b_hy, b_yaw)
    t = cb - ca
    for axis in (*axes_a, *axes_b):
        n = axis / (np.linalg.norm(axis) + 1e-12)
        ra = sum(abs(float(np.dot(n, e))) for e in axes_a)
        rb = sum(abs(float(np.dot(n, e))) for e in axes_b)
        if abs(float(np.dot(n, t))) > ra + rb + 1e-9:
            return False
    return True


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
def _shelf_rows(layout: WarehouseLayout) -> List[List[WallSpec]]:
    """Group shelf blocks by their y-centre (one list per lengthwise row)."""
    shelves = [w for w in layout.walls if w.name.startswith("shelf_")]
    rows: Dict[float, List[WallSpec]] = {}
    for w in shelves:
        rows.setdefault(round(w.cy, 3), []).append(w)
    return [rows[k] for k in sorted(rows)]


def validate_layout(layout: WarehouseLayout) -> None:
    """Raise ValueError if any layout invariant is violated.

    Checks (docs/multi_plan.md sec 2): lengthwise aisles >= 2.2 m between shelf
    faces (and shelf-to-perimeter), every spawn/object_spot >= 0.5 m from any
    wall, spots/spawns non-overlapping, spots outside the delivery pad, and
    zones not overlapping walls.

    Args:
        layout: Layout to validate.

    Raises:
        ValueError: On the first invariant that fails, with a diagnostic message.
    """
    half_x, half_y = layout.hall_x / 2.0, layout.hall_y / 2.0

    # --- Aisle widths between the two lengthwise shelf rows and perimeter. ---
    rows = _shelf_rows(layout)
    if len(rows) != 2:
        raise ValueError(f"expected 2 shelf rows, found {len(rows)}")
    row_faces = []
    for row in rows:
        cy = row[0].cy
        hy = row[0].half_y
        row_faces.append((cy - hy, cy + hy))  # (south face, north face)
    row_faces.sort()
    south_row, north_row = row_faces
    inner_north = half_y - _WALL_T
    inner_south = -half_y + _WALL_T
    aisles = {
        "north": inner_north - north_row[1],
        "middle": north_row[0] - south_row[1],
        "south": south_row[0] - inner_south,
    }
    for name, width in aisles.items():
        if width < _MIN_AISLE_M - 1e-6:
            raise ValueError(
                f"{name} aisle width {width:.3f} m < {_MIN_AISLE_M} m minimum"
            )

    # --- Spawns and object_spots clear of every wall. ---
    for cs, (sx, sy, _yaw) in layout.spawn_poses.items():
        if not (-half_x < sx < half_x and -half_y < sy < half_y):
            raise ValueError(f"spawn {cs} at ({sx},{sy}) outside hall")
        for w in layout.walls:
            d = _point_rect_distance(sx, sy, w)
            if d < _MIN_CLEAR_M - 1e-6:
                raise ValueError(
                    f"spawn {cs} only {d:.3f} m from wall {w.name} "
                    f"(< {_MIN_CLEAR_M} m)"
                )
    for i, (ox, oy) in enumerate(layout.object_spots):
        for w in layout.walls:
            d = _point_rect_distance(ox, oy, w)
            if d < _MIN_CLEAR_M - 1e-6:
                raise ValueError(
                    f"object_spot {i} at ({ox:.2f},{oy:.2f}) only {d:.3f} m "
                    f"from wall {w.name} (< {_MIN_CLEAR_M} m)"
                )

    # --- Pairwise spacing. ---
    for i in range(len(layout.object_spots)):
        for j in range(i + 1, len(layout.object_spots)):
            ax, ay = layout.object_spots[i]
            bx, by = layout.object_spots[j]
            if math.hypot(ax - bx, ay - by) < _MIN_SPOT_SPACING_M - 1e-6:
                raise ValueError(f"object_spots {i},{j} closer than "
                                 f"{_MIN_SPOT_SPACING_M} m")
    spawns = list(layout.spawn_poses.values())
    for i in range(len(spawns)):
        for j in range(i + 1, len(spawns)):
            if math.hypot(spawns[i][0] - spawns[j][0],
                          spawns[i][1] - spawns[j][1]) < _MIN_SPAWN_SPACING_M:
                raise ValueError(f"spawns {i},{j} closer than "
                                 f"{_MIN_SPAWN_SPACING_M} m")

    # --- Object spots outside the delivery pad (keep the destination clear). ---
    delivery = next((z for z in layout.zones if z.name == "delivery"), None)
    if delivery is not None:
        for i, (ox, oy) in enumerate(layout.object_spots):
            if (abs(ox - delivery.cx) <= delivery.half_x
                    and abs(oy - delivery.cy) <= delivery.half_y):
                raise ValueError(f"object_spot {i} lies inside delivery pad")

    # --- Zones do not overlap solid walls. ---
    for z in layout.zones:
        for w in layout.walls:
            if _obb_overlap(w, z.cx, z.cy, z.half_x, z.half_y):
                raise ValueError(f"zone {z.name} overlaps wall {w.name}")


# ---------------------------------------------------------------------------
# Layout builders
# ---------------------------------------------------------------------------
def _perimeter_walls(half_x: float, half_y: float) -> List[WallSpec]:
    """Build the four perimeter walls of a hall-centered rectangle."""
    t = _WALL_T
    return [
        WallSpec(half_x, 0.0, t, half_y, name="wall_E", rgba=_PERIM_RGBA),
        WallSpec(-half_x, 0.0, t, half_y, name="wall_W", rgba=_PERIM_RGBA),
        WallSpec(0.0, half_y, half_x, t, name="wall_N", rgba=_PERIM_RGBA),
        WallSpec(0.0, -half_y, half_x, t, name="wall_S", rgba=_PERIM_RGBA),
    ]


def _shelf_walls(gap: float, row_cy: float, block_len: float) -> List[WallSpec]:
    """Build the four shelf blocks (two lengthwise rows split by a mid gap).

    Args:
        gap: Full width of the mid-row crossover gap (m), centred at x=0.
        row_cy: Absolute y-offset of each row from the hall centre (m).
        block_len: Full length of each shelf block along x (m).

    Returns:
        Four WallSpec blocks (row A north, row B south; west/east each).
    """
    hx = block_len / 2.0
    hy = 0.35  # 0.7 m deep
    cx = gap / 2.0 + hx
    height = 1.8
    blocks = []
    for row, sign in (("A", 1.0), ("B", -1.0)):
        cy = sign * row_cy
        blocks.append(WallSpec(-cx, cy, hx, hy, height=height,
                               rgba=_SHELF_RGBA, name=f"shelf_{row}_w"))
        blocks.append(WallSpec(cx, cy, hx, hy, height=height,
                               rgba=_SHELF_RGBA, name=f"shelf_{row}_e"))
    return blocks


def _partition_walls(alcove_x: float, sw_cy: float) -> List[WallSpec]:
    """Build the NE L-partition (2 walls) and the SW freestanding wall.

    Args:
        alcove_x: x of the vertical L-arm (left edge of the NE alcove).
        sw_cy: y-centre of the SW freestanding wall.

    Returns:
        Three WallSpec blocks: NE vertical arm, NE horizontal arm, SW wall.
    """
    t = _WALL_T
    return [
        # NE alcove: vertical arm up to the north wall.
        WallSpec(alcove_x, 4.75, t, 1.25, height=2.0,
                 rgba=_PART_RGBA, name="part_ne_v"),
        # NE alcove: horizontal arm (leaves an opening on its east end).
        WallSpec(alcove_x + 1.0, 3.5, 1.0, t, height=2.0,
                 rgba=_PART_RGBA, name="part_ne_h"),
        # SW freestanding short wall.
        WallSpec(-4.5, sw_cy, t, 0.75, height=2.0,
                 rgba=_PART_RGBA, name="part_sw"),
    ]


def _zones_and_spawns(
    delivery_xy: Tuple[float, float]
) -> Tuple[List[Zone], Dict[str, Tuple[float, float, float]]]:
    """Build the delivery pad, the four home-bay zones and spawn poses."""
    zones: List[Zone] = [
        Zone("delivery", delivery_xy[0], delivery_xy[1], 1.0, 1.0, _DELIVERY_RGBA),
    ]
    spawns: Dict[str, Tuple[float, float, float]] = {}
    bay_x = {"Alpha": -5.0, "Bravo": -2.0, "Charlie": 2.0, "Delta": 5.0}
    for cs in CALLSIGNS:
        bx = bay_x[cs]
        zones.append(Zone(f"bay_{cs}", bx, -5.0, 0.8, 0.7, _BAY_RGBA[cs]))
        spawns[cs] = (bx, -5.0, math.pi / 2.0)  # facing +y into the hall
    return zones, spawns


def _hero_object_spots() -> List[Tuple[float, float]]:
    """Return the 8 fixed hero object spots (aisles + NE alcove)."""
    return [
        (-1.5, 0.0),   # middle aisle, occluded by shelf B from south bays
        (1.5, 0.0),    # middle aisle, occluded by shelf B
        (0.0, 0.7),    # central crossover
        (-1.5, 3.3),   # north aisle, occluded by both rows
        (1.5, 3.3),    # north aisle
        (6.5, 4.7),    # NE alcove (occluded by L-partition)
        (-5.0, 0.5),   # west open lane
        (3.0, -0.5),   # east-central near delivery lane
    ]


def hero_layout() -> WarehouseLayout:
    """Return the fixed, video-tuned hero warehouse layout.

    Geometry (docs/multi_plan.md sec 2): 16x12 m hall; two double-sided shelf
    rows (each 4.5 m x 0.7 m x 1.8 m) split by a 1.4 m mid-row gap into four
    blocks -> three lengthwise aisles (~3.55/3.3/3.55 m) plus a central
    crossover; an L-partition forming an occluded NE alcove; a short SW wall; a
    2x2 m green delivery pad on the east side; four colour-accented home bays
    along the south wall with spawns facing into the hall; eight object spots.

    Returns:
        A validated :class:`WarehouseLayout`.

    Raises:
        ValueError: If the assembled layout violates an invariant (a bug).
    """
    half_x, half_y = 8.0, 6.0
    walls = _perimeter_walls(half_x, half_y)
    walls += _shelf_walls(gap=1.4, row_cy=2.0, block_len=1.55)
    walls += _partition_walls(alcove_x=5.0, sw_cy=-3.25)
    zones, spawns = _zones_and_spawns(delivery_xy=(5.8, -1.0))
    layout = WarehouseLayout(
        hall_x=16.0, hall_y=12.0, walls=walls, zones=zones,
        spawn_poses=spawns, object_spots=_hero_object_spots(), name="hero",
    )
    validate_layout(layout)
    return layout


def sample_layout(rng: np.random.Generator, *,
                  enforce_reachable: bool = True) -> WarehouseLayout:
    """Return a seeded, validity-preserving variation of the hero layout.

    Jitters the mid-row gap, the shelf-row separation, the NE alcove x, the SW
    wall y, the delivery pad position and each object spot within ranges that
    keep every invariant satisfied. Retries on the rare invalid draw and falls
    back to :func:`hero_layout` if no valid variation is found.

    Reachability gate (docs/multi_plan.md sec 5b): unless ``enforce_reachable``
    is disabled, a draw is only accepted if the A* planner can also route every
    home bay to every object spot (and to the delivery pad) at BOTH the deployed
    0.40 m and the 0.45 m stress-margin inflation — the jitter can otherwise seal
    the NE alcove spot, which A* cannot reach even at 0.40 m (verified: ~30% of
    ungated draws). This closes the same alcove-sealing risk the fixed hero
    layout is verified against (it seals only at 0.50 m).

    Args:
        rng: Caller-owned NumPy generator; advances state.
        enforce_reachable: Gate accepted draws on the 0.40/0.45 m plan-
            reachability check (default True). Only turn off for fast geometry-
            only unit tests.

    Returns:
        A validated (and, by default, plan-reachable) :class:`WarehouseLayout`.
    """
    for _ in range(96):
        gap = float(rng.uniform(1.3, 1.8))
        row_cy = float(rng.uniform(1.9, 2.3))
        block_len = float(rng.uniform(1.45, 1.7))
        alcove_x = float(rng.uniform(4.6, 5.4))
        sw_cy = float(rng.uniform(-3.6, -2.9))
        delivery_xy = (float(rng.uniform(5.4, 6.1)), float(rng.uniform(-1.4, -0.6)))

        walls = _perimeter_walls(8.0, 6.0)
        walls += _shelf_walls(gap=gap, row_cy=row_cy, block_len=block_len)
        walls += _partition_walls(alcove_x=alcove_x, sw_cy=sw_cy)
        zones, spawns = _zones_and_spawns(delivery_xy=delivery_xy)

        spots: List[Tuple[float, float]] = []
        for bx, by in _hero_object_spots():
            spots.append((round(bx + float(rng.uniform(-0.3, 0.3)), 3),
                          round(by + float(rng.uniform(-0.3, 0.3)), 3)))

        seed_tag = int(rng.integers(0, 1_000_000))
        layout = WarehouseLayout(
            hall_x=16.0, hall_y=12.0, walls=walls, zones=zones,
            spawn_poses=spawns, object_spots=spots, name=f"sample_{seed_tag}",
        )
        try:
            validate_layout(layout)
        except ValueError:
            continue
        if enforce_reachable and not _all_pairs_reachable(layout)[0]:
            continue
        return layout
    return hero_layout()


# ---------------------------------------------------------------------------
# Multi-room layout (F6): a 20x14 m shell partitioned into four named rooms
# connected by permanent open doorways, so objects are findable only by
# exploring room to room. The rooms are the single source of truth for the
# search partition and the F3 "currently in {room}" position reports.
# ---------------------------------------------------------------------------
_ROOMS_HALL_X: float = 20.0
_ROOMS_HALL_Y: float = 14.0
_PART_H: float = 2.2          # interior divider height (< 2.5 perimeter)
_DOOR_W: float = 2.0          # doorway opening width (>= _MIN_DOORWAY_M)
_MIN_DOORWAY_M: float = 1.8   # required clear doorway width
_SPLIT_Y_S: float = -3.5      # loading room | storage rooms boundary (y)
_SPLIT_Y_N: float = 3.5       # storage rooms | back room boundary (y)
_SPLIT_X: float = 0.0         # storage A | storage B divider (x)


def _rooms_boxes() -> Tuple[Room, ...]:
    """Return the four room bounding boxes tiling the 20x14 m interior.

    The interior [-10, 10] x [-7, 7] is partitioned into a full-width south
    "loading room" strip, a west/east pair of "storage A"/"storage B"
    quadrants split by a central divider, and a full-width north "back room"
    strip. Boxes share edges only (no gaps, no interior overlap).
    """
    hx, hy = _ROOMS_HALL_X / 2.0, _ROOMS_HALL_Y / 2.0
    mid_y = (_SPLIT_Y_S + _SPLIT_Y_N) / 2.0
    return (
        Room("loading room", 0.0, (_SPLIT_Y_S - hy) / 2.0,
             hx, (hy + _SPLIT_Y_S) / 2.0),
        Room("storage A", (_SPLIT_X - hx) / 2.0, mid_y,
             (hx + _SPLIT_X) / 2.0, (_SPLIT_Y_N - _SPLIT_Y_S) / 2.0),
        Room("storage B", (_SPLIT_X + hx) / 2.0, mid_y,
             (hx - _SPLIT_X) / 2.0, (_SPLIT_Y_N - _SPLIT_Y_S) / 2.0),
        Room("back room", 0.0, (_SPLIT_Y_N + hy) / 2.0,
             hx, (hy - _SPLIT_Y_N) / 2.0),
    )


def _divider_walls(orientation: str, line: float, lo: float, hi: float,
                   gaps: List[Tuple[float, float]], *, height: float,
                   rgba: Tuple[float, float, float, float],
                   name_prefix: str) -> List[WallSpec]:
    """Build solid wall segments along a divider line, minus doorway gaps.

    Args:
        orientation: ``"h"`` for a wall at constant ``y=line`` running along x,
            or ``"v"`` for a wall at constant ``x=line`` running along y.
        line: Fixed coordinate of the divider (y for "h", x for "v").
        lo: Lower bound of the running axis (m).
        hi: Upper bound of the running axis (m).
        gaps: ``(center, width)`` doorway openings along the running axis.
        height: Wall block height (m).
        rgba: Wall colour.
        name_prefix: Prefix for generated segment names (``f"{prefix}_{k}"``).

    Returns:
        The solid segments filling ``[lo, hi]`` on the divider line except the
        given gaps (zero-length segments are dropped).
    """
    solids: List[Tuple[float, float]] = []
    cursor = lo
    for center, width in sorted(gaps):
        gstart, gend = center - width / 2.0, center + width / 2.0
        if gstart - cursor > 1e-9:
            solids.append((cursor, gstart))
        cursor = max(cursor, gend)
    if hi - cursor > 1e-9:
        solids.append((cursor, hi))

    t = _WALL_T
    walls: List[WallSpec] = []
    for k, (a, b) in enumerate(solids):
        mid = (a + b) / 2.0
        half = (b - a) / 2.0
        if orientation == "h":
            walls.append(WallSpec(mid, line, half, t, height=height,
                                  rgba=rgba, name=f"{name_prefix}_{k}"))
        else:
            walls.append(WallSpec(line, mid, t, half, height=height,
                                  rgba=rgba, name=f"{name_prefix}_{k}"))
    return walls


def _room_divider_walls() -> List[WallSpec]:
    """Build the three interior dividers with their four open doorways."""
    hx = _ROOMS_HALL_X / 2.0
    doors = [(_SPLIT_X - hx / 2.0, _DOOR_W), (_SPLIT_X + hx / 2.0, _DOOR_W)]
    walls: List[WallSpec] = []
    # South divider: loading <-> storage A (west door) and <-> storage B (east).
    walls += _divider_walls("h", _SPLIT_Y_S, -hx, hx, doors,
                            height=_PART_H, rgba=_PART_RGBA,
                            name_prefix="wall_div_s")
    # North divider: storage A <-> back (west door) and storage B <-> back (east).
    walls += _divider_walls("h", _SPLIT_Y_N, -hx, hx, doors,
                            height=_PART_H, rgba=_PART_RGBA,
                            name_prefix="wall_div_n")
    # Vertical divider between storage A and storage B: solid, no doorway (the
    # A<->B route runs through the loading or back room, giving A* a cycle).
    walls += _divider_walls("v", _SPLIT_X, _SPLIT_Y_S, _SPLIT_Y_N, [],
                            height=_PART_H, rgba=_PART_RGBA,
                            name_prefix="wall_div_ab")
    return walls


def _room_shelves() -> List[WallSpec]:
    """Build the four storage shelf blocks (two in storage A, two in storage B).

    Blocks are pulled clear of the ~x=+/-5 north-south doorway corridors so a
    robot can transit each storage room even at 0.45 m planner inflation.
    """
    hy = 0.35  # 0.7 m deep, matching the hero shelves
    return [
        WallSpec(-7.2, 1.4, 1.4, hy, height=1.8, rgba=_SHELF_RGBA, name="shelf_A_n"),
        WallSpec(-3.0, -1.4, 1.4, hy, height=1.8, rgba=_SHELF_RGBA, name="shelf_A_s"),
        WallSpec(7.2, 1.4, 1.4, hy, height=1.8, rgba=_SHELF_RGBA, name="shelf_B_n"),
        WallSpec(3.0, -1.4, 1.4, hy, height=1.8, rgba=_SHELF_RGBA, name="shelf_B_s"),
    ]


def _room_zones_and_spawns() -> Tuple[
    List[Zone], Dict[str, Tuple[float, float, float]]
]:
    """Build the delivery pad (in storage B), the four home bays and spawns."""
    zones: List[Zone] = [
        Zone("delivery", 8.0, -2.0, 1.0, 1.0, _DELIVERY_RGBA),  # 2x2 m, storage B
    ]
    spawns: Dict[str, Tuple[float, float, float]] = {}
    bay_x = {"Alpha": -6.0, "Bravo": -2.0, "Charlie": 2.0, "Delta": 6.0}
    for cs in CALLSIGNS:
        bx = bay_x[cs]
        zones.append(Zone(f"bay_{cs}", bx, -5.5, 0.8, 0.7, _BAY_RGBA[cs]))
        spawns[cs] = (bx, -5.5, math.pi / 2.0)  # facing +y into the loading room
    return zones, spawns


def _room_object_spots() -> List[Tuple[float, float]]:
    """Return the 11 object spots spread across all four rooms.

    Several sit in deep corners not visible from any doorway (the storage and
    back-room corners), forcing room-to-room exploration to find the target.
    """
    return [
        (-9.0, -6.0),  # loading room, SW corner
        (9.0, -6.0),   # loading room, SE corner
        (-9.0, 2.6),   # storage A, deep NW corner (occluded by shelf_A_n)
        (-9.0, -2.6),  # storage A, deep SW corner
        (-2.5, 2.6),   # storage A, near the A|B divider (north)
        (9.0, 2.6),    # storage B, deep NE corner (occluded by shelf_B_n)
        (2.5, 2.6),    # storage B, near the A|B divider (north)
        (2.5, -2.6),   # storage B, near the A|B divider (south)
        (-9.0, 6.0),   # back room, deep NW corner
        (9.0, 6.0),    # back room, deep NE corner
        (0.0, 6.0),    # back room, north-central
    ]


def _divider_gaps(layout: WarehouseLayout, orientation: str, line: float,
                  lo: float, hi: float) -> List[float]:
    """Return the widths of the open gaps along one interior divider line."""
    tol = 1e-6
    spans: List[Tuple[float, float]] = []
    for w in layout.walls:
        if orientation == "h" and abs(w.cy - line) < tol and w.half_y <= _WALL_T + tol:
            spans.append((w.cx - w.half_x, w.cx + w.half_x))
        elif orientation == "v" and abs(w.cx - line) < tol and w.half_x <= _WALL_T + tol:
            spans.append((w.cy - w.half_y, w.cy + w.half_y))
    spans.sort()
    gaps: List[float] = []
    cursor = lo
    for a, b in spans:
        if a - cursor > tol:
            gaps.append(a - cursor)
        cursor = max(cursor, b)
    if hi - cursor > tol:
        gaps.append(hi - cursor)
    return gaps


def _validate_rooms(rooms: Tuple[Room, ...], half_x: float,
                    half_y: float) -> None:
    """Raise if the rooms do not tile the interior without gaps or overlaps."""
    if not rooms:
        raise ValueError("rooms_layout must define at least one room")
    for i in range(len(rooms)):
        for j in range(i + 1, len(rooms)):
            a, b = rooms[i], rooms[j]
            if (abs(a.cx - b.cx) < a.half_x + b.half_x - 1e-6
                    and abs(a.cy - b.cy) < a.half_y + b.half_y - 1e-6):
                raise ValueError(f"rooms {a.name!r},{b.name!r} overlap")
        r = rooms[i]
        if (r.cx - r.half_x < -half_x - 1e-6 or r.cx + r.half_x > half_x + 1e-6
                or r.cy - r.half_y < -half_y - 1e-6
                or r.cy + r.half_y > half_y + 1e-6):
            raise ValueError(f"room {r.name!r} extends outside the hall")
    area = sum(4.0 * r.half_x * r.half_y for r in rooms)
    hall_area = 4.0 * half_x * half_y
    if abs(area - hall_area) > 1e-6:
        raise ValueError(
            f"rooms do not tile interior: total area {area:.3f} != {hall_area:.3f}"
        )


def validate_rooms_layout(layout: WarehouseLayout) -> None:
    """Raise ValueError if the multi-room layout violates an invariant.

    Checks (mirroring the hero validity contract, minus the single-hall
    shelf-aisle rule): every spawn/object_spot >= 0.5 m from any wall, spots
    and spawns non-overlapping, spots outside the delivery pad, zones clear of
    walls, every doorway >= 1.8 m wide, and the rooms tile the interior with no
    gaps or overlaps. Plan-reachability at 0.40/0.45 m inflation is a separate
    unit-test gate (it needs the planner).

    Args:
        layout: Multi-room layout to validate.

    Raises:
        ValueError: On the first invariant that fails.
    """
    half_x, half_y = layout.hall_x / 2.0, layout.hall_y / 2.0

    for cs, (sx, sy, _yaw) in layout.spawn_poses.items():
        if not (-half_x < sx < half_x and -half_y < sy < half_y):
            raise ValueError(f"spawn {cs} at ({sx},{sy}) outside hall")
        for w in layout.walls:
            d = _point_rect_distance(sx, sy, w)
            if d < _MIN_CLEAR_M - 1e-6:
                raise ValueError(
                    f"spawn {cs} only {d:.3f} m from wall {w.name} "
                    f"(< {_MIN_CLEAR_M} m)"
                )
    for i, (ox, oy) in enumerate(layout.object_spots):
        for w in layout.walls:
            d = _point_rect_distance(ox, oy, w)
            if d < _MIN_CLEAR_M - 1e-6:
                raise ValueError(
                    f"object_spot {i} at ({ox:.2f},{oy:.2f}) only {d:.3f} m "
                    f"from wall {w.name} (< {_MIN_CLEAR_M} m)"
                )

    for i in range(len(layout.object_spots)):
        for j in range(i + 1, len(layout.object_spots)):
            ax, ay = layout.object_spots[i]
            bx, by = layout.object_spots[j]
            if math.hypot(ax - bx, ay - by) < _MIN_SPOT_SPACING_M - 1e-6:
                raise ValueError(f"object_spots {i},{j} closer than "
                                 f"{_MIN_SPOT_SPACING_M} m")
    spawns = list(layout.spawn_poses.values())
    for i in range(len(spawns)):
        for j in range(i + 1, len(spawns)):
            if math.hypot(spawns[i][0] - spawns[j][0],
                          spawns[i][1] - spawns[j][1]) < _MIN_SPAWN_SPACING_M:
                raise ValueError(f"spawns {i},{j} closer than "
                                 f"{_MIN_SPAWN_SPACING_M} m")

    delivery = next((z for z in layout.zones if z.name == "delivery"), None)
    if delivery is not None:
        for i, (ox, oy) in enumerate(layout.object_spots):
            if (abs(ox - delivery.cx) <= delivery.half_x
                    and abs(oy - delivery.cy) <= delivery.half_y):
                raise ValueError(f"object_spot {i} lies inside delivery pad")

    for z in layout.zones:
        for w in layout.walls:
            if _obb_overlap(w, z.cx, z.cy, z.half_x, z.half_y):
                raise ValueError(f"zone {z.name} overlaps wall {w.name}")

    # --- Doorways: every gap in the two horizontal dividers >= 1.8 m. ---
    for line, tag in ((_SPLIT_Y_S, "south"), (_SPLIT_Y_N, "north")):
        for gw in _divider_gaps(layout, "h", line, -half_x, half_x):
            if gw < _MIN_DOORWAY_M - 1e-6:
                raise ValueError(
                    f"{tag} divider doorway {gw:.3f} m < {_MIN_DOORWAY_M} m minimum"
                )

    _validate_rooms(layout.rooms, half_x, half_y)


def rooms_layout() -> WarehouseLayout:
    """Return the fixed multi-room warehouse layout (F6).

    Geometry: a 20x14 m shell partitioned into four named rooms connected by
    permanent open doorways (no swinging panels):

    * "loading room" — full-width south strip (3.5 m deep) with the four
      colour-accented home bays and spawns (identical callsigns/colours to the
      hero layout, facing +y into the hall).
    * "storage A" — west upper quadrant, two shelf blocks.
    * "storage B" — east upper quadrant, two shelf blocks + the 2x2 m green
      delivery pad.
    * "back room" — full-width north strip (3.5 m deep).

    Four 2.0 m doorways connect loading<->A, loading<->B, A<->back and B<->back
    in a cycle (the A|B divider is solid), so A* always has a route choice.
    Eleven object spots are spread across all four rooms, several in deep
    corners not visible from any doorway.

    Returns:
        A validated :class:`WarehouseLayout` with ``rooms`` populated.

    Raises:
        ValueError: If the assembled layout violates an invariant (a bug).
    """
    half_x, half_y = _ROOMS_HALL_X / 2.0, _ROOMS_HALL_Y / 2.0
    walls = _perimeter_walls(half_x, half_y)
    walls += _room_divider_walls()
    walls += _room_shelves()
    zones, spawns = _room_zones_and_spawns()
    layout = WarehouseLayout(
        hall_x=_ROOMS_HALL_X, hall_y=_ROOMS_HALL_Y, walls=walls, zones=zones,
        spawn_poses=spawns, object_spots=_room_object_spots(),
        rooms=_rooms_boxes(), name="rooms",
    )
    validate_rooms_layout(layout)
    return layout


def _point_box_distance(px: float, py: float, room: Room) -> float:
    """Return the shortest distance from a point to a room's axis-aligned box."""
    ox = max(abs(px - room.cx) - room.half_x, 0.0)
    oy = max(abs(py - room.cy) - room.half_y, 0.0)
    return math.hypot(ox, oy)


def room_of(layout: WarehouseLayout, xy: Tuple[float, float]) -> str:
    """Return the name of the room containing ``xy`` (nearest on a boundary).

    A point strictly inside one room box returns that room. A point on a shared
    boundary or in a doorway (contained by, or equidistant from, two boxes)
    resolves to the room whose centre is nearest, so every point maps to
    exactly one name.

    Args:
        layout: A layout whose ``rooms`` are populated (e.g. ``rooms_layout``).
        xy: World point (x, y) in metres.

    Returns:
        The containing (or nearest) room name.

    Raises:
        ValueError: If ``layout`` has no rooms.
    """
    if not layout.rooms:
        raise ValueError("layout has no rooms; room_of requires a rooms_layout()")
    px, py = float(xy[0]), float(xy[1])
    best: Room = layout.rooms[0]
    best_key = (float("inf"), float("inf"))
    for r in layout.rooms:
        d = _point_box_distance(px, py, r)
        cd = math.hypot(px - r.cx, py - r.cy)
        key = (d, cd)
        if key < best_key:
            best_key = key
            best = r
    return best.name


# ---------------------------------------------------------------------------
# Plan-reachability gate (docs/multi_plan.md sec 5b) + randomized rooms family.
# ---------------------------------------------------------------------------
# The deployed planner clears the robot footprint at 0.40 m; 0.45 m is the stress
# margin the hero + rooms layouts are verified against (the hero NE alcove seals
# only at 0.50 m). Every seeded layout variant must route at BOTH before use.
_PLAN_INFLATIONS: Tuple[float, float] = (0.40, 0.45)


def _all_pairs_reachable(
    layout: WarehouseLayout, *,
    inflations: Tuple[float, ...] = _PLAN_INFLATIONS,
    resolution: float = 0.1, snap_radius_m: float = 0.4,
    include_delivery: bool = True,
) -> Tuple[bool, str]:
    """Whether every bay -> object_spot (+ delivery pad) plans at each inflation.

    Implements the docs/multi_plan.md sec 5b gate: a seeded layout variant is
    usable only if grid A* can route every home bay to every object spot (and,
    by default, to the delivery pad) at BOTH the deployed 0.40 m clearance
    inflation and the 0.45 m stress margin. It rasterizes the walls-only
    occupancy grid the deployed planner shares (``occupancy_grid`` + ``inflate``)
    and snaps endpoints within ``snap_radius_m`` exactly as the fleet's planner
    does, so a pass here means the fleet can actually navigate the layout.

    The planner imports are deferred so ``layout.py`` keeps its load-time purity
    (no module-level planner/MuJoCo dependency); the planner is pulled in only
    when a variant is gated. Grid reachability is undirected and transitive, so
    gating bay->spot and bay->delivery also guarantees spot->delivery (the carry
    leg) whenever both endpoints share the bay's connected component.

    Args:
        layout: Layout to test (hero or rooms family).
        inflations: Robot-clearance dilation radii to require routable (m).
        resolution: Occupancy grid cell size (m).
        snap_radius_m: Endpoint snap radius, matching the deployed planner (m).
        include_delivery: Also require every bay -> delivery-pad route.

    Returns:
        ``(ok, diagnostic)`` — ``ok`` True when all pairs route at all
        inflations; otherwise ``diagnostic`` names the first failing
        bay/goal/inflation.
    """
    from code.planner.astar import PathNotFoundError, plan_path
    from code.planner.grid import inflate
    from code.warehouse.occupancy import occupancy_grid

    goals: List[Tuple[str, Tuple[float, float]]] = [
        (f"spot{i}", (float(x), float(y)))
        for i, (x, y) in enumerate(layout.object_spots)
    ]
    if include_delivery:
        for z in layout.zones:
            if z.name == "delivery":
                goals.append(("delivery", (float(z.cx), float(z.cy))))
                break

    og = occupancy_grid(layout, resolution)
    for r in inflations:
        ig = inflate(og, r)
        for cs, (sx, sy, _yaw) in layout.spawn_poses.items():
            for tag, (gx, gy) in goals:
                try:
                    path = plan_path(ig, (float(sx), float(sy)), (gx, gy),
                                     snap_radius_m=snap_radius_m)
                except PathNotFoundError:
                    return (False, f"bay {cs} -> {tag} ({gx:.2f},{gy:.2f}) "
                                   f"unreachable at inflation {r:.2f}")
                if not path:
                    return (False, f"bay {cs} -> {tag} ({gx:.2f},{gy:.2f}) "
                                   f"empty path at inflation {r:.2f}")
    return True, ""


# Randomized rooms-family jitter ranges (metres). The three split lines stay
# fixed so the four room boxes and their tiling are preserved verbatim; only the
# doorway, shelf, object-spot and delivery-pad placements vary WITHIN their room.
_RM_DOOR_W: Tuple[float, float] = (1.9, 2.3)      # doorway width (>= _MIN_DOORWAY_M)
_RM_WDOOR_X: Tuple[float, float] = (-7.5, -2.5)   # west doorway centre (storage A x)
_RM_EDOOR_X: Tuple[float, float] = (2.5, 7.5)     # east doorway centre (storage B x)
_RM_SHELF_HX: Tuple[float, float] = (1.0, 1.4)    # shelf half-length along x
_RM_ASHELF_X: Tuple[float, float] = (-8.5, -2.2)  # storage-A shelf cx
_RM_BSHELF_X: Tuple[float, float] = (2.2, 8.5)    # storage-B shelf cx
_RM_NSHELF_Y: Tuple[float, float] = (0.6, 2.4)    # north-of-centre shelf cy band
_RM_SSHELF_Y: Tuple[float, float] = (-2.4, -0.6)  # south-of-centre shelf cy band
_RM_DELIV_X: Tuple[float, float] = (1.6, 8.6)     # delivery-pad cx (storage B)
_RM_DELIV_Y: Tuple[float, float] = (-2.4, 2.4)    # delivery-pad cy (storage B)
_RM_SPOT_MARGIN_M: float = 0.7                    # keep spots this far inside a room box
_RM_SPOTS_MIN: int = 2                            # per-room object-spot count (inclusive)
_RM_SPOTS_MAX: int = 3


def _uni(rng: np.random.Generator, lohi: Tuple[float, float]) -> float:
    """Draw one float uniformly from an inclusive ``(lo, hi)`` range."""
    return float(rng.uniform(lohi[0], lohi[1]))


def _sampled_room_walls(rng: np.random.Generator) -> List[WallSpec]:
    """Build perimeter + jittered dividers (4 doorways) + jittered storage shelves.

    Doorway centres jitter independently on the north and south dividers (west
    door within storage A's x-half, east within storage B's), and each width is
    drawn >= :data:`_MIN_DOORWAY_M`. The four storage shelves jitter in x/y and
    half-length within their room. The solid A|B divider is preserved so the only
    A<->B routes run through the loading/back rooms.
    """
    hx = _ROOMS_HALL_X / 2.0
    walls = _perimeter_walls(hx, _ROOMS_HALL_Y / 2.0)
    for line, prefix in ((_SPLIT_Y_S, "wall_div_s"), (_SPLIT_Y_N, "wall_div_n")):
        gaps = [(_uni(rng, _RM_WDOOR_X), _uni(rng, _RM_DOOR_W)),
                (_uni(rng, _RM_EDOOR_X), _uni(rng, _RM_DOOR_W))]
        walls += _divider_walls("h", line, -hx, hx, gaps, height=_PART_H,
                                rgba=_PART_RGBA, name_prefix=prefix)
    walls += _divider_walls("v", _SPLIT_X, _SPLIT_Y_S, _SPLIT_Y_N, [],
                            height=_PART_H, rgba=_PART_RGBA,
                            name_prefix="wall_div_ab")
    shelf_specs = (
        ("shelf_A_n", _RM_ASHELF_X, _RM_NSHELF_Y),
        ("shelf_A_s", _RM_ASHELF_X, _RM_SSHELF_Y),
        ("shelf_B_n", _RM_BSHELF_X, _RM_NSHELF_Y),
        ("shelf_B_s", _RM_BSHELF_X, _RM_SSHELF_Y),
    )
    for name, xr, yr in shelf_specs:
        walls.append(WallSpec(round(_uni(rng, xr), 3), round(_uni(rng, yr), 3),
                              round(_uni(rng, _RM_SHELF_HX), 3), 0.35,
                              height=1.8, rgba=_SHELF_RGBA, name=name))
    return walls


def _sampled_room_zones_and_spawns(
    rng: np.random.Generator,
) -> Tuple[List[Zone], Dict[str, Tuple[float, float, float]]]:
    """Build the jittered delivery pad (in storage B) + the fixed home bays.

    The four home bays / spawns are identical to the fixed rooms layout (kept in
    the loading room so the reachability gate always starts from the same
    poses); only the 2x2 m delivery pad's centre jitters within storage B.
    """
    zones: List[Zone] = [
        Zone("delivery", round(_uni(rng, _RM_DELIV_X), 3),
             round(_uni(rng, _RM_DELIV_Y), 3), 1.0, 1.0, _DELIVERY_RGBA),
    ]
    spawns: Dict[str, Tuple[float, float, float]] = {}
    bay_x = {"Alpha": -6.0, "Bravo": -2.0, "Charlie": 2.0, "Delta": 6.0}
    for cs in CALLSIGNS:
        bx = bay_x[cs]
        zones.append(Zone(f"bay_{cs}", bx, -5.5, 0.8, 0.7, _BAY_RGBA[cs]))
        spawns[cs] = (bx, -5.5, math.pi / 2.0)
    return zones, spawns


def _sampled_room_spots(
    rng: np.random.Generator, walls: List[WallSpec], delivery: Zone,
    rooms: Tuple[Room, ...], probe: WarehouseLayout,
) -> Optional[List[Tuple[float, float]]]:
    """Sample 2-3 clear object spots per room (or None if a room can't be filled).

    Each spot is drawn inside its room box (shrunk by :data:`_RM_SPOT_MARGIN_M`),
    kept >= 0.5 m from every wall, >= 0.8 m from other spots, outside the delivery
    pad, and confirmed by :func:`room_of` to lie in the intended room. Returns
    ``None`` when a room cannot place its minimum count within the inner attempt
    budget (the caller then resamples the whole layout).
    """
    spots: List[Tuple[float, float]] = []

    def _clear(x: float, y: float) -> bool:
        if any(_point_rect_distance(x, y, w) < _MIN_CLEAR_M for w in walls):
            return False
        if any(math.hypot(x - sx, y - sy) < _MIN_SPOT_SPACING_M
               for sx, sy in spots):
            return False
        if (abs(x - delivery.cx) <= delivery.half_x
                and abs(y - delivery.cy) <= delivery.half_y):
            return False
        return True

    for room in rooms:
        target = int(rng.integers(_RM_SPOTS_MIN, _RM_SPOTS_MAX + 1))
        placed = 0
        for _ in range(80):
            if placed >= target:
                break
            x = _uni(rng, (room.cx - room.half_x + _RM_SPOT_MARGIN_M,
                           room.cx + room.half_x - _RM_SPOT_MARGIN_M))
            y = _uni(rng, (room.cy - room.half_y + _RM_SPOT_MARGIN_M,
                           room.cy + room.half_y - _RM_SPOT_MARGIN_M))
            if _clear(x, y) and room_of(probe, (x, y)) == room.name:
                spots.append((round(x, 3), round(y, 3)))
                placed += 1
        if placed < _RM_SPOTS_MIN:
            return None
    return spots


def sample_rooms_layout(rng: np.random.Generator, *, max_attempts: int = 200,
                        enforce_reachable: bool = True) -> WarehouseLayout:
    """Return a seeded randomized variant of the four-room layout (F6 family).

    Reject-and-resample generator for a randomized 4-room warehouse that keeps
    the ``rooms_layout`` structure — same 20x14 m shell, same four named rooms
    and tiling, same solid A|B divider, same home bays — while randomizing:

    * the four doorway centres (independently on the north/south dividers) and
      their widths (always >= :data:`_MIN_DOORWAY_M`);
    * the four storage-shelf placements (x, y and half-length within their room);
    * 2-3 object spots per room (>= 0.5 m wall clearance, >= 0.8 m apart);
    * the 2x2 m delivery pad position within storage B.

    Every candidate must pass :func:`validate_rooms_layout` AND the sec-5b
    plan-reachability gate (:func:`_all_pairs_reachable`): every bay routes to
    every object spot and the delivery pad at BOTH 0.40 m and 0.45 m inflation.
    Deterministic given ``rng`` (the same seed replays the same layout).

    Args:
        rng: Caller-owned NumPy generator; advances state.
        max_attempts: Cap on full resample attempts before giving up.
        enforce_reachable: Run the 0.40/0.45 m reachability gate (default True;
            disable only for fast geometry-only unit tests).

    Returns:
        A validated, plan-reachable :class:`WarehouseLayout` with ``rooms`` set.

    Raises:
        RuntimeError: If no valid + reachable layout is found in ``max_attempts``
            attempts (message includes the most recent rejection diagnostics).
    """
    rooms = _rooms_boxes()
    seed_tag = int(rng.integers(0, 1_000_000))
    diagnostics: List[str] = []
    for attempt in range(max_attempts):
        walls = _sampled_room_walls(rng)
        zones, spawns = _sampled_room_zones_and_spawns(rng)
        probe = WarehouseLayout(
            hall_x=_ROOMS_HALL_X, hall_y=_ROOMS_HALL_Y, walls=walls, zones=zones,
            spawn_poses=spawns, object_spots=[], rooms=rooms)
        spots = _sampled_room_spots(rng, walls, zones[0], rooms, probe)
        if spots is None:
            diagnostics.append(f"attempt {attempt}: could not place per-room spots")
            continue
        layout = WarehouseLayout(
            hall_x=_ROOMS_HALL_X, hall_y=_ROOMS_HALL_Y, walls=walls, zones=zones,
            spawn_poses=spawns, object_spots=spots, rooms=rooms,
            name=f"sample_rooms_{seed_tag}")
        try:
            validate_rooms_layout(layout)
        except ValueError as e:
            diagnostics.append(f"attempt {attempt}: invalid geometry: {e}")
            continue
        if enforce_reachable:
            ok, diag = _all_pairs_reachable(layout)
            if not ok:
                diagnostics.append(f"attempt {attempt}: unreachable: {diag}")
                continue
        return layout
    raise RuntimeError(
        f"sample_rooms_layout: no valid+reachable layout in {max_attempts} "
        f"attempts (seed_tag={seed_tag}); last diagnostics: {diagnostics[-6:]}"
    )
