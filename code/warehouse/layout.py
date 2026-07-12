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
from typing import Dict, List, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Robot callsigns and their demo accent colours (helmet/pad tint per robot).
# ---------------------------------------------------------------------------
CALLSIGNS: Tuple[str, ...] = ("Alpha", "Bravo", "Charlie", "Delta")
_BAY_RGBA: Dict[str, Tuple[float, float, float, float]] = {
    "Alpha": (0.86, 0.16, 0.16, 0.55),    # red
    "Bravo": (0.20, 0.35, 0.86, 0.55),    # blue
    "Charlie": (0.92, 0.80, 0.16, 0.55),  # yellow
    "Delta": (0.59, 0.24, 0.78, 0.55),    # purple
}

_PERIM_RGBA: Tuple[float, float, float, float] = (0.74, 0.75, 0.78, 1.0)
_SHELF_RGBA: Tuple[float, float, float, float] = (0.55, 0.40, 0.24, 1.0)
_PART_RGBA: Tuple[float, float, float, float] = (0.62, 0.64, 0.68, 1.0)
_DELIVERY_RGBA: Tuple[float, float, float, float] = (0.20, 0.72, 0.32, 0.55)

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
class WarehouseLayout:
    """A complete warehouse layout (walls, zones, spawns, object spots).

    Attributes:
        hall_x: Full hall extent along x (m); hall spans [-hall_x/2, hall_x/2].
        hall_y: Full hall extent along y (m); hall spans [-hall_y/2, hall_y/2].
        walls: Perimeter + shelf + partition blocks.
        zones: Visual floor pads (delivery pad + home bays).
        spawn_poses: callsign -> (x, y, yaw) robot start pose.
        object_spots: (x, y) positions where objects may be placed.
        name: Short layout identifier (e.g. "hero", "sample_7").
    """

    hall_x: float = 16.0
    hall_y: float = 12.0
    walls: List[WallSpec] = dataclasses.field(default_factory=list)
    zones: List[Zone] = dataclasses.field(default_factory=list)
    spawn_poses: Dict[str, Tuple[float, float, float]] = dataclasses.field(
        default_factory=dict
    )
    object_spots: List[Tuple[float, float]] = dataclasses.field(default_factory=list)
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


def sample_layout(rng: np.random.Generator) -> WarehouseLayout:
    """Return a seeded, validity-preserving variation of the hero layout.

    Jitters the mid-row gap, the shelf-row separation, the NE alcove x, the SW
    wall y, the delivery pad position and each object spot within ranges that
    keep every invariant satisfied. Retries on the rare invalid draw and falls
    back to :func:`hero_layout` if no valid variation is found.

    Args:
        rng: Caller-owned NumPy generator; advances state.

    Returns:
        A validated :class:`WarehouseLayout`.
    """
    for _ in range(64):
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
            return layout
        except ValueError:
            continue
    return hero_layout()
