"""Unit tests for the multi-room F6 layout (rooms_layout, Room, room_of).

Covers the layout invariants (counts, doorway widths, clearances, tiling), the
``room_of`` point->name resolver, and the CRITICAL plan-reachability gate: every
bay->object_spot pair must be routable at 0.40 AND 0.45 m planner inflation (the
hero NE alcove sealed at 0.50 m; the open doorways are the same risk class).
"""

import dataclasses
import math
import unittest

from code.planner.astar import PathNotFoundError, plan_path
from code.planner.grid import inflate
from code.warehouse.layout import (
    CALLSIGNS,
    Room,
    WarehouseLayout,
    _BAY_RGBA,
    _MIN_DOORWAY_M,
    _divider_gaps,
    hero_layout,
    room_of,
    rooms_layout,
    validate_rooms_layout,
)
from code.warehouse.occupancy import occupancy_grid

_MIN_CLEAR = 0.5
_ROOM_NAMES = {"loading room", "storage A", "storage B", "back room"}


def _point_rect_distance(px, py, w):
    """Distance from a point to an axis-aligned wall footprint (local copy)."""
    ox = max(abs(px - w.cx) - w.half_x, 0.0)
    oy = max(abs(py - w.cy) - w.half_y, 0.0)
    return math.hypot(ox, oy)


class TestRoomDataclass(unittest.TestCase):
    def test_room_is_frozen(self) -> None:
        r = Room("storage A", -5.0, 0.0, 5.0, 3.5)
        with self.assertRaises(dataclasses.FrozenInstanceError):
            r.cx = 1.0  # type: ignore[misc]

    def test_layout_rooms_default_empty(self) -> None:
        self.assertEqual(WarehouseLayout().rooms, ())

    def test_hero_has_no_rooms(self) -> None:
        # The additive rooms field must leave the frozen hero layout unchanged.
        self.assertEqual(hero_layout().rooms, ())


class TestRoomsLayout(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.layout = rooms_layout()

    def test_validates(self) -> None:
        validate_rooms_layout(self.layout)  # must not raise

    def test_shell_is_20x14(self) -> None:
        self.assertEqual((self.layout.hall_x, self.layout.hall_y), (20.0, 14.0))

    def test_name(self) -> None:
        self.assertEqual(self.layout.name, "rooms")

    def test_four_named_rooms(self) -> None:
        self.assertEqual({r.name for r in self.layout.rooms}, _ROOM_NAMES)

    def test_wall_composition(self) -> None:
        names = [w.name for w in self.layout.walls]
        self.assertEqual(len(names), len(set(names)))  # unique
        perim = [n for n in names if n.startswith("wall_") and "div" not in n]
        dividers = [n for n in names if n.startswith("wall_div_")]
        shelves = [n for n in names if n.startswith("shelf_")]
        self.assertEqual(len(perim), 4)       # perimeter
        self.assertEqual(len(shelves), 4)     # 2 in storage A + 2 in storage B
        # South divider (2 doors -> 3 segs), north (3 segs), A|B divider (1 seg).
        self.assertEqual(len(dividers), 7)

    def test_bays_identical_to_hero(self) -> None:
        self.assertEqual(set(self.layout.spawn_poses), set(CALLSIGNS))
        for cs in CALLSIGNS:
            bay = next(z for z in self.layout.zones if z.name == f"bay_{cs}")
            self.assertEqual(bay.rgba, _BAY_RGBA[cs])  # colours match hero
        for _cs, (_x, y, yaw) in self.layout.spawn_poses.items():
            self.assertLess(y, 0.0)                      # south loading room
            self.assertAlmostEqual(yaw, math.pi / 2.0)   # facing into the hall

    def test_delivery_pad_is_2x2(self) -> None:
        d = next(z for z in self.layout.zones if z.name == "delivery")
        self.assertEqual((d.half_x, d.half_y), (1.0, 1.0))
        # Delivery pad sits in storage B (east upper quadrant).
        self.assertEqual(room_of(self.layout, (d.cx, d.cy)), "storage B")

    def test_object_spots_count_and_spread(self) -> None:
        spots = self.layout.object_spots
        self.assertGreaterEqual(len(spots), 10)
        self.assertLessEqual(len(spots), 12)
        by_room = {name: 0 for name in _ROOM_NAMES}
        for xy in spots:
            by_room[room_of(self.layout, xy)] += 1
        for name in _ROOM_NAMES:  # every room holds at least one spot
            self.assertGreaterEqual(by_room[name], 1, f"no spot in {name}")

    def test_spawn_and_spot_clearance(self) -> None:
        for _cs, (x, y, _yaw) in self.layout.spawn_poses.items():
            for w in self.layout.walls:
                self.assertGreaterEqual(_point_rect_distance(x, y, w), _MIN_CLEAR)
        for (x, y) in self.layout.object_spots:
            for w in self.layout.walls:
                self.assertGreaterEqual(_point_rect_distance(x, y, w), _MIN_CLEAR)

    def test_four_doorways_wide_enough(self) -> None:
        south = _divider_gaps(self.layout, "h", -3.5, -10.0, 10.0)
        north = _divider_gaps(self.layout, "h", 3.5, -10.0, 10.0)
        self.assertEqual(len(south), 2)
        self.assertEqual(len(north), 2)
        for gw in south + north:
            self.assertGreaterEqual(gw, _MIN_DOORWAY_M)

    def test_ab_divider_is_solid(self) -> None:
        # No doorway between storage A and storage B (forces the routing cycle).
        self.assertEqual(_divider_gaps(self.layout, "v", 0.0, -3.5, 3.5), [])


class TestRoomsTileInterior(unittest.TestCase):
    """Every free occupancy cell belongs to exactly one room bounding box."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.layout = rooms_layout()
        cls.og = occupancy_grid(cls.layout, resolution=0.1)

    def test_grid_covers_20x14(self) -> None:
        self.assertEqual(self.og.shape, (140, 200))

    def test_free_cells_partition_exactly(self) -> None:
        og, rooms = self.og, self.layout.rooms
        ny, nx = og.shape
        gaps = multi = 0
        for iy in range(ny):
            for ix in range(nx):
                if og.grid[iy, ix]:
                    continue
                x, y = og.cell_to_world((iy, ix))
                cnt = sum(
                    1 for r in rooms
                    if r.cx - r.half_x < x < r.cx + r.half_x
                    and r.cy - r.half_y < y < r.cy + r.half_y
                )
                gaps += cnt == 0
                multi += cnt > 1
        self.assertEqual(gaps, 0, "free cells outside every room bbox (gap)")
        self.assertEqual(multi, 0, "free cells inside >1 room bbox (overlap)")


class TestRoomOf(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.layout = rooms_layout()

    def test_interior_points(self) -> None:
        cases = {
            (-6.0, -5.5): "loading room",
            (0.0, -5.0): "loading room",
            (-9.0, 2.6): "storage A",
            (-2.5, 2.6): "storage A",
            (9.0, 2.6): "storage B",
            (2.5, -2.6): "storage B",
            (0.0, 6.0): "back room",
            (-9.0, 6.0): "back room",
        }
        for xy, name in cases.items():
            self.assertEqual(room_of(self.layout, xy), name, xy)

    def test_room_centres_map_to_self(self) -> None:
        for r in self.layout.rooms:
            self.assertEqual(room_of(self.layout, (r.cx, r.cy)), r.name)

    def test_boundary_resolves_to_nearest(self) -> None:
        # On the loading|storage-A boundary: nearer storage A centre wins.
        self.assertEqual(room_of(self.layout, (-5.0, -3.5)), "storage A")
        # Every result is a valid, human-readable room name.
        self.assertIn(room_of(self.layout, (0.0, 3.5)), _ROOM_NAMES)

    def test_raises_without_rooms(self) -> None:
        with self.assertRaises(ValueError):
            room_of(hero_layout(), (0.0, 0.0))


class TestReachabilityGate(unittest.TestCase):
    """CRITICAL: every bay -> object_spot pair routes at 0.40 AND 0.45 m inflation."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.layout = rooms_layout()
        cls.og = occupancy_grid(cls.layout, resolution=0.1)

    def _assert_all_reachable(self, inflation: float) -> None:
        ig = inflate(self.og, inflation)
        for cs, (sx, sy, _yaw) in self.layout.spawn_poses.items():
            for i, (ox, oy) in enumerate(self.layout.object_spots):
                try:
                    path = plan_path(ig, (sx, sy), (ox, oy), snap_radius_m=0.4)
                except PathNotFoundError as e:  # pragma: no cover - failure path
                    self.fail(
                        f"unreachable at inflation {inflation}: bay {cs} -> "
                        f"spot {i} ({ox},{oy}): {e}"
                    )
                self.assertGreater(len(path), 0)

    def test_reachable_at_040(self) -> None:
        self._assert_all_reachable(0.40)

    def test_reachable_at_045(self) -> None:
        self._assert_all_reachable(0.45)


if __name__ == "__main__":
    unittest.main()
