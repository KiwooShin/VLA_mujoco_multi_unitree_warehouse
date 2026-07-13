"""Unit tests for the six-robot multi-room layout (rooms6_layout).

Covers the scaled-up 24x16 m layout invariants (six bays, four named rooms, one
extra shelf per storage room, fourteen spots, >= 2.0 m doorways), ``room_of`` on
the larger footprint, the CRITICAL plan-reachability gate (every one of the SIX
bays routes to every object spot at 0.40 AND 0.45 m inflation), and the seeded
``sample_rooms6_layout`` family. The four-robot ``rooms_layout`` must stay
byte-identical, so those layouts are re-asserted here as a regression guard.
"""

import dataclasses
import math
import unittest

import numpy as np

from code.planner.astar import PathNotFoundError, plan_path
from code.planner.grid import inflate
from code.warehouse.layout import (
    CALLSIGNS,
    CALLSIGNS6,
    _BAY_RGBA,
    _SPLIT6_X,
    _SPLIT6_Y_N,
    _SPLIT6_Y_S,
    _all_pairs_reachable,
    _divider_gaps,
    callsigns_for_layout,
    hero_layout,
    room_of,
    rooms6_layout,
    rooms_layout,
    sample_rooms6_layout,
    validate_rooms_layout,
)
from code.warehouse.occupancy import occupancy_grid

_MIN_CLEAR = 0.5
_ROOM_NAMES = {"loading room", "storage A", "storage B", "back room"}
_DOOR6_MIN = 2.0  # the six-robot spec doorway floor (>= the 1.8 m global minimum)


def _point_rect_distance(px, py, w):
    ox = max(abs(px - w.cx) - w.half_x, 0.0)
    oy = max(abs(py - w.cy) - w.half_y, 0.0)
    return math.hypot(ox, oy)


class TestCallsigns6(unittest.TestCase):
    def test_callsigns6_extends_hero_roster(self) -> None:
        # Additive: the four hero callsigns unchanged, Echo/Foxtrot appended.
        self.assertEqual(CALLSIGNS6[:4], CALLSIGNS)
        self.assertEqual(CALLSIGNS6[4:], ("Echo", "Foxtrot"))
        self.assertEqual(CALLSIGNS, ("Alpha", "Bravo", "Charlie", "Delta"))

    def test_callsigns_for_layout_follows_bays(self) -> None:
        self.assertEqual(callsigns_for_layout(hero_layout()), CALLSIGNS)
        self.assertEqual(callsigns_for_layout(rooms_layout()), CALLSIGNS)
        self.assertEqual(callsigns_for_layout(rooms6_layout()), CALLSIGNS6)

    def test_new_bay_colours_distinct(self) -> None:
        for cs in CALLSIGNS6:
            self.assertIn(cs, _BAY_RGBA)
        rgbas = [_BAY_RGBA[cs] for cs in CALLSIGNS6]
        self.assertEqual(len(rgbas), len(set(rgbas)))  # all six distinct


class TestRooms6Layout(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.layout = rooms6_layout()

    def test_validates(self) -> None:
        validate_rooms_layout(self.layout,
                              divider_lines=(_SPLIT6_Y_S, _SPLIT6_Y_N))

    def test_shell_is_24x16(self) -> None:
        self.assertEqual((self.layout.hall_x, self.layout.hall_y), (24.0, 16.0))

    def test_name(self) -> None:
        self.assertEqual(self.layout.name, "rooms6")

    def test_four_named_rooms(self) -> None:
        self.assertEqual({r.name for r in self.layout.rooms}, _ROOM_NAMES)

    def test_six_bays(self) -> None:
        self.assertEqual(len(self.layout.spawn_poses), 6)
        self.assertEqual(tuple(self.layout.spawn_poses), CALLSIGNS6)
        for cs in CALLSIGNS6:
            bay = next(z for z in self.layout.zones if z.name == f"bay_{cs}")
            self.assertEqual(bay.rgba, _BAY_RGBA[cs])
        # All six spawn in the loading room, facing +y into the hall.
        for _cs, (_x, y, yaw) in self.layout.spawn_poses.items():
            self.assertLess(y, 0.0)
            self.assertAlmostEqual(yaw, math.pi / 2.0)
        spawn_rooms = {room_of(self.layout, (x, y))
                       for (x, y, _y) in self.layout.spawn_poses.values()}
        self.assertEqual(spawn_rooms, {"loading room"})

    def test_wall_composition_extra_shelf_per_room(self) -> None:
        names = [w.name for w in self.layout.walls]
        self.assertEqual(len(names), len(set(names)))
        perim = [n for n in names if n.startswith("wall_") and "div" not in n]
        dividers = [n for n in names if n.startswith("wall_div_")]
        shelves = [n for n in names if n.startswith("shelf_")]
        self.assertEqual(len(perim), 4)
        # Three shelves per storage room (one more than the four-robot layout).
        self.assertEqual(len(shelves), 6)
        self.assertEqual(len([n for n in shelves if n.startswith("shelf_A")]), 3)
        self.assertEqual(len([n for n in shelves if n.startswith("shelf_B")]), 3)
        # Two horizontal dividers (2 doors -> 3 segs each) + solid A|B (1 seg).
        self.assertEqual(len(dividers), 7)

    def test_delivery_pad_in_storage_b(self) -> None:
        d = next(z for z in self.layout.zones if z.name == "delivery")
        self.assertEqual((d.half_x, d.half_y), (1.0, 1.0))
        self.assertEqual(room_of(self.layout, (d.cx, d.cy)), "storage B")

    def test_fourteen_spots_spread(self) -> None:
        spots = self.layout.object_spots
        self.assertEqual(len(spots), 14)
        by_room = {name: 0 for name in _ROOM_NAMES}
        for xy in spots:
            by_room[room_of(self.layout, xy)] += 1
        for name in _ROOM_NAMES:
            self.assertGreaterEqual(by_room[name], 2, f"too few spots in {name}")

    def test_spawn_and_spot_clearance(self) -> None:
        for _cs, (x, y, _yaw) in self.layout.spawn_poses.items():
            for w in self.layout.walls:
                self.assertGreaterEqual(_point_rect_distance(x, y, w), _MIN_CLEAR)
        for (x, y) in self.layout.object_spots:
            for w in self.layout.walls:
                self.assertGreaterEqual(_point_rect_distance(x, y, w), _MIN_CLEAR)

    def test_four_doorways_at_least_2m(self) -> None:
        south = _divider_gaps(self.layout, "h", _SPLIT6_Y_S, -12.0, 12.0)
        north = _divider_gaps(self.layout, "h", _SPLIT6_Y_N, -12.0, 12.0)
        self.assertEqual((len(south), len(north)), (2, 2))
        for gw in south + north:
            self.assertGreaterEqual(gw, _DOOR6_MIN)

    def test_ab_divider_is_solid(self) -> None:
        self.assertEqual(
            _divider_gaps(self.layout, "v", _SPLIT6_X, _SPLIT6_Y_S, _SPLIT6_Y_N), [])


class TestRooms6TileInterior(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.layout = rooms6_layout()
        cls.og = occupancy_grid(cls.layout, resolution=0.1)

    def test_grid_covers_24x16(self) -> None:
        self.assertEqual(self.og.shape, (160, 240))

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
                    and r.cy - r.half_y < y < r.cy + r.half_y)
                gaps += cnt == 0
                multi += cnt > 1
        self.assertEqual(gaps, 0)
        self.assertEqual(multi, 0)


class TestRooms6ReachabilityGate(unittest.TestCase):
    """CRITICAL: every SIX-bay -> object_spot pair routes at 0.40 AND 0.45 m."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.layout = rooms6_layout()
        cls.og = occupancy_grid(cls.layout, resolution=0.1)

    def _assert_all_reachable(self, inflation: float) -> None:
        ig = inflate(self.og, inflation)
        for cs, (sx, sy, _yaw) in self.layout.spawn_poses.items():
            for i, (ox, oy) in enumerate(self.layout.object_spots):
                try:
                    path = plan_path(ig, (sx, sy), (ox, oy), snap_radius_m=0.4)
                except PathNotFoundError as e:  # pragma: no cover
                    self.fail(f"unreachable @{inflation}: bay {cs} -> spot {i} "
                              f"({ox},{oy}): {e}")
                self.assertGreater(len(path), 0)

    def test_reachable_at_040(self) -> None:
        self._assert_all_reachable(0.40)

    def test_reachable_at_045(self) -> None:
        self._assert_all_reachable(0.45)

    def test_object_stamped_gate_passes(self) -> None:
        ok, diag = _all_pairs_reachable(self.layout)
        self.assertTrue(ok, diag)


class TestSampleRooms6Layout(unittest.TestCase):
    def test_deterministic_given_same_seed(self) -> None:
        a = sample_rooms6_layout(np.random.default_rng(3))
        b = sample_rooms6_layout(np.random.default_rng(3))
        self.assertEqual([dataclasses.astuple(w) for w in a.walls],
                         [dataclasses.astuple(w) for w in b.walls])
        self.assertEqual(a.object_spots, b.object_spots)

    def test_valid_reachable_and_structured_across_seeds(self) -> None:
        for seed in range(4):
            layout = sample_rooms6_layout(np.random.default_rng(seed))
            self.assertEqual((layout.hall_x, layout.hall_y), (24.0, 16.0))
            self.assertEqual(tuple(layout.spawn_poses), CALLSIGNS6)
            self.assertEqual(layout.rooms, rooms6_layout().rooms)  # tiling verbatim
            validate_rooms_layout(layout,
                                  divider_lines=(_SPLIT6_Y_S, _SPLIT6_Y_N))
            ok, diag = _all_pairs_reachable(layout)
            self.assertTrue(ok, f"seed {seed} unreachable: {diag}")
            # solid A|B divider + four >= 2.0 m doorways preserved
            self.assertEqual(
                _divider_gaps(layout, "v", _SPLIT6_X, _SPLIT6_Y_S, _SPLIT6_Y_N), [])
            for line in (_SPLIT6_Y_S, _SPLIT6_Y_N):
                for gw in _divider_gaps(layout, "h", line, -12.0, 12.0):
                    self.assertGreaterEqual(gw, _DOOR6_MIN)

    def test_exhaustion_raises(self) -> None:
        with self.assertRaises(RuntimeError):
            sample_rooms6_layout(np.random.default_rng(0), max_attempts=0)


class TestFourRobotLayoutsUnchanged(unittest.TestCase):
    """Regression guard: the four-robot layouts must stay byte-identical."""

    def test_hero_and_rooms_still_four_bays(self) -> None:
        self.assertEqual(tuple(hero_layout().spawn_poses), CALLSIGNS)
        self.assertEqual(tuple(rooms_layout().spawn_poses), CALLSIGNS)

    def test_rooms_shell_and_shelves_unchanged(self) -> None:
        rl = rooms_layout()
        self.assertEqual((rl.hall_x, rl.hall_y), (20.0, 14.0))
        shelves = [w for w in rl.walls if w.name.startswith("shelf_")]
        self.assertEqual(len(shelves), 4)  # still two per storage room


if __name__ == "__main__":
    unittest.main()
