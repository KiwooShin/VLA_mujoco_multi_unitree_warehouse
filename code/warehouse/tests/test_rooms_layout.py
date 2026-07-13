"""Unit tests for the multi-room F6 layout (rooms_layout, Room, room_of).

Covers the layout invariants (counts, doorway widths, clearances, tiling), the
``room_of`` point->name resolver, and the CRITICAL plan-reachability gate: every
bay->object_spot pair must be routable at 0.40 AND 0.45 m planner inflation (the
hero NE alcove sealed at 0.50 m; the open doorways are the same risk class).
"""

import dataclasses
import math
import unittest

import numpy as np

from code.planner.astar import PathNotFoundError, plan_path
from code.planner.grid import inflate
from code.warehouse.layout import (
    CALLSIGNS,
    Room,
    WarehouseLayout,
    _BAY_RGBA,
    _MIN_DOORWAY_M,
    _RM_SPOTS_MAX,
    _RM_SPOTS_MIN,
    _all_pairs_reachable,
    _divider_gaps,
    hero_layout,
    room_of,
    rooms_layout,
    sample_rooms_layout,
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


class TestSampleRoomsLayout(unittest.TestCase):
    """The randomized four-room family: determinism, validity, reachability."""

    def test_deterministic_given_same_seed(self) -> None:
        a = sample_rooms_layout(np.random.default_rng(7))
        b = sample_rooms_layout(np.random.default_rng(7))
        self.assertEqual([dataclasses.astuple(w) for w in a.walls],
                         [dataclasses.astuple(w) for w in b.walls])
        self.assertEqual(a.object_spots, b.object_spots)
        self.assertEqual([dataclasses.astuple(z) for z in a.zones],
                         [dataclasses.astuple(z) for z in b.zones])

    def test_valid_and_reachable_across_seeds(self) -> None:
        for seed in range(3):
            layout = sample_rooms_layout(np.random.default_rng(seed))
            validate_rooms_layout(layout)                 # geometry gate
            ok, diag = _all_pairs_reachable(layout)       # sec-5b plan gate
            self.assertTrue(ok, f"seed {seed} unreachable: {diag}")

    def test_structure_preserved(self) -> None:
        # Same shell, same four rooms/tiling, same bays, solid A|B divider.
        layout = sample_rooms_layout(np.random.default_rng(1))
        self.assertEqual((layout.hall_x, layout.hall_y), (20.0, 14.0))
        self.assertEqual({r.name for r in layout.rooms}, _ROOM_NAMES)
        self.assertEqual(layout.rooms, rooms_layout().rooms)  # tiling verbatim
        self.assertEqual(set(layout.spawn_poses), set(CALLSIGNS))
        self.assertEqual(_divider_gaps(layout, "v", 0.0, -3.5, 3.5), [])

    def test_doorways_wide_enough(self) -> None:
        layout = sample_rooms_layout(np.random.default_rng(2))
        south = _divider_gaps(layout, "h", -3.5, -10.0, 10.0)
        north = _divider_gaps(layout, "h", 3.5, -10.0, 10.0)
        self.assertEqual((len(south), len(north)), (2, 2))
        for gw in south + north:
            self.assertGreaterEqual(gw, _MIN_DOORWAY_M)

    def test_per_room_spot_counts(self) -> None:
        # 2-3 object spots in every room (loading/A/B/back), all >=0.5 m clear.
        for seed in range(3):
            layout = sample_rooms_layout(np.random.default_rng(seed))
            by_room = {name: 0 for name in _ROOM_NAMES}
            for xy in layout.object_spots:
                by_room[room_of(layout, xy)] += 1
            for name, cnt in by_room.items():
                self.assertGreaterEqual(cnt, _RM_SPOTS_MIN, (seed, name))
                self.assertLessEqual(cnt, _RM_SPOTS_MAX, (seed, name))

    def test_delivery_pad_in_storage_b(self) -> None:
        layout = sample_rooms_layout(np.random.default_rng(0))
        d = next(z for z in layout.zones if z.name == "delivery")
        self.assertEqual((d.half_x, d.half_y), (1.0, 1.0))
        self.assertEqual(room_of(layout, (d.cx, d.cy)), "storage B")

    def test_reachability_gate_actually_gates(self) -> None:
        # With the gate off, geometry-valid draws may include unreachable ones;
        # with it on, the returned layout must pass _all_pairs_reachable.
        gated = sample_rooms_layout(np.random.default_rng(5), enforce_reachable=True)
        self.assertTrue(_all_pairs_reachable(gated)[0])

    def test_exhaustion_raises_with_diagnostics(self) -> None:
        with self.assertRaises(RuntimeError) as ctx:
            sample_rooms_layout(np.random.default_rng(0), max_attempts=0)
        self.assertIn("no valid+reachable layout", str(ctx.exception))


class TestAllPairsReachable(unittest.TestCase):
    """The sec-5b reachability helper used by both sampler families."""

    def test_fixed_layouts_pass(self) -> None:
        self.assertTrue(_all_pairs_reachable(rooms_layout())[0])
        self.assertTrue(_all_pairs_reachable(hero_layout())[0])

    def test_detects_unreachable_goal(self) -> None:
        # A spot walled off inside a solid interior block is unroutable.
        base = rooms_layout()
        boxed = dataclasses.replace(base, object_spots=[(0.0, 0.0)])
        # Wrap the spot in a small solid block so A* cannot snap out to it.
        from code.warehouse.layout import WallSpec
        wall = WallSpec(0.0, 0.0, 0.6, 0.6, height=2.0, name="wall_trap")
        boxed = dataclasses.replace(boxed, walls=list(base.walls) + [wall])
        ok, diag = _all_pairs_reachable(boxed, include_delivery=False)
        self.assertFalse(ok)
        self.assertIn("unreachable", diag)

    def test_object_in_corridor_blocks_route(self) -> None:
        # Finding 6: the gate stamps objects, so an OBJECT (not a wall) that seals
        # the only corridor to a goal spot is now caught — a walls-only gate would
        # certify this layout as routable.
        from code.warehouse.layout import WallSpec
        # A vertical corridor (two tall walls) with a ~2 m gap around y=0.
        walls = [WallSpec(0.0, 2.5, 0.15, 1.5, height=2.0, name="corr_top"),
                 WallSpec(0.0, -2.5, 0.15, 1.5, height=2.0, name="corr_bot")]
        # spot0 sits IN the gap; spot1 is on the far side of the corridor.
        layout = WarehouseLayout(
            hall_x=10.0, hall_y=6.0, walls=walls,
            spawn_poses={"bay": (-4.0, 0.0, 0.0)},
            object_spots=[(0.0, 0.0), (4.0, 0.0)], name="corridor_probe")
        # With the corridor object stamped, bay -> spot1 must thread the gap the
        # object now seals -> rejected.
        ok, diag = _all_pairs_reachable(layout, include_delivery=False)
        self.assertFalse(ok)
        self.assertIn("spot1", diag)
        self.assertIn("object-stamped", diag)
        # Remove the corridor object (leave only the far goal) -> the gap is clear
        # again and the same bay -> spot1 route certifies. This is the empty-grid
        # route a walls-only gate would have (wrongly) accepted with the object.
        open_layout = dataclasses.replace(layout, object_spots=[(4.0, 0.0)])
        self.assertTrue(
            _all_pairs_reachable(open_layout, include_delivery=False)[0])

    def test_committed_sampler_seeds_pass_object_stamped_gate(self) -> None:
        # Finding 6: the 16 committed sampler seeds (rooms 0-7, hero 0-7) that the
        # release evals ran on must still certify under the object-stamped gate
        # (verified byte-identical layouts) — a fail here is a FINDING, not a
        # silent reseed.
        from code.warehouse.layout import sample_layout
        for seed in range(8):
            rl = sample_rooms_layout(np.random.default_rng(seed))
            ok, diag = _all_pairs_reachable(rl)
            self.assertTrue(ok, f"rooms seed {seed}: {diag}")
            hl = sample_layout(np.random.default_rng(seed))
            ok, diag = _all_pairs_reachable(hl)
            self.assertTrue(ok, f"hero seed {seed}: {diag}")


if __name__ == "__main__":
    unittest.main()
