"""Unit tests for code.warehouse.layout (geometry spec + validity invariants)."""

import dataclasses
import math
import unittest

import numpy as np

from code.warehouse.layout import (
    CALLSIGNS,
    Zone,
    WallSpec,
    WarehouseLayout,
    _partition_walls,
    _perimeter_walls,
    _point_rect_distance,
    _obb_overlap,
    _shelf_walls,
    _shelf_rows,
    _zones_and_spawns,
    hero_layout,
    sample_layout,
    validate_layout,
)

_MIN_AISLE = 2.2
_MIN_CLEAR = 0.5


class TestDataclasses(unittest.TestCase):
    def test_wallspec_is_frozen(self) -> None:
        w = WallSpec(0.0, 0.0, 1.0, 1.0, name="w")
        with self.assertRaises(dataclasses.FrozenInstanceError):
            w.cx = 5.0  # type: ignore[misc]

    def test_wallspec_defaults(self) -> None:
        w = WallSpec(0.0, 0.0, 1.0, 1.0)
        self.assertEqual(w.yaw, 0.0)
        self.assertEqual(w.height, 2.5)
        self.assertEqual(len(w.rgba), 4)

    def test_zone_is_frozen(self) -> None:
        z = Zone("z", 0.0, 0.0, 1.0, 1.0)
        with self.assertRaises(dataclasses.FrozenInstanceError):
            z.cx = 3.0  # type: ignore[misc]

    def test_warehouselayout_default_extents(self) -> None:
        wl = WarehouseLayout()
        self.assertEqual(wl.hall_x, 16.0)
        self.assertEqual(wl.hall_y, 12.0)


class TestGeometryHelpers(unittest.TestCase):
    def test_point_inside_rect_zero_distance(self) -> None:
        w = WallSpec(0.0, 0.0, 1.0, 0.5)
        self.assertEqual(_point_rect_distance(0.0, 0.0, w), 0.0)

    def test_point_outside_axis_aligned(self) -> None:
        w = WallSpec(0.0, 0.0, 1.0, 0.5)
        self.assertAlmostEqual(_point_rect_distance(2.0, 0.0, w), 1.0)
        self.assertAlmostEqual(_point_rect_distance(0.0, 1.5, w), 1.0)

    def test_point_distance_respects_yaw(self) -> None:
        # A thin wall rotated 90 deg: its long axis now runs along world y.
        w = WallSpec(0.0, 0.0, 2.0, 0.1, yaw=math.pi / 2.0)
        # A point 3 m along world y lies just past the (rotated) long half-axis.
        self.assertAlmostEqual(_point_rect_distance(0.0, 3.0, w), 1.0, places=6)
        # A point 3 m along world x is now off the short axis.
        self.assertAlmostEqual(_point_rect_distance(3.0, 0.0, w), 2.9, places=6)

    def test_obb_overlap_true_and_false(self) -> None:
        w = WallSpec(0.0, 0.0, 1.0, 1.0)
        self.assertTrue(_obb_overlap(w, 0.5, 0.5, 1.0, 1.0))
        self.assertFalse(_obb_overlap(w, 5.0, 5.0, 1.0, 1.0))


class TestHeroLayout(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.layout = hero_layout()

    def test_hero_validates(self) -> None:
        validate_layout(self.layout)  # must not raise

    def test_hero_name(self) -> None:
        self.assertEqual(self.layout.name, "hero")

    def test_wall_counts(self) -> None:
        names = [w.name for w in self.layout.walls]
        perim = [n for n in names if n.startswith("wall_")]
        shelves = [n for n in names if n.startswith("shelf_")]
        parts = [n for n in names if n.startswith("part_")]
        self.assertEqual(len(perim), 4)
        self.assertEqual(len(shelves), 4)   # 2 rows x 2 blocks
        self.assertEqual(len(parts), 3)     # NE L (2) + SW (1)
        self.assertEqual(len(self.layout.walls), 11)

    def test_wall_names_unique(self) -> None:
        names = [w.name for w in self.layout.walls]
        self.assertEqual(len(names), len(set(names)))

    def test_two_shelf_rows(self) -> None:
        rows = _shelf_rows(self.layout)
        self.assertEqual(len(rows), 2)
        for row in rows:
            self.assertEqual(len(row), 2)

    def test_three_aisles_wide_enough(self) -> None:
        rows = _shelf_rows(self.layout)
        faces = sorted((r[0].cy - r[0].half_y, r[0].cy + r[0].half_y) for r in rows)
        inner_n = self.layout.hall_y / 2.0 - 0.1
        inner_s = -self.layout.hall_y / 2.0 + 0.1
        north = inner_n - faces[1][1]
        middle = faces[1][0] - faces[0][1]
        south = faces[0][0] - inner_s
        for w in (north, middle, south):
            self.assertGreaterEqual(w, _MIN_AISLE)

    def test_mid_row_crossover_gap(self) -> None:
        # The west/east blocks of a row leave a ~1.4 m gap centred on x=0.
        row = _shelf_rows(self.layout)[0]
        row_sorted = sorted(row, key=lambda w: w.cx)
        west, east = row_sorted
        gap = (east.cx - east.half_x) - (west.cx + west.half_x)
        self.assertAlmostEqual(gap, 1.4, places=6)

    def test_four_spawns_named_by_callsign(self) -> None:
        self.assertEqual(set(self.layout.spawn_poses), set(CALLSIGNS))

    def test_spawns_face_into_hall(self) -> None:
        for _cs, (_x, y, yaw) in self.layout.spawn_poses.items():
            self.assertLess(y, 0.0)  # south wall bays
            self.assertAlmostEqual(yaw, math.pi / 2.0)  # facing +y (into hall)

    def test_spawn_clearance_from_walls(self) -> None:
        for _cs, (x, y, _yaw) in self.layout.spawn_poses.items():
            for w in self.layout.walls:
                self.assertGreaterEqual(_point_rect_distance(x, y, w), _MIN_CLEAR)

    def test_eight_object_spots_clear_of_walls(self) -> None:
        self.assertEqual(len(self.layout.object_spots), 8)
        for (x, y) in self.layout.object_spots:
            for w in self.layout.walls:
                self.assertGreaterEqual(_point_rect_distance(x, y, w), _MIN_CLEAR)

    def test_zones_delivery_and_bays(self) -> None:
        names = {z.name for z in self.layout.zones}
        self.assertIn("delivery", names)
        for cs in CALLSIGNS:
            self.assertIn(f"bay_{cs}", names)
        self.assertEqual(len(self.layout.zones), 5)


class TestValidation(unittest.TestCase):
    def _base_walls(self):
        walls = _perimeter_walls(8.0, 6.0)
        walls += _shelf_walls(gap=1.4, row_cy=2.0, block_len=1.55)
        walls += _partition_walls(alcove_x=5.0, sw_cy=-3.25)
        return walls

    def test_rejects_spot_too_close_to_wall(self) -> None:
        zones, spawns = _zones_and_spawns((5.8, -1.0))
        bad = WarehouseLayout(
            walls=self._base_walls(), zones=zones, spawn_poses=spawns,
            object_spots=[(7.6, 0.0)],  # 0.3 m from east wall inner face
        )
        with self.assertRaises(ValueError):
            validate_layout(bad)

    def test_rejects_narrow_middle_aisle(self) -> None:
        zones, spawns = _zones_and_spawns((5.8, -1.0))
        walls = _perimeter_walls(8.0, 6.0)
        walls += _shelf_walls(gap=1.4, row_cy=1.0, block_len=1.55)  # rows too close
        walls += _partition_walls(alcove_x=5.0, sw_cy=-3.25)
        bad = WarehouseLayout(walls=walls, zones=zones, spawn_poses=spawns,
                              object_spots=[(-1.5, 0.0)])
        with self.assertRaises(ValueError):
            validate_layout(bad)

    def test_rejects_spot_inside_delivery(self) -> None:
        zones, spawns = _zones_and_spawns((5.8, -1.0))
        bad = WarehouseLayout(
            walls=self._base_walls(), zones=zones, spawn_poses=spawns,
            object_spots=[(5.8, -1.0)],  # centre of the delivery pad
        )
        with self.assertRaises(ValueError):
            validate_layout(bad)

    def test_rejects_overlapping_spots(self) -> None:
        zones, spawns = _zones_and_spawns((5.8, -1.0))
        bad = WarehouseLayout(
            walls=self._base_walls(), zones=zones, spawn_poses=spawns,
            object_spots=[(0.0, 0.0), (0.2, 0.0)],  # < 0.8 m apart
        )
        with self.assertRaises(ValueError):
            validate_layout(bad)


class TestSampleLayout(unittest.TestCase):
    def test_deterministic_given_same_seed(self) -> None:
        a = sample_layout(np.random.default_rng(11))
        b = sample_layout(np.random.default_rng(11))
        self.assertEqual([dataclasses.astuple(w) for w in a.walls],
                         [dataclasses.astuple(w) for w in b.walls])
        self.assertEqual(a.object_spots, b.object_spots)

    def test_valid_across_seeds(self) -> None:
        for seed in range(12):
            layout = sample_layout(np.random.default_rng(seed))
            validate_layout(layout)  # must not raise
            self.assertEqual(len(layout.object_spots), 8)
            self.assertEqual(set(layout.spawn_poses), set(CALLSIGNS))

    def test_varies_from_hero(self) -> None:
        # At least one seed should perturb the geometry away from the hero spots.
        hero = hero_layout()
        differs = any(
            sample_layout(np.random.default_rng(s)).object_spots != hero.object_spots
            for s in range(5)
        )
        self.assertTrue(differs)


if __name__ == "__main__":
    unittest.main()
