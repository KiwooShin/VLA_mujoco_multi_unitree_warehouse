"""Unit tests for region partitioning + serpentine patrols (code.fleet.search)."""

from __future__ import annotations

import unittest

import numpy as np

from code.fleet import search as S
from code.warehouse.arena import warehouse_scene_cfg
from code.warehouse.layout import hero_layout


def _cfg():
    return warehouse_scene_cfg(hero_layout(), rng=np.random.default_rng(0))


class TestRegions(unittest.TestCase):
    def test_region_bounds_partition_thirds(self) -> None:
        hall_y = 12.0
        s = S.region_bounds("south", hall_y)
        m = S.region_bounds("middle", hall_y)
        n = S.region_bounds("north", hall_y)
        self.assertEqual(s, (-6.0, -2.0))
        self.assertEqual(m, (-2.0, 2.0))
        self.assertEqual(n, (2.0, 6.0))
        # Bands tile the hall with no gaps or overlaps.
        self.assertAlmostEqual(s[1], m[0])
        self.assertAlmostEqual(m[1], n[0])

    def test_unknown_region_raises(self) -> None:
        with self.assertRaises(ValueError):
            S.region_bounds("east", 12.0)

    def test_region_centroid_in_band(self) -> None:
        for r in S.SEARCH_REGIONS:
            cx, cy = S.region_centroid(r, 16.0, 12.0)
            lo, hi = S.region_bounds(r, 12.0)
            self.assertTrue(lo <= cy <= hi)
            self.assertEqual(cx, 0.0)


class TestPatrol(unittest.TestCase):
    def test_waypoints_nonempty_and_free(self) -> None:
        from code.apps.warehouse_demo.planning import build_inflated_grid

        cfg = _cfg()
        grid = build_inflated_grid(cfg, 0.10, 0.40)
        for r in S.SEARCH_REGIONS:
            wp = S.patrol_waypoints(cfg, r)
            self.assertGreater(len(wp), 3, f"region {r} too sparse")
            for x, y in wp:
                self.assertTrue(grid.is_free((x, y)),
                                f"waypoint {(x, y)} not free")
                lo, hi = S.region_bounds(r, cfg["hall_y"])
                self.assertTrue(lo - 1.0 <= y <= hi + 1.0)

    def test_patrol_deterministic(self) -> None:
        cfg = _cfg()
        a = S.patrol_waypoints(cfg, "north")
        b = S.patrol_waypoints(cfg, "north")
        self.assertEqual(a, b)

    def test_nearest_neighbour_starts_near_entry(self) -> None:
        cfg = _cfg()
        start = (-7.0, 3.0)
        wp = S.patrol_waypoints(cfg, "north", start_xy=start)
        first_d = (wp[0][0] - start[0]) ** 2 + (wp[0][1] - start[1]) ** 2
        # The first waypoint is the closest of the whole set to the entry point.
        for x, y in wp[1:]:
            self.assertLessEqual(first_d,
                                 (x - start[0]) ** 2 + (y - start[1]) ** 2 + 1e-9)

    def test_free_centroid_within_hall(self) -> None:
        cfg = _cfg()
        for r in S.SEARCH_REGIONS:
            cx, cy = S.free_centroid(cfg, r)
            self.assertTrue(-8.0 < cx < 8.0 and -6.0 < cy < 6.0)


class _FakeUnit:
    """A minimal RobotUnit stand-in for the search controller."""

    def __init__(self, xy=(0.0, 0.0)) -> None:
        self._xy = xy
        self.goals = []
        self.done = False
        self.active = True
        self.halted = False

    @property
    def xy(self):
        return self._xy

    def assign_goal(self, xy) -> bool:
        self.goals.append(xy)
        return True

    def halt(self) -> None:
        self.halted = True
        self.active = False


class TestSearchController(unittest.TestCase):
    def test_start_assigns_first_waypoint(self) -> None:
        cfg = _cfg()
        unit = _FakeUnit((-7.0, 3.0))
        ctrl = S.SearchController(unit)
        self.assertTrue(ctrl.start(cfg, "north"))
        self.assertEqual(len(unit.goals), 1)
        self.assertEqual(ctrl.region, "north")
        self.assertTrue(ctrl.active)

    def test_tick_advances_on_arrival(self) -> None:
        cfg = _cfg()
        unit = _FakeUnit((-7.0, 3.0))
        ctrl = S.SearchController(unit)
        ctrl.start(cfg, "north")
        unit.done = True  # arrived at the first waypoint
        ctrl.tick()
        self.assertGreaterEqual(len(unit.goals), 2)

    def test_stop_halts_unit(self) -> None:
        cfg = _cfg()
        unit = _FakeUnit((-7.0, 3.0))
        ctrl = S.SearchController(unit)
        ctrl.start(cfg, "north")
        ctrl.stop()
        self.assertFalse(ctrl.active)
        self.assertTrue(unit.halted)


if __name__ == "__main__":
    unittest.main()
