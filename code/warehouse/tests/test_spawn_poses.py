"""Tests for random robot spawn-pose sampling (Demo Set v2).

:func:`~code.warehouse.layout.sample_spawn_poses` must be deterministic per seed,
place every robot in a genuinely free cell (clear of walls/shelves/objects),
respect the minimum pairwise spacing, and only accept poses that can plan to every
room and the delivery pad. Pure geometry/planning — no MuJoCo.
"""

from __future__ import annotations

import math
import unittest

import numpy as np

from code.apps.warehouse_demo.planning import add_object_obstacles
from code.planner.astar import PathNotFoundError, plan_path
from code.planner.grid import inflate
from code.warehouse.layout import (CALLSIGNS, CALLSIGNS6, _GATE_OBJECT_SIZE_M,
                                   _SPAWN_MIN_SPACING_M, hero_layout, room_of,
                                   rooms6_layout, rooms_layout,
                                   sample_rooms_layout, sample_spawn_poses)
from code.warehouse.occupancy import occupancy_grid

_LAYOUTS = [("hero", hero_layout(), CALLSIGNS),
            ("rooms", rooms_layout(), CALLSIGNS),
            ("rooms6", rooms6_layout(), CALLSIGNS6),
            ("sampled", sample_rooms_layout(np.random.default_rng(3)), CALLSIGNS)]


class TestSpawnSampler(unittest.TestCase):
    def test_deterministic_per_seed(self) -> None:
        for _name, lay, cs in _LAYOUTS:
            a = sample_spawn_poses(lay, cs, seed=5)
            b = sample_spawn_poses(lay, cs, seed=5)
            self.assertEqual(a, b)

    def test_seed_changes_poses(self) -> None:
        lay = rooms_layout()
        self.assertNotEqual(sample_spawn_poses(lay, CALLSIGNS, seed=1),
                            sample_spawn_poses(lay, CALLSIGNS, seed=2))

    def test_one_pose_per_callsign(self) -> None:
        for _name, lay, cs in _LAYOUTS:
            poses = sample_spawn_poses(lay, cs, seed=7)
            self.assertEqual(list(poses.keys()), list(cs))

    def test_minimum_spacing(self) -> None:
        for _name, lay, cs in _LAYOUTS:
            pts = [(x, y) for (x, y, _yaw) in sample_spawn_poses(lay, cs, seed=9).values()]
            for i in range(len(pts)):
                for j in range(i + 1, len(pts)):
                    d = math.hypot(pts[i][0] - pts[j][0], pts[i][1] - pts[j][1])
                    self.assertGreaterEqual(d + 1e-9, _SPAWN_MIN_SPACING_M)

    def test_poses_are_in_free_cells(self) -> None:
        # Every spawn is free at the deployed 0.40 m clearance on the object-stamped grid.
        for _name, lay, cs in _LAYOUTS:
            gate_objects = [{"x": float(x), "y": float(y), "size": _GATE_OBJECT_SIZE_M}
                            for (x, y) in lay.object_spots]
            og = occupancy_grid(lay, 0.1)
            stamped = add_object_obstacles(og, gate_objects, None, 0.0)
            grid = inflate(stamped, 0.40)
            for cs_name, (x, y, _yaw) in sample_spawn_poses(lay, cs, seed=11).items():
                self.assertTrue(grid.is_free((x, y)),
                                f"{_name}/{cs_name} spawned in an occupied cell")

    def test_reachability_to_every_room_and_pad(self) -> None:
        lay = rooms_layout()
        poses = sample_spawn_poses(lay, CALLSIGNS, seed=13)
        og = occupancy_grid(lay, 0.1)
        grid = inflate(og, 0.40)
        # A representative point inside each room + the delivery pad.
        targets = [(r.cx, r.cy) for r in lay.rooms]
        for z in lay.zones:
            if z.name == "delivery":
                targets.append((z.cx, z.cy))
        for cs, (x, y, _yaw) in poses.items():
            for gx, gy in targets:
                try:
                    path = plan_path(grid, (x, y), (gx, gy), snap_radius_m=0.6)
                except PathNotFoundError:
                    self.fail(f"{cs} at ({x:.1f},{y:.1f}) cannot reach ({gx:.1f},{gy:.1f})")
                self.assertTrue(path)

    def test_spawns_land_in_multiple_rooms(self) -> None:
        # Random spawns should genuinely scatter, not cluster in one room.
        lay = rooms_layout()
        pts = [(x, y) for (x, y, _y) in sample_spawn_poses(lay, CALLSIGNS, seed=4).values()]
        rooms = {room_of(lay, p) for p in pts}
        self.assertGreaterEqual(len(rooms), 2)

    def test_explicit_objects_are_avoided(self) -> None:
        # A spawn never lands on an explicitly-placed object.
        lay = rooms_layout()
        objs = [{"color_name": "red", "shape_name": "cube", "size": 0.24,
                 "x": float(x), "y": float(y)} for (x, y) in lay.object_spots]
        poses = sample_spawn_poses(lay, CALLSIGNS, seed=6, objects=objs)
        for (x, y, _yaw) in poses.values():
            for o in objs:
                self.assertGreater(math.hypot(x - o["x"], y - o["y"]), 0.5)

    def test_too_many_robots_raises(self) -> None:
        # A tiny min-spacing budget that cannot fit the roster fails deterministically.
        with self.assertRaises(RuntimeError):
            sample_spawn_poses(hero_layout(), CALLSIGNS, seed=0, min_spacing_m=100.0)


if __name__ == "__main__":
    unittest.main()
