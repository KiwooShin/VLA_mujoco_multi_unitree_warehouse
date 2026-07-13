"""Unit tests for code.apps.warehouse_demo.nav_rollout pure logic.

Covers command shaping + the OOD turn guard, termination classification, the
point/wall clearance geometry, NavResult serialization, scene_cfg -> layout
reconstruction, and the object-obstacle planning grid. No simulator is stepped
(see test_smoke.py for the headless sim smoke).
"""

from __future__ import annotations

import json
import math
import unittest

import numpy as np

from code.apps.warehouse_demo.nav_rollout import (
    NavParams,
    NavResult,
    classify_termination,
    shape_command,
)
from code.apps.warehouse_demo.planning import (
    add_object_obstacles,
    build_inflated_grid,
    layout_from_scene_cfg,
    min_wall_clearance,
    point_obb_distance,
)
from code.warehouse.arena import warehouse_scene_cfg
from code.warehouse.layout import hero_layout
from code.warehouse.occupancy import occupancy_grid


class TestPointObbDistance(unittest.TestCase):
    def test_inside_is_zero(self) -> None:
        wall = {"cx": 0.0, "cy": 0.0, "half_x": 1.0, "half_y": 0.5, "yaw": 0.0}
        self.assertEqual(point_obb_distance(0.2, 0.1, wall), 0.0)

    def test_axis_aligned_distance(self) -> None:
        wall = {"cx": 0.0, "cy": 0.0, "half_x": 1.0, "half_y": 0.5, "yaw": 0.0}
        # 3 m to the right of a box whose right face is at x=1 -> distance 2.
        self.assertAlmostEqual(point_obb_distance(3.0, 0.0, wall), 2.0, places=6)
        # Diagonally off the corner (2,1.5): corner at (1,0.5) -> hypot(1,1).
        self.assertAlmostEqual(point_obb_distance(2.0, 1.5, wall),
                               math.hypot(1.0, 1.0), places=6)

    def test_yawed_rect(self) -> None:
        wall = {"cx": 0.0, "cy": 0.0, "half_x": 1.0, "half_y": 0.1,
                "yaw": math.pi / 2.0}
        # Rotated 90 deg, so the long axis is along y; point 3 m along +y.
        self.assertAlmostEqual(point_obb_distance(0.0, 3.0, wall), 2.0, places=6)

    def test_min_wall_clearance_picks_nearest(self) -> None:
        walls = [
            {"cx": 5.0, "cy": 0.0, "half_x": 0.1, "half_y": 5.0, "yaw": 0.0},
            {"cx": 0.0, "cy": 3.0, "half_x": 5.0, "half_y": 0.1, "yaw": 0.0},
        ]
        # At origin: 4.9 to the vertical wall, 2.9 to the horizontal -> 2.9.
        self.assertAlmostEqual(min_wall_clearance(0.0, 0.0, walls), 2.9, places=6)

    def test_min_wall_clearance_empty(self) -> None:
        self.assertEqual(min_wall_clearance(0.0, 0.0, []), float("inf"))


class TestClassifyTermination(unittest.TestCase):
    def test_continue(self) -> None:
        self.assertEqual(classify_termination(False, 0.75, 5, 100), (False, ""))

    def test_fall(self) -> None:
        self.assertEqual(classify_termination(False, 0.4, 5, 100), (True, "fall"))

    def test_fall_precedes_success(self) -> None:
        # A robot that reaches the goal but has already collapsed is a fall.
        self.assertEqual(classify_termination(True, 0.4, 5, 100), (True, "fall"))

    def test_success(self) -> None:
        self.assertEqual(classify_termination(True, 0.75, 5, 100),
                         (True, "success"))

    def test_timeout_on_last_step(self) -> None:
        self.assertEqual(classify_termination(False, 0.75, 99, 100),
                         (True, "timeout"))


class TestShapeCommand(unittest.TestCase):
    def setUp(self) -> None:
        self.params = NavParams()

    def test_forward_when_aligned(self) -> None:
        cmd, dist, yaw_err, is_turn = shape_command(
            (0.0, 0.0), 0.0, (3.0, 0.0), self.params, turn_run=0)
        self.assertGreater(cmd[0], 0.0)
        self.assertFalse(is_turn)
        self.assertAlmostEqual(dist, 3.0, places=5)

    def test_turn_in_place_when_target_behind(self) -> None:
        cmd, _, _, is_turn = shape_command(
            (0.0, 0.0), 0.0, (-3.0, 0.0), self.params, turn_run=0)
        self.assertAlmostEqual(cmd[0], 0.0, places=6)
        self.assertTrue(is_turn)
        self.assertGreater(abs(cmd[2]), 0.0)

    def test_turn_guard_injects_forward_creep(self) -> None:
        # Same behind-target turn, but the guard trips after a long turn run.
        cmd, _, _, is_turn = shape_command(
            (0.0, 0.0), 0.0, (-3.0, 0.0), self.params,
            turn_run=self.params.max_turn_run)
        self.assertTrue(is_turn)
        self.assertAlmostEqual(cmd[0], self.params.turn_break_vx, places=6)

    def test_command_speed_capped(self) -> None:
        p = NavParams(max_vx=0.4, max_wz=0.5)
        cmd, _, _, _ = shape_command((0.0, 0.0), 0.0, (10.0, 0.0), p, turn_run=0)
        self.assertLessEqual(cmd[0], 0.4 + 1e-6)
        self.assertLessEqual(abs(cmd[2]), 0.5 + 1e-6)


class TestNavResultSerialization(unittest.TestCase):
    def test_to_dict_is_json_serializable(self) -> None:
        res = NavResult(
            success=True, outcome="success", steps=612,
            path_length_planned=8.2, path_length_walked=8.4,
            path_efficiency=0.976, min_wall_clearance=0.31, fell=False,
            wall_collision=False, time_s=1.2, goal_xy=(6.5, 4.7),
            max_turn_run=40, planned_path=[(-5.0, -5.0), (6.5, 4.7)],
        )
        d = res.to_dict()
        s = json.dumps(d)  # must not raise
        back = json.loads(s)
        self.assertEqual(back["outcome"], "success")
        self.assertEqual(back["goal_xy"], [6.5, 4.7])
        self.assertEqual(back["planned_path"][0], [-5.0, -5.0])
        self.assertTrue(back["success"])


class TestLayoutReconstruction(unittest.TestCase):
    def test_recovers_hall_dims(self) -> None:
        cfg = warehouse_scene_cfg(hero_layout(), rng=np.random.default_rng(0))
        layout = layout_from_scene_cfg(cfg)
        self.assertAlmostEqual(layout.hall_x, 16.0, places=6)
        self.assertAlmostEqual(layout.hall_y, 12.0, places=6)
        self.assertEqual(len(layout.walls), len(cfg["walls"]))


class TestObjectObstacles(unittest.TestCase):
    def setUp(self) -> None:
        self.layout = hero_layout()
        self.cfg = warehouse_scene_cfg(self.layout, rng=np.random.default_rng(0))
        self.og = occupancy_grid(self.layout, resolution=0.1)

    def test_non_goal_object_marked_goal_excluded(self) -> None:
        objects = self.cfg["objects"]
        goal = (-5.0, 0.5)  # west-lane spot: object here should stay free
        og2 = add_object_obstacles(self.og, objects, goal, exclude_radius=0.5)
        # A different object at (-1.5, 0.0) must now be occupied.
        iy, ix = og2.world_to_cell((-1.5, 0.0))
        self.assertTrue(bool(og2.grid[iy, ix]))
        # The goal object's own cell stays free (approachable) in open lane.
        giy, gix = og2.world_to_cell(goal)
        self.assertFalse(bool(og2.grid[giy, gix]))

    def test_build_inflated_grid_shape_preserved(self) -> None:
        grid = build_inflated_grid(self.cfg, 0.1, NavParams().inflate_radius,
                                   goal_xy=(-5.0, 0.5))
        self.assertEqual(grid.grid.shape, self.og.grid.shape)
        self.assertAlmostEqual(grid.resolution, 0.1, places=6)
        # Inflation strictly adds occupancy vs. the walls-only grid.
        self.assertGreaterEqual(int(grid.grid.sum()), int(self.og.grid.sum()))


if __name__ == "__main__":
    unittest.main()
