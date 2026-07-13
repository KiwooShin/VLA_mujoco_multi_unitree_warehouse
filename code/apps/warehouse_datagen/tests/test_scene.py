"""Pure-logic tests for the warehouse episode-plan sampler (no simulator).

Covers: determinism given seed, plan structure per mode, free-space validity of
spawns/routes, and command-distribution sanity (primitive fraction, route
diversity), without compiling any MuJoCo model.
"""

from __future__ import annotations

import math
import unittest

import numpy as np

from code.apps.warehouse_datagen.scene import (
    GRID_RES, INFLATE_RADIUS, MIN_ROUTE_LEN_M, EpisodePlan,
    free_cells_world, los_free, sample_episode_plan,
)
from code.apps.warehouse_demo.planning import build_inflated_grid
from code.planner.astar import path_length


class SamplePlanTest(unittest.TestCase):
    def test_deterministic_given_seed(self) -> None:
        p1 = sample_episode_plan(7, 3)
        p2 = sample_episode_plan(7, 3)
        self.assertEqual(p1.mode, p2.mode)
        self.assertEqual(p1.spawn_xy, p2.spawn_xy)
        self.assertAlmostEqual(p1.spawn_yaw, p2.spawn_yaw, places=9)
        self.assertEqual(p1.goal_xy, p2.goal_xy)
        self.assertEqual(p1.instruction, p2.instruction)
        self.assertEqual(p1.path, p2.path)
        self.assertEqual(p1.fixed_target, p2.fixed_target)

    def test_different_indices_differ(self) -> None:
        seen = {sample_episode_plan(7, i).spawn_xy for i in range(6)}
        self.assertGreater(len(seen), 1)

    def test_plan_structure_and_free_space(self) -> None:
        for i in range(12):
            plan = sample_episode_plan(11, i, primitive_frac=0.3)
            self.assertIsInstance(plan, EpisodePlan)
            grid = build_inflated_grid(plan.scene_cfg, GRID_RES, INFLATE_RADIUS)
            # spawn must be a free cell
            self.assertTrue(grid.is_free(plan.spawn_xy),
                            f"ep{i} spawn {plan.spawn_xy} not free")
            self.assertTrue(-math.pi <= plan.spawn_yaw <= math.pi)
            if plan.mode == "route":
                self.assertIsNotNone(plan.path)
                self.assertGreaterEqual(len(plan.path), 2)
                self.assertGreaterEqual(path_length(plan.path), MIN_ROUTE_LEN_M - 1e-6)
                self.assertIsNone(plan.fixed_target)
            else:
                self.assertIsNone(plan.path)
                self.assertIsNotNone(plan.fixed_target)
                # primitive target must have a clear line of sight from spawn
                self.assertTrue(los_free(grid, plan.spawn_xy, plan.fixed_target))

    def test_primitive_fraction_bounds(self) -> None:
        modes = [sample_episode_plan(3, i, primitive_frac=1.0).mode for i in range(8)]
        self.assertTrue(all(m == "primitive" for m in modes))
        modes0 = [sample_episode_plan(3, i, primitive_frac=0.0).mode for i in range(8)]
        # With frac=0 nearly all are routes (rare fallback to primitive allowed).
        self.assertGreaterEqual(sum(m == "route" for m in modes0), 6)

    def test_free_cells_are_free(self) -> None:
        plan = sample_episode_plan(1, 0)
        grid = build_inflated_grid(plan.scene_cfg, GRID_RES, INFLATE_RADIUS)
        cells = free_cells_world(grid)
        self.assertGreater(len(cells), 100)
        for c in cells[:: max(1, len(cells) // 20)]:
            self.assertTrue(grid.is_free((float(c[0]), float(c[1]))))


if __name__ == "__main__":
    unittest.main()
