"""Headless sim smoke test for run_nav_rollout (~100 control steps).

Skips when the WBC ONNX / G1 assets or the MuJoCo runtime are unavailable
(fresh clone without ``third_party/`` symlinks), mirroring the baseline
renderer-smoke convention in code/warehouse/tests/test_arena.py.
"""

from __future__ import annotations

import unittest

import numpy as np


class TestNavRolloutSmoke(unittest.TestCase):
    """One short, video-free rollout to exercise the real teacher + planner."""

    @classmethod
    def setUpClass(cls) -> None:
        try:
            from code.apps.warehouse_demo.nav_rollout import run_nav_rollout
            from code.sim.teacher import WBCTeacher
            from code.warehouse.arena import warehouse_scene_cfg
            from code.warehouse.layout import hero_layout

            cls._run = staticmethod(run_nav_rollout)
            cls.teacher = WBCTeacher(use_gpu=False)
            cls.layout = hero_layout()
            cls.cfg = warehouse_scene_cfg(cls.layout, robot="Bravo",
                                          rng=np.random.default_rng(0))
        except Exception as e:  # pragma: no cover - environment-dependent
            raise unittest.SkipTest(f"WBC/MuJoCo assets unavailable: {e}")

    def test_short_rollout_runs_and_plans(self) -> None:
        from code.apps.warehouse_demo.nav_rollout import NavParams, NavResult

        # Bravo bay (-2, -5) -> ~1.5 m straight north into the open aisle.
        res = self._run(self.cfg, (-2.0, -3.5), max_steps=100,
                        teacher=self.teacher, params=NavParams())
        self.assertIsInstance(res, NavResult)
        self.assertGreater(res.steps, 0)
        self.assertLessEqual(res.steps, 100)
        self.assertIn(res.outcome, {"success", "timeout"})
        self.assertFalse(res.fell, "robot fell during a short clean rollout")
        self.assertGreater(res.path_length_planned, 0.0)
        self.assertGreaterEqual(res.path_length_walked, 0.0)
        # A plausible clearance was measured (not NaN) and the pelvis stayed up.
        self.assertEqual(res.min_wall_clearance, res.min_wall_clearance)  # not NaN


if __name__ == "__main__":
    unittest.main()
