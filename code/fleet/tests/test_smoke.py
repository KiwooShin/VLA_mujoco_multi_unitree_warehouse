"""Headless 4-robot co-simulation smoke + cross-visibility proof.

Builds the full fleet (four federated-physics G1s + the shared viz model),
steps it ~100 control steps and checks nobody falls, the robots move, and the
viz model stays in sync with each robot's physics. Also renders the mandatory
cross-visibility check (Alpha's ego sees Bravo). Skips on a fresh clone without
the WBC ONNX / G1 assets or a working EGL renderer.
"""

from __future__ import annotations

import math
import unittest

import numpy as np


def _robot_qpos(nq: int, x: float, y: float, yaw: float) -> np.ndarray:
    q = np.zeros(nq)
    q[0:3] = [x, y, 0.72]
    q[3:7] = [math.cos(yaw / 2.0), 0.0, 0.0, math.sin(yaw / 2.0)]
    q[7:22] = [-0.1, 0, 0, 0.3, -0.2, 0, -0.1, 0, 0, 0.3, -0.2, 0, 0, 0, 0]
    return q


class TestFleetSmoke(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        try:
            from code.apps.warehouse_demo.bev import fit_bev_camera
            from code.fleet.fleet import Fleet
            from code.warehouse.layout import hero_layout

            layout = hero_layout()
            spots = layout.object_spots
            goals = {"Alpha": spots[7], "Bravo": spots[4],
                     "Charlie": spots[3], "Delta": spots[6]}
            cls.fleet = Fleet(layout, goals, use_gpu=True, build_viz=True)
            cam = fit_bev_camera(
                layout.hall_x, layout.hall_y, width=320, height=240,
                fovy_deg=float(cls.fleet.viz.model.vis.global_.fovy))
            cls.fleet.viz.render_bev(cam)  # probe EGL; raises if unavailable
        except unittest.SkipTest:
            raise
        except Exception as e:  # pragma: no cover - environment-dependent
            raise unittest.SkipTest(f"WBC/MuJoCo/EGL unavailable: {e}")

    @classmethod
    def tearDownClass(cls) -> None:
        if hasattr(cls, "fleet"):
            cls.fleet.close()

    def test_00_all_robots_planned_and_walking(self) -> None:
        from code.fleet.robot_unit import RobotState

        for name, unit in self.fleet.units.items():
            self.assertTrue(unit.plan_ok, f"{name} failed to plan")
            self.assertEqual(unit.state, RobotState.WALKING)
            self.assertGreater(unit.planned_length, 0.0)

    def test_01_short_run_moves_without_falling(self) -> None:
        self.fleet.run(100)
        self.assertEqual(self.fleet.step_count, 100)
        self.assertFalse(self.fleet.any_fell, "a robot fell in a short clean run")
        moved = [u.walked_length for u in self.fleet.units.values()]
        self.assertTrue(any(w > 0.05 for w in moved), f"nobody moved: {moved}")

    def test_02_viz_stays_in_sync_with_physics(self) -> None:
        # After stepping, each robot's viz pelvis must match its physics xy.
        for name, unit in self.fleet.units.items():
            p = self.fleet.viz.pelvis_xpos(name)
            self.assertAlmostEqual(p[0], unit.xy[0], places=2)
            self.assertAlmostEqual(p[1], unit.xy[1], places=2)

    def test_03_cross_visibility_pixel_diff(self) -> None:
        from code.fleet.viz import FleetViz
        from code.warehouse.arena import warehouse_scene_cfg
        from code.warehouse.layout import CALLSIGNS, hero_layout

        cfg = warehouse_scene_cfg(hero_layout(), rng=np.random.default_rng(0))
        viz = FleetViz(cfg, CALLSIGNS)
        nq = viz.robot_nq
        yaw = math.pi / 2.0
        far = {"Charlie": _robot_qpos(nq, 100, 100, 0),
               "Delta": _robot_qpos(nq, 120, 120, 0)}
        try:
            viz.sync({"Alpha": _robot_qpos(nq, 0.0, -1.0, yaw),
                      "Bravo": _robot_qpos(nq, 0.0, 1.5, 0.0), **far})
            near = viz.render_ego("Alpha", yaw)
            viz.sync({"Alpha": _robot_qpos(nq, 0.0, -1.0, yaw),
                      "Bravo": _robot_qpos(nq, 150, 150, 0.0), **far})
            away = viz.render_ego("Alpha", yaw)
        finally:
            viz.close()
        diff = np.abs(near.astype(np.int16) - away.astype(np.int16))
        frac = float((diff.max(axis=2) > 20).mean())
        # Bravo standing 2.5 m ahead must occupy a measurable chunk of the view.
        self.assertGreater(frac, 0.01,
                           f"Alpha's ego barely changed with Bravo present: {frac}")


if __name__ == "__main__":
    unittest.main()
