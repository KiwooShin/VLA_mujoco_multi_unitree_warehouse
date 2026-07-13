"""Six-robot fleet construction — the callsign-agnostic scale-up (N=6).

Builds the full six-robot fleet on ``rooms6_layout`` (six federated-physics G1s
+ ONE shared viz model) and asserts the abstractions derive everything from the
roster: six units, one shared kinematic model whose qpos tiles into six equal
per-robot blocks, correct pause priorities, and — in groundnet mode — a single
detector object shared across all six per-robot perception states. Skips on a
fresh clone without the WBC ONNX / G1 assets / EGL / GROUND_NET checkpoint.
"""

from __future__ import annotations

import unittest

from code.warehouse.layout import CALLSIGNS6, callsigns_for_layout, rooms6_layout


class TestSixRobotFleet(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        try:
            from code.apps.warehouse_demo.bev import fit_bev_camera
            from code.fleet.fleet import Fleet

            cls.layout = rooms6_layout()
            cls.callsigns = list(callsigns_for_layout(cls.layout))
            spots = cls.layout.object_spots
            # Cross-hall goals: west bays drive east, east bays drive west.
            gi = {"Alpha": 6, "Bravo": 11, "Charlie": 7,
                  "Delta": 2, "Echo": 3, "Foxtrot": 10}
            goals = {cs: (float(spots[i][0]), float(spots[i][1]))
                     for cs, i in gi.items()}
            cls.fleet = Fleet(cls.layout, goals, callsigns=cls.callsigns,
                              use_gpu=True, build_viz=True)
            cam = fit_bev_camera(
                cls.layout.hall_x, cls.layout.hall_y, width=320, height=240,
                fovy_deg=float(cls.fleet.viz.model.vis.global_.fovy))
            cls.fleet.viz.render_bev(cam)  # probe EGL
        except unittest.SkipTest:
            raise
        except Exception as e:  # pragma: no cover - environment-dependent
            raise unittest.SkipTest(f"WBC/MuJoCo/EGL unavailable: {e}")

    @classmethod
    def tearDownClass(cls) -> None:
        if hasattr(cls, "fleet"):
            cls.fleet.close()

    def test_six_units_from_roster(self) -> None:
        self.assertEqual(tuple(self.fleet.units), CALLSIGNS6)
        self.assertEqual(len(self.fleet.units), 6)
        # Pause priority is exactly the roster order (Alpha highest).
        self.assertEqual(self.fleet.priorities,
                         {c: i for i, c in enumerate(CALLSIGNS6)})

    def test_one_shared_viz_model_tiles_six_blocks(self) -> None:
        viz = self.fleet.viz
        self.assertEqual(len(viz.callsigns), 6)
        # ONE model; its qpos partitions into six equal per-robot blocks.
        self.assertEqual(viz.model.nq % 6, 0)
        self.assertEqual(viz.robot_nq, viz.model.nq // 6)
        self.assertEqual(set(viz.qpos_addr), set(CALLSIGNS6))
        self.assertEqual(len(set(viz.qpos_addr.values())), 6)

    def test_all_six_plan_and_walk(self) -> None:
        from code.fleet.robot_unit import RobotState
        for name, unit in self.fleet.units.items():
            self.assertTrue(unit.plan_ok, f"{name} failed to plan")
            self.assertEqual(unit.state, RobotState.WALKING)

    def test_short_run_syncs_all_six_without_falling(self) -> None:
        self.fleet.run(80)
        self.assertFalse(self.fleet.any_fell)
        for name, unit in self.fleet.units.items():
            p = self.fleet.viz.pelvis_xpos(name)
            self.assertAlmostEqual(p[0], unit.xy[0], places=2)
            self.assertAlmostEqual(p[1], unit.xy[1], places=2)


class TestSixRobotSharedDetector(unittest.TestCase):
    """Groundnet: one detector object is shared across all six perceptions."""

    def test_shared_detector_across_six(self) -> None:
        try:
            from code.fleet.perception_bridge import load_shared_detector
        except Exception as e:  # pragma: no cover
            raise unittest.SkipTest(f"perception bridge unavailable: {e}")
        if load_shared_detector() is None:
            raise unittest.SkipTest("GROUND_NET checkpoint unavailable")
        from code.fleet.mission import MissionRunner
        layout = rooms6_layout()
        try:
            mr = MissionRunner(layout=layout,
                               callsigns=callsigns_for_layout(layout),
                               use_gpu=True, perception_mode="groundnet",
                               search_deadline_steps=100)
        except unittest.SkipTest:
            raise
        except Exception as e:  # pragma: no cover
            raise unittest.SkipTest(f"WBC/MuJoCo/EGL unavailable: {e}")
        try:
            self.assertEqual(set(mr.perceptions), set(CALLSIGNS6))
            self.assertEqual(len(mr.perceptions), 6)
            # The detector weights are shared; only each robot's mutable
            # GroundNetState is per-robot (the anti-singleton isolation).
            shared = mr.perceptions["Alpha"]._state.detector
            self.assertIsNotNone(shared)
            for cs in CALLSIGNS6:
                self.assertTrue(mr.perceptions[cs].has_detector)
                # The SAME model object (not six copies) backs all six robots.
                self.assertIs(mr.perceptions[cs]._state.detector, shared)
        finally:
            mr.close()


if __name__ == "__main__":
    unittest.main()
