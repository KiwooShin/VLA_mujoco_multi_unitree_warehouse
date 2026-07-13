"""Unit tests for the shared viz model: qpos address mapping + kinematic sync.

Builds the (compile-only) viz model and drives ``mj_forward`` — no ONNX policy
and no EGL renderer are needed, so these run wherever the G1 XML is present.
Skips cleanly on a fresh clone without the ``third_party/`` assets.
"""

from __future__ import annotations

import math
import unittest

import numpy as np


def _setup_or_skip():
    try:
        from code.fleet.viz import ACCENT_RGBA, FleetViz, build_viz_model
        from code.warehouse.arena import warehouse_scene_cfg
        from code.warehouse.layout import CALLSIGNS, hero_layout

        cfg = warehouse_scene_cfg(hero_layout(), rng=np.random.default_rng(0))
        return cfg, list(CALLSIGNS), FleetViz, build_viz_model, ACCENT_RGBA
    except Exception as e:  # pragma: no cover - environment-dependent
        raise unittest.SkipTest(f"MuJoCo/G1 assets unavailable: {e}")


def _robot_qpos(nq: int, x: float, y: float, yaw: float) -> np.ndarray:
    q = np.zeros(nq)
    q[0:3] = [x, y, 0.72]
    q[3:7] = [math.cos(yaw / 2.0), 0.0, 0.0, math.sin(yaw / 2.0)]
    q[7:22] = [-0.1, 0, 0, 0.3, -0.2, 0, -0.1, 0, 0, 0.3, -0.2, 0, 0, 0, 0]
    return q


class TestVizQposMapping(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.cfg, cls.callsigns, cls.FleetViz, cls.build_viz_model, cls.accent = \
            _setup_or_skip()
        cls.model, cls.addr = cls.build_viz_model(cls.cfg, cls.callsigns)

    def test_blocks_are_contiguous_in_callsign_order(self) -> None:
        n = len(self.callsigns)
        self.assertEqual(self.model.nq % n, 0)
        robot_nq = self.model.nq // n
        self.assertEqual(robot_nq, 36)  # G1: 7 free + 29 hinge
        expected = {c: i * robot_nq for i, c in enumerate(self.callsigns)}
        self.assertEqual(self.addr, expected)

    def test_addr_points_at_free_joint(self) -> None:
        import mujoco

        for name in self.callsigns:
            jid = self.model.joint(f"{name.lower()}_floating_base_joint")
            self.assertEqual(int(jid.qposadr[0]), self.addr[name])
            self.assertEqual(jid.type, mujoco.mjtJoint.mjJNT_FREE)

    def test_torso_tint_baked_in(self) -> None:
        for name in self.callsigns:
            gid = self.model.geom(f"{name.lower()}_accent_torso_0").id
            np.testing.assert_allclose(
                self.model.geom_rgba[gid], self.accent[name], atol=1e-6)


class TestVizSync(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.cfg, cls.callsigns, cls.FleetViz, _, _ = _setup_or_skip()
        cls.viz = cls.FleetViz(cls.cfg, cls.callsigns)

    def test_sync_places_each_pelvis(self) -> None:
        # A distinct known pose per robot; the viz pelvis must land there.
        targets = {"Alpha": (2.0, -1.0, 0.4), "Bravo": (-3.0, 2.5, -0.7),
                   "Charlie": (1.5, 3.3, 1.2), "Delta": (-5.0, 0.5, 0.0)}
        poses = {n: _robot_qpos(self.viz.robot_nq, x, y, yaw)
                 for n, (x, y, yaw) in targets.items()}
        self.viz.sync(poses)
        for n, (x, y, _yaw) in targets.items():
            p = self.viz.pelvis_xpos(n)
            self.assertAlmostEqual(p[0], x, places=3)
            self.assertAlmostEqual(p[1], y, places=3)
            self.assertAlmostEqual(p[2], 0.72, places=3)

    def test_sync_copies_joint_angles(self) -> None:
        # Swinging the left hip pitch in the physics qpos must move the viz foot
        # (a body downstream of that joint) — proving joint angles sync, not just
        # the root pose.
        base = _robot_qpos(self.viz.robot_nq, 0.0, 0.0, 0.0)
        others = {c: base.copy() for c in self.callsigns if c != "Alpha"}
        self.viz.sync({"Alpha": base, **others})
        foot0 = self.viz.data.body("alpha_left_ankle_roll_link").xpos.copy()
        bent = base.copy()
        bent[7] = -0.9  # left_hip_pitch_joint (first hinge, qpos idx 7)
        self.viz.sync({"Alpha": bent, **{c: base.copy() for c in others}})
        foot1 = self.viz.data.body("alpha_left_ankle_roll_link").xpos.copy()
        self.assertGreater(float(np.linalg.norm(foot1 - foot0)), 1e-2)

    def test_wrong_length_raises(self) -> None:
        with self.assertRaises(ValueError):
            self.viz.sync({"Alpha": np.zeros(self.viz.robot_nq - 1)})


if __name__ == "__main__":
    unittest.main()
