"""Unit tests for code.warehouse.arena (scene_cfg contract + MJCF build)."""

import unittest

import mujoco
import numpy as np

from code.sim.arena_build import EGO_W, GROUNDING_W, PROXIMITY_W, TP_W
from code.warehouse.arena import (
    build_warehouse_arena,
    warehouse_scene_cfg,
)
from code.warehouse.layout import hero_layout

_BASELINE_KEYS = {
    "arena_size", "robot_xy", "robot_yaw", "objects", "target_index",
    "instruction", "stop_r", "horizon", "lighting", "difficulty",
}
_WAREHOUSE_KEYS = {"walls", "zones", "layout_name", "hall_x", "hall_y"}


class TestSceneCfgContract(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.layout = hero_layout()
        cls.cfg = warehouse_scene_cfg(
            cls.layout, robot="Alpha", rng=np.random.default_rng(0), target_index=0
        )

    def test_has_baseline_and_warehouse_keys(self) -> None:
        for key in _BASELINE_KEYS | _WAREHOUSE_KEYS:
            self.assertIn(key, self.cfg)

    def test_difficulty_is_warehouse(self) -> None:
        self.assertEqual(self.cfg["difficulty"], "warehouse")

    def test_robot_pose_matches_spawn(self) -> None:
        sx, sy, syaw = self.layout.spawn_poses["Alpha"]
        self.assertEqual(self.cfg["robot_xy"], (sx, sy))
        self.assertEqual(self.cfg["robot_yaw"], syaw)

    def test_arena_size_covers_hall(self) -> None:
        self.assertEqual(self.cfg["arena_size"], 8.0)

    def test_hall_dims_are_exact(self) -> None:
        self.assertEqual(self.cfg["hall_x"], self.layout.hall_x)
        self.assertEqual(self.cfg["hall_y"], self.layout.hall_y)

    def test_objects_have_baseline_shape(self) -> None:
        objs = self.cfg["objects"]
        self.assertEqual(len(objs), len(self.layout.object_spots))
        for obj in objs:
            for key in ("color_name", "color_rgb", "shape_name", "size", "x", "y"):
                self.assertIn(key, obj)

    def test_objects_unique_color_shape(self) -> None:
        combos = {(o["color_name"], o["shape_name"]) for o in self.cfg["objects"]}
        self.assertEqual(len(combos), len(self.cfg["objects"]))

    def test_objects_placed_at_spots(self) -> None:
        placed = {(round(o["x"], 3), round(o["y"], 3)) for o in self.cfg["objects"]}
        spots = {(round(x, 3), round(y, 3)) for (x, y) in self.layout.object_spots}
        self.assertEqual(placed, spots)

    def test_instruction_auto_generated(self) -> None:
        self.assertTrue(self.cfg["instruction"])
        tgt = self.cfg["objects"][self.cfg["target_index"]]
        self.assertIn(tgt["color_name"], self.cfg["instruction"])
        self.assertIn(tgt["shape_name"], self.cfg["instruction"])

    def test_walls_serialized_as_dicts(self) -> None:
        walls = self.cfg["walls"]
        self.assertEqual(len(walls), len(self.layout.walls))
        self.assertTrue(all(isinstance(w, dict) and "name" in w for w in walls))

    def test_zones_serialized_and_layout_name(self) -> None:
        self.assertEqual(len(self.cfg["zones"]), len(self.layout.zones))
        self.assertEqual(self.cfg["layout_name"], "hero")

    def test_explicit_objects_passed_through(self) -> None:
        objs = [{"color_name": "red", "color_rgb": (220, 40, 40),
                 "shape_name": "ball", "size": 0.24, "x": 0.0, "y": 0.0}]
        cfg = warehouse_scene_cfg(self.layout, objects=objs)
        self.assertEqual(cfg["objects"], objs)

    def test_unknown_robot_raises(self) -> None:
        with self.assertRaises(KeyError):
            warehouse_scene_cfg(self.layout, robot="Zulu")

    def test_bad_target_index_raises(self) -> None:
        with self.assertRaises(ValueError):
            warehouse_scene_cfg(self.layout, rng=np.random.default_rng(0),
                                target_index=999)


class TestBuildWarehouseArena(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.layout = hero_layout()
        cls.cfg = warehouse_scene_cfg(cls.layout, rng=np.random.default_rng(1))
        cls.model = build_warehouse_arena(cls.cfg)

    def _gid(self, name: str) -> int:
        return mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, name)

    def test_compiles_to_mjmodel(self) -> None:
        self.assertIsInstance(self.model, mujoco.MjModel)

    def test_walls_present_by_name(self) -> None:
        for wname in ("wall_E", "wall_W", "wall_N", "wall_S",
                      "shelf_A_w", "shelf_A_e", "shelf_B_w", "shelf_B_e",
                      "part_ne_v", "part_ne_h", "part_sw"):
            self.assertGreaterEqual(self._gid(wname), 0, f"missing wall {wname}")

    def test_objects_present_by_name(self) -> None:
        for i in range(len(self.cfg["objects"])):
            self.assertGreaterEqual(self._gid(f"obj_{i}"), 0, f"missing obj_{i}")

    def test_zones_present_by_name(self) -> None:
        for zname in ("zone_delivery", "zone_bay_Alpha", "zone_bay_Bravo",
                      "zone_bay_Charlie", "zone_bay_Delta"):
            self.assertGreaterEqual(self._gid(zname), 0, f"missing {zname}")

    def test_zone_pads_are_noncolliding(self) -> None:
        gid = self._gid("zone_delivery")
        self.assertEqual(self.model.geom_contype[gid], 0)
        self.assertEqual(self.model.geom_conaffinity[gid], 0)

    def test_offscreen_buffer_covers_all_cameras(self) -> None:
        self.assertGreaterEqual(self.model.vis.global_.offwidth,
                                max(EGO_W, GROUNDING_W, PROXIMITY_W, TP_W))

    def test_floor_recolored(self) -> None:
        fid = self.model.geom("floor").id
        np.testing.assert_allclose(self.model.geom_rgba[fid],
                                   [0.86, 0.86, 0.88, 1.0], atol=1e-6)

    def test_overhead_lights_added(self) -> None:
        self.assertGreaterEqual(self.model.nlight, 5)  # 1 g1 + 4 overhead

    def test_yawed_wall_has_rotation_quat(self) -> None:
        # All hero walls are axis-aligned, so the build path must at least
        # produce identity quats for them (yaw=0 -> quat [1,0,0,0]).
        gid = self._gid("wall_E")
        np.testing.assert_allclose(self.model.geom_quat[gid], [1, 0, 0, 0],
                                   atol=1e-6)


class TestRendererSmoke(unittest.TestCase):
    """Real one-frame EGL render against the warehouse model; skips if no EGL."""

    @classmethod
    def setUpClass(cls) -> None:
        from code.sim.arena_render import ArenaRenderer
        try:
            layout = hero_layout()
            cfg = warehouse_scene_cfg(layout, rng=np.random.default_rng(2))
            cls.model = build_warehouse_arena(cfg)
            cls.renderer = ArenaRenderer(cls.model)
        except Exception as e:  # pragma: no cover - environment-dependent
            raise unittest.SkipTest(f"EGL/MuJoCo renderer unavailable: {e}")
        cls.cfg = cfg

    @classmethod
    def tearDownClass(cls) -> None:
        if hasattr(cls, "renderer"):
            cls.renderer.close()

    def setUp(self) -> None:
        self.data = mujoco.MjData(self.model)
        mujoco.mj_resetData(self.model, self.data)
        rx, ry = self.cfg["robot_xy"]
        self.data.qpos[0:2] = [rx, ry]
        self.data.qpos[2] = 0.79
        self.data.qpos[3:7] = [1, 0, 0, 0]
        mujoco.mj_forward(self.model, self.data)

    def test_render_ego_frame(self) -> None:
        rgb, depth, intr = self.renderer.render_ego(self.data, yaw=self.cfg["robot_yaw"])
        self.assertEqual(rgb.shape, (240, 320, 3))
        self.assertEqual(depth.shape, (240, 320))

    def test_render_tp_frame(self) -> None:
        tp_cam = self.renderer.make_tp_cam()
        self.renderer.update_tp_cam(tp_cam, self.data)
        rgb = self.renderer.render_tp(self.data, tp_cam)
        self.assertEqual(rgb.shape, (480, 640, 3))


if __name__ == "__main__":
    unittest.main()
