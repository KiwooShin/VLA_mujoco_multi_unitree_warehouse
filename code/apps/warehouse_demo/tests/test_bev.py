"""Unit tests for code.apps.warehouse_demo.bev (framing + projection math)."""

from __future__ import annotations

import math
import unittest

from code.apps.warehouse_demo.bev import BevCamera, fit_bev_camera


class TestProjection(unittest.TestCase):
    """Pinhole projection math against known, hand-computed cameras."""

    def test_lookat_projects_to_center(self) -> None:
        cam = BevCamera(lookat=(1.0, 2.0, 0.0), distance=8.0, azimuth_deg=90.0,
                        elevation_deg=-60.0, fovy_deg=45.0, width=640, height=480)
        u, v, depth = cam.project((1.0, 2.0, 0.0))
        self.assertAlmostEqual(u, 320.0, places=4)
        self.assertAlmostEqual(v, 240.0, places=4)
        self.assertAlmostEqual(depth, 8.0, places=4)

    def test_depth_decreases_toward_camera(self) -> None:
        cam = BevCamera(lookat=(0.0, 0.0, 0.0), distance=5.0, azimuth_deg=0.0,
                        elevation_deg=0.0, fovy_deg=90.0, width=100, height=100)
        # Camera looks along +x from eye=(-5,0,0); a point 1 m nearer along x.
        _, _, depth = cam.project((-1.0, 0.0, 0.0))
        self.assertAlmostEqual(depth, 4.0, places=4)

    def test_known_axis_alignment(self) -> None:
        # az=0, el=0 -> forward=+x, image-right=-y (world), image-up=+z (world).
        cam = BevCamera(lookat=(0.0, 0.0, 0.0), distance=5.0, azimuth_deg=0.0,
                        elevation_deg=0.0, fovy_deg=90.0, width=100, height=100)
        # f = (h/2)/tan(45) = 50.
        u_up, v_up, _ = cam.project((0.0, 0.0, 1.0))   # 1 m up in world
        self.assertAlmostEqual(u_up, 50.0, places=4)
        self.assertAlmostEqual(v_up, 40.0, places=4)   # up -> smaller v
        u_y, v_y, _ = cam.project((0.0, 1.0, 0.0))     # 1 m +y in world
        self.assertAlmostEqual(u_y, 40.0, places=4)    # +y maps left (right=-y)
        self.assertAlmostEqual(v_y, 50.0, places=4)

    def test_project_xy_returns_ints(self) -> None:
        cam = fit_bev_camera(16.0, 12.0)
        px = cam.project_xy((0.0, 0.0))
        self.assertIsInstance(px[0], int)
        self.assertIsInstance(px[1], int)


class TestFitBevCamera(unittest.TestCase):
    """Framing math: the whole hall must fall inside the frame."""

    def test_distance_matches_topdown_fit(self) -> None:
        cam = fit_bev_camera(16.0, 12.0, width=640, height=480, fovy_deg=45.0,
                             margin=1.3)
        half_tan = math.tan(math.radians(45.0) / 2.0)
        expected = 1.3 * max((8.0) / (half_tan * (640 / 480)), 6.0 / half_tan)
        self.assertAlmostEqual(cam.distance, expected, places=4)

    def test_all_hall_corners_in_frame(self) -> None:
        cam = fit_bev_camera(16.0, 12.0, width=640, height=480)
        for cx in (-8.0, 8.0):
            for cy in (-6.0, 6.0):
                u, v, depth = cam.project((cx, cy, 0.0))
                self.assertGreater(depth, 0.0)
                self.assertTrue(0 <= u <= 640, f"u={u} out of frame")
                self.assertTrue(0 <= v <= 480, f"v={v} out of frame")

    def test_north_up_east_right(self) -> None:
        # Default azimuth 90/elevation -78 gives a north-up, east-right map.
        cam = fit_bev_camera(16.0, 12.0)
        u_e, _, _ = cam.project((6.0, 0.0, 0.0))   # east
        u_w, _, _ = cam.project((-6.0, 0.0, 0.0))  # west
        self.assertGreater(u_e, u_w)               # east is to the right
        _, v_n, _ = cam.project((0.0, 5.0, 0.0))   # north
        _, v_s, _ = cam.project((0.0, -5.0, 0.0))  # south
        self.assertLess(v_n, v_s)                  # north is up (smaller v)

    def test_bad_margin_raises(self) -> None:
        with self.assertRaises(ValueError):
            fit_bev_camera(16.0, 12.0, margin=0.0)

    def test_bad_hall_raises(self) -> None:
        with self.assertRaises(ValueError):
            fit_bev_camera(0.0, 12.0)


class TestMjvCamera(unittest.TestCase):
    def test_to_mjv_camera_fields(self) -> None:
        cam = BevCamera(lookat=(0.0, 0.0, 0.1), distance=18.0, azimuth_deg=90.0,
                        elevation_deg=-78.0)
        mjv = cam.to_mjv_camera()
        self.assertAlmostEqual(float(mjv.distance), 18.0, places=5)
        self.assertAlmostEqual(float(mjv.azimuth), 90.0, places=5)
        self.assertAlmostEqual(float(mjv.elevation), -78.0, places=5)


if __name__ == "__main__":
    unittest.main()
