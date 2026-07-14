"""BEV overlay drawing on fake frames: multi-ring, pad, path (cv2, no MuJoCo)."""

from __future__ import annotations

import unittest

import numpy as np

from code.apps.demos import draw
from code.apps.demos.models import FrameState, Pad, PlannedPath, Ring, RobotFrame
from code.apps.warehouse_demo.bev import fit_bev_camera


def _cam():
    return fit_bev_camera(16.0, 12.0, width=960, height=600, fovy_deg=45.0)


def _blank():
    return np.zeros((600, 960, 3), dtype=np.uint8)


class TestBevOverlays(unittest.TestCase):
    def test_multiple_rings_each_add_ink(self):
        cam = _cam()
        one = _blank()
        draw.draw_ring(one, cam, Ring((0.0, 0.0)), step=0)
        n_one = int(np.count_nonzero(one))
        self.assertGreater(n_one, 0)

        two = _blank()
        draw.draw_ring(two, cam, Ring((-3.0, 2.0)), step=0)
        draw.draw_ring(two, cam, Ring((3.0, -2.0), color=(255, 0, 0)), step=0)
        self.assertGreater(int(np.count_nonzero(two)), n_one)

    def test_ring_color_override_is_used(self):
        cam = _cam()
        img = _blank()
        draw.draw_ring(img, cam, Ring((0.0, 0.0), color=(255, 0, 0)), step=0)
        blue = (img[:, :, 0] > 200) & (img[:, :, 2] < 60)
        self.assertTrue(bool(blue.any()))

    def test_draw_bev_overlays_handles_full_generic_state(self):
        cam = _cam()
        img = _blank()
        state = FrameState(
            step=7,
            robots=[RobotFrame("Alpha", (0.0, 0.0), 0.3, trail=[(0, 0), (1, 1), (2, 1)]),
                    RobotFrame("Bravo", (-2.0, 1.0), 1.2, trail=[(-2, 1), (-1, 1)])],
            rings=[Ring((3.0, 2.0)), Ring((-3.0, -1.0))],
            pads=[Pad((4.0, -2.0), 1.0, 1.0)],
            planned_paths=[PlannedPath([(0, 0), (2, 2), (4, -1)])])
        draw.draw_bev_overlays(img, cam, state)
        self.assertGreater(int(np.count_nonzero(img)), 0)

    def test_empty_state_draws_nothing_without_error(self):
        cam = _cam()
        img = _blank()
        draw.draw_bev_overlays(img, cam, FrameState(step=0))
        self.assertEqual(int(np.count_nonzero(img)), 0)


if __name__ == "__main__":
    unittest.main()
