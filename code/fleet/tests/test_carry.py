"""Mock pickup / carry / release tests (skip without WBC/MuJoCo/EGL).

Builds a real fleet with two red objects — one beside Alpha's bay (in reach), one
in the far alcove (out of reach) — and checks the pickup radius gate, that a
carried object tracks the hand, and that release drops it on the delivery pad.
"""

from __future__ import annotations

import math
import unittest

from code.sim.arena_build import COLORS

_CMAP = dict(COLORS)


def _objects():
    return [
        {"color_name": "red", "color_rgb": _CMAP["red"], "shape_name": "cube",
         "size": 0.24, "x": -4.6, "y": -5.0},   # ~0.4 m from Alpha's bay
        {"color_name": "red", "color_rgb": _CMAP["red"], "shape_name": "ball",
         "size": 0.24, "x": 6.5, "y": 4.7},      # far NE alcove
    ]


class TestCarry(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        try:
            from code.fleet.mission import MissionRunner

            cls.mr = MissionRunner(objects=_objects(), use_gpu=True)
        except unittest.SkipTest:
            raise
        except Exception as e:  # pragma: no cover - environment-dependent
            raise unittest.SkipTest(f"WBC/MuJoCo/EGL unavailable: {e}")

    @classmethod
    def tearDownClass(cls) -> None:
        if hasattr(cls, "mr"):
            cls.mr.close()

    def test_pickup_within_radius_succeeds(self) -> None:
        carry = self.mr.carry
        self.assertTrue(carry.pickup("Alpha", 0))
        self.assertTrue(carry.carrying("Alpha"))
        self.assertTrue(carry.is_carried(0))
        self.assertEqual(carry.carried_index("Alpha"), 0)

    def test_pickup_out_of_range_fails(self) -> None:
        self.assertFalse(self.mr.carry.pickup("Bravo", 1))
        self.assertFalse(self.mr.carry.carrying("Bravo"))

    def test_carry_tracks_hand(self) -> None:
        carry = self.mr.carry
        if not carry.carrying("Alpha"):
            carry.pickup("Alpha", 0)
        carry.update()
        ax, ay = self.mr.fleet.units["Alpha"].xy
        obj = self.mr.scene_cfg["objects"][0]
        # The carried object sits within a hand's-reach of the pelvis.
        self.assertLess(math.hypot(obj["x"] - ax, obj["y"] - ay), 1.0)

    def test_release_places_on_pad(self) -> None:
        carry = self.mr.carry
        if not carry.carrying("Alpha"):
            carry.pickup("Alpha", 0)
        carry.set_destination("Alpha", (5.8, -1.0))
        idx = carry.release("Alpha")
        self.assertEqual(idx, 0)
        self.assertFalse(carry.carrying("Alpha"))
        obj = self.mr.scene_cfg["objects"][0]
        self.assertAlmostEqual(obj["x"], 5.8, places=3)
        self.assertAlmostEqual(obj["y"], -1.0, places=3)


if __name__ == "__main__":
    unittest.main()
