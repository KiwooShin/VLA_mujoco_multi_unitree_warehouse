"""Unit tests for the path-length task allocator (code.fleet.allocator)."""

from __future__ import annotations

import unittest

import numpy as np

from code.comms.messages import ObjectQuery
from code.fleet import allocator as AL
from code.sim.arena_build import COLORS
from code.warehouse.arena import warehouse_scene_cfg
from code.warehouse.layout import CALLSIGNS, hero_layout

_CMAP = dict(COLORS)


def _objects(target_spot: int):
    fill = [("orange", "cube", 0.24), ("blue", "cylinder", 0.22),
            ("green", "ball", 0.24), ("yellow", "cone", 0.26),
            ("purple", "cube", 0.24), ("cyan", "cylinder", 0.22),
            ("blue", "ball", 0.24)]
    out = []
    fi = 0
    for i, (x, y) in enumerate(hero_layout().object_spots):
        if i == target_spot:
            c, s, sz = "red", "cube", 0.24
        else:
            c, s, sz = fill[fi]
            fi += 1
        out.append({"color_name": c, "color_rgb": _CMAP[c], "shape_name": s,
                    "size": float(sz), "x": float(x), "y": float(y)})
    return out


def _cfg(target_spot: int):
    return warehouse_scene_cfg(hero_layout(), objects=_objects(target_spot),
                               rng=np.random.default_rng(0))


def _poses():
    lay = hero_layout()
    return {cs: AL.RobotPose(lay.spawn_poses[cs][:2], lay.spawn_poses[cs][2], 0.74)
            for cs in CALLSIGNS}


class TestPlannedLength(unittest.TestCase):
    def test_reachable_and_unreachable(self) -> None:
        cfg = _cfg(6)
        d = AL.planned_path_length(cfg, (-5.0, -5.0), (5.8, -1.0))
        self.assertTrue(4.0 < d < 40.0)
        # A point outside the hall is unreachable.
        self.assertEqual(AL.planned_path_length(cfg, (-5.0, -5.0), (99.0, 99.0)),
                         float("inf"))


class TestAllocateVisible(unittest.TestCase):
    def test_picks_shortest_path_to_visible_object(self) -> None:
        # Spot 6 (-5, 0.5) is visible to Alpha at its spawn.
        cfg = _cfg(6)
        poses = _poses()
        res = AL.allocate(poses, cfg, ObjectQuery("red", "cube"), list(CALLSIGNS))
        self.assertEqual(res.reason, "visible")
        self.assertIsNotNone(res.target_xy)
        # Independently recompute every cost to the object and check the argmin.
        gt = {cs: AL.planned_path_length(cfg, poses[cs].xy, res.target_xy)
              for cs in CALLSIGNS}
        expected = min(CALLSIGNS, key=lambda c: gt[c])
        self.assertEqual(res.winner, expected)
        for cs in CALLSIGNS:
            self.assertAlmostEqual(res.costs[cs], gt[cs], places=3)


class TestAllocateSearch(unittest.TestCase):
    def test_search_mode_scores_nearest_region(self) -> None:
        # Alcove spot 5 is hidden from all -> search-mode allocation.
        cfg = _cfg(5)
        poses = _poses()
        res = AL.allocate(poses, cfg, ObjectQuery("red", "cube"), list(CALLSIGNS))
        self.assertEqual(res.reason, "search")
        # Each idle robot got a region and the winner is the global cost argmin.
        self.assertEqual(set(res.region), set(CALLSIGNS))
        expected = min(CALLSIGNS, key=lambda c: res.costs[c])
        self.assertEqual(res.winner, expected)


class TestAllocateEdges(unittest.TestCase):
    def test_no_idle_robots(self) -> None:
        cfg = _cfg(6)
        res = AL.allocate(_poses(), cfg, ObjectQuery("red", "cube"), [])
        self.assertIsNone(res.winner)
        self.assertIn("no idle robot", res.describe())

    def test_tiebreak_is_callsign_order(self) -> None:
        # Two robots at the SAME pose tie; the earlier callsign must win.
        cfg = _cfg(6)
        lay = hero_layout()
        same = AL.RobotPose(lay.spawn_poses["Bravo"][:2],
                            lay.spawn_poses["Bravo"][2], 0.74)
        poses = {"Bravo": same, "Charlie": same}
        res = AL.allocate(poses, cfg, ObjectQuery("red", "cube"),
                          ["Bravo", "Charlie"])
        self.assertEqual(res.winner, "Bravo")


class TestNonFinitePose(unittest.TestCase):
    """Robustness: a NaN/inf robot pose is skipped, never scored or chosen."""

    def test_nan_pose_skipped_finite_wins(self) -> None:
        cfg = _cfg(6)
        lay = hero_layout()
        good = AL.RobotPose(lay.spawn_poses["Alpha"][:2],
                            lay.spawn_poses["Alpha"][2], 0.74)
        broken = AL.RobotPose((float("nan"), float("inf")), 0.0, 0.74)
        poses = {"Alpha": good, "Bravo": broken}
        res = AL.allocate(poses, cfg, ObjectQuery("red", "cube"),
                          ["Alpha", "Bravo"])
        self.assertEqual(res.winner, "Alpha")
        self.assertEqual(res.costs["Bravo"], float("inf"))

    def test_all_non_finite_yields_no_winner(self) -> None:
        cfg = _cfg(6)
        broken = AL.RobotPose((float("nan"), 0.0), 0.0, 0.74)
        res = AL.allocate({"Alpha": broken}, cfg, ObjectQuery("red", "cube"),
                          ["Alpha"])
        self.assertIsNone(res.winner)


if __name__ == "__main__":
    unittest.main()
