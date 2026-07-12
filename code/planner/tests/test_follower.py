"""Unit tests for code.planner.follower.WaypointFollower.

Covers monotonic progress, pure-pursuit lookahead targeting, arrival/done
semantics, degenerate 1-2 waypoint paths, and sensible re-targeting when the
robot is shoved off the path or the path loops back on itself.
"""

from __future__ import annotations

import math
import unittest

from code.planner.follower import WaypointFollower

_STRAIGHT = [(float(x), 0.0) for x in range(6)]  # (0,0)..(5,0)


class ConstructorTest(unittest.TestCase):
    def test_empty_path_raises(self) -> None:
        with self.assertRaises(ValueError):
            WaypointFollower([])

    def test_bad_params_raise(self) -> None:
        with self.assertRaises(ValueError):
            WaypointFollower(_STRAIGHT, arrive_radius=0.0)
        with self.assertRaises(ValueError):
            WaypointFollower(_STRAIGHT, lookahead=-1.0)


class LookaheadTest(unittest.TestCase):
    def test_target_is_lookahead_ahead(self) -> None:
        wf = WaypointFollower(_STRAIGHT, arrive_radius=0.35, lookahead=0.8)
        tgt = wf.target((2.0, 0.0))
        self.assertIsNotNone(tgt)
        assert tgt is not None
        self.assertAlmostEqual(tgt[0], 2.8, places=6)
        self.assertAlmostEqual(tgt[1], 0.0, places=6)

    def test_returns_goal_when_within_lookahead(self) -> None:
        wf = WaypointFollower(_STRAIGHT, arrive_radius=0.35, lookahead=0.8)
        tgt = wf.target((4.5, 0.0))  # 0.5 m from goal (5,0) < lookahead
        self.assertEqual(tgt, (5.0, 0.0))


class ProgressTest(unittest.TestCase):
    def test_progress_monotonic_along_path(self) -> None:
        wf = WaypointFollower(_STRAIGHT, arrive_radius=0.35, lookahead=0.8)
        last = -1.0
        x = 0.0
        while x <= 4.6:
            wf.target((x, 0.0))
            self.assertGreaterEqual(wf.progress_fraction + 1e-9, last)
            last = wf.progress_fraction
            x += 0.25
        self.assertGreater(wf.progress_fraction, 0.5)

    def test_progress_fraction_bounds(self) -> None:
        wf = WaypointFollower(_STRAIGHT)
        wf.target((0.0, 0.0))
        self.assertGreaterEqual(wf.progress_fraction, 0.0)
        self.assertLessEqual(wf.progress_fraction, 1.0)


class ArrivalTest(unittest.TestCase):
    def test_arrival_returns_none_and_sets_done(self) -> None:
        wf = WaypointFollower(_STRAIGHT, arrive_radius=0.35, lookahead=0.8)
        self.assertIsNone(wf.target((4.9, 0.0)))  # 0.1 m from goal
        self.assertTrue(wf.done)
        self.assertEqual(wf.progress_fraction, 1.0)

    def test_stays_done_after_arrival(self) -> None:
        wf = WaypointFollower(_STRAIGHT, arrive_radius=0.35)
        wf.target((5.0, 0.0))
        self.assertIsNone(wf.target((0.0, 0.0)))
        self.assertTrue(wf.done)


class ShortPathTest(unittest.TestCase):
    def test_single_waypoint_path(self) -> None:
        wf = WaypointFollower([(2.0, 2.0)], arrive_radius=0.35, lookahead=0.8)
        self.assertEqual(wf.target((0.0, 0.0)), (2.0, 2.0))  # steer at goal
        self.assertFalse(wf.done)
        self.assertIsNone(wf.target((2.1, 2.0)))  # within arrive radius
        self.assertTrue(wf.done)

    def test_two_waypoint_path(self) -> None:
        wf = WaypointFollower([(0.0, 0.0), (3.0, 0.0)], lookahead=0.8,
                              arrive_radius=0.35)
        tgt = wf.target((0.0, 0.0))
        assert tgt is not None
        self.assertAlmostEqual(tgt[0], 0.8, places=6)
        self.assertEqual(wf.target((2.5, 0.0)), (3.0, 0.0))  # within lookahead


class OffPathTest(unittest.TestCase):
    def test_lateral_offset_projects_forward(self) -> None:
        wf = WaypointFollower(_STRAIGHT, arrive_radius=0.35, lookahead=0.8)
        tgt = wf.target((2.0, 1.0))  # 1 m off to the side
        assert tgt is not None
        self.assertAlmostEqual(tgt[0], 2.8, places=6)
        self.assertAlmostEqual(tgt[1], 0.0, places=6)

    def test_no_backward_recapture(self) -> None:
        wf = WaypointFollower(_STRAIGHT, arrive_radius=0.35, lookahead=0.8)
        wf.target((3.0, 0.0))
        pf_mid = wf.progress_fraction
        # Robot shoved back near the start: progress must not regress.
        wf.target((0.5, 1.0))
        self.assertGreaterEqual(wf.progress_fraction + 1e-9, pf_mid)
        tgt = wf.target((0.5, 1.0))
        assert tgt is not None
        self.assertGreaterEqual(tgt[0], 3.0 - 1e-6)


class LoopPathTest(unittest.TestCase):
    def test_loop_progress_monotonic(self) -> None:
        # A path that returns close to its own start must not re-capture early
        # waypoints once progress is made.
        loop = [(0.0, 0.0), (3.0, 0.0), (3.0, 3.0), (0.0, 3.0), (0.0, 0.5)]
        wf = WaypointFollower(loop, arrive_radius=0.3, lookahead=0.8)
        # Scripted traversal around the loop.
        waypoints = [
            (0.0, 0.0), (1.5, 0.0), (3.0, 0.0), (3.0, 1.5), (3.0, 3.0),
            (1.5, 3.0), (0.0, 3.0), (0.0, 1.5),
        ]
        last = -1.0
        for w in waypoints:
            wf.target(w)
            self.assertGreaterEqual(wf.progress_fraction + 1e-9, last)
            last = wf.progress_fraction
        # Now near the physical start again but late in the path: no reset.
        self.assertGreater(wf.progress_fraction, 0.6)


if __name__ == "__main__":
    unittest.main()
