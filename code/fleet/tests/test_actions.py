"""Unit tests for the protocol->world bridge (code.fleet.actions).

Uses scripted fakes for the unit / search controller / carry manager so the
bridge's wiring is exercised without stepping any physics.
"""

from __future__ import annotations

import math
import unittest

import numpy as np

from code.comms.messages import ObjectQuery
from code.fleet.actions import FleetRobotActions
from code.sim.arena_build import COLORS
from code.warehouse.arena import warehouse_scene_cfg
from code.warehouse.layout import hero_layout

_CMAP = dict(COLORS)


class _FakeUnit:
    def __init__(self, xy, yaw, *, done=False, fell=False) -> None:
        self._xy = xy
        self._yaw = yaw
        self.done = done
        self.fell = fell
        self.goals = []
        self.plan_ok = True
        self.halted = False

    @property
    def xy(self):
        return self._xy

    @property
    def yaw(self):
        return self._yaw

    @property
    def base_height(self):
        return 0.74

    def assign_goal(self, xy) -> bool:
        self.goals.append(tuple(xy))
        return self.plan_ok

    def halt(self) -> None:
        self.halted = True


class _FakeSearch:
    def __init__(self, ok=True) -> None:
        self.started = []
        self.stopped = 0
        self._ok = ok

    def start(self, cfg, region) -> bool:
        self.started.append(region)
        return self._ok

    def stop(self) -> None:
        self.stopped += 1


class _FakeCarry:
    def __init__(self, ok=True) -> None:
        self.picked = []
        self.dest = []
        self._carried = set()
        self._ok = ok

    def pickup(self, robot, index) -> bool:
        self.picked.append((robot, index))
        if self._ok:
            self._carried.add(index)
        return self._ok

    def set_destination(self, robot, xy) -> None:
        self.dest.append((robot, tuple(xy)))

    def is_carried(self, index) -> bool:
        return index in self._carried


def _scene():
    # Two red objects: a cube at (-5, 0.5) [visible to Alpha's south bay pose]
    # and a distractor red ball far away.
    objs = [
        {"color_name": "red", "color_rgb": _CMAP["red"], "shape_name": "cube",
         "size": 0.24, "x": -5.0, "y": 0.5},
        {"color_name": "red", "color_rgb": _CMAP["red"], "shape_name": "ball",
         "size": 0.24, "x": 6.5, "y": 4.7},
    ]
    return warehouse_scene_cfg(hero_layout(), objects=objs,
                               rng=np.random.default_rng(0))


class TestCanSee(unittest.TestCase):
    def test_sees_object_in_open_lane(self) -> None:
        cfg = _scene()
        unit = _FakeUnit((-5.0, -5.0), math.pi / 2)  # Alpha bay, facing +y
        act = FleetRobotActions("Alpha", unit, cfg, _FakeSearch(), _FakeCarry())
        loc = act.can_see(ObjectQuery("red", "cube"))
        self.assertEqual(loc, (-5.0, 0.5))

    def test_hidden_object_returns_none(self) -> None:
        cfg = _scene()
        unit = _FakeUnit((-5.0, -5.0), math.pi / 2)
        act = FleetRobotActions("Alpha", unit, cfg, _FakeSearch(), _FakeCarry())
        # The red BALL sits in the far alcove -> not visible from the bay.
        self.assertIsNone(act.can_see(ObjectQuery("red", "ball")))

    def test_carried_object_not_seen(self) -> None:
        cfg = _scene()
        unit = _FakeUnit((-5.0, -5.0), math.pi / 2)
        carry = _FakeCarry()
        carry._carried.add(0)  # the red cube is being carried
        act = FleetRobotActions("Alpha", unit, cfg, _FakeSearch(), carry)
        self.assertIsNone(act.can_see(ObjectQuery("red", "cube")))


class TestNavAndSearch(unittest.TestCase):
    def test_goto_plans_and_arrived_reflects_unit(self) -> None:
        cfg = _scene()
        unit = _FakeUnit((-5.0, -5.0), 0.0, done=False)
        act = FleetRobotActions("Alpha", unit, cfg, _FakeSearch(), _FakeCarry())
        act.goto((1.0, 2.0))
        self.assertEqual(unit.goals[-1], (1.0, 2.0))
        self.assertFalse(act.arrived())
        unit.done = True
        self.assertTrue(act.arrived())

    def test_goto_retries_when_plan_fails(self) -> None:
        cfg = _scene()
        unit = _FakeUnit((0.0, 0.0), 0.0)
        unit.plan_ok = False
        act = FleetRobotActions("Alpha", unit, cfg, _FakeSearch(), _FakeCarry())
        act.goto((5.0, 0.0))
        # First attempt at the target, then a pulled-in approach point.
        self.assertEqual(len(unit.goals), 2)
        self.assertLess(unit.goals[1][0], 5.0)

    def test_search_start_and_abort(self) -> None:
        cfg = _scene()
        search = _FakeSearch()
        act = FleetRobotActions("Alpha", _FakeUnit((0, 0), 0), cfg, search,
                                _FakeCarry())
        act.start_search(ObjectQuery("red", "cube"), "north")
        self.assertEqual(search.started, ["north"])
        act.abort_search()
        self.assertEqual(search.stopped, 1)


class TestManipulation(unittest.TestCase):
    def test_pickup_nearest_match(self) -> None:
        cfg = _scene()
        unit = _FakeUnit((-5.0, 0.5), 0.0)  # standing on the red cube
        carry = _FakeCarry()
        act = FleetRobotActions("Alpha", unit, cfg, _FakeSearch(), carry)
        act.pickup(ObjectQuery("red", None))  # matches cube (near) and ball (far)
        self.assertEqual(carry.picked, [("Alpha", 0)])  # picks the nearer cube

    def test_deliver_sets_destination_and_plans(self) -> None:
        cfg = _scene()
        unit = _FakeUnit((-5.0, 0.5), 0.0)
        carry = _FakeCarry()
        act = FleetRobotActions("Alpha", unit, cfg, _FakeSearch(), carry)
        act.deliver((5.8, -1.0))
        self.assertEqual(carry.dest, [("Alpha", (5.8, -1.0))])
        self.assertEqual(unit.goals[-1], (5.8, -1.0))


class TestFailureAndBooleans(unittest.TestCase):
    """Robustness: failed()/failure_reason and the pickup/search return values."""

    def test_failed_true_when_unit_fell(self) -> None:
        cfg = _scene()
        unit = _FakeUnit((-5.0, -5.0), 0.0, fell=True)
        act = FleetRobotActions("Alpha", unit, cfg, _FakeSearch(), _FakeCarry())
        self.assertTrue(act.failed())
        self.assertEqual(act.failure_reason(), "robot fell")

    def test_failed_true_when_goal_unreachable(self) -> None:
        cfg = _scene()
        unit = _FakeUnit((0.0, 0.0), 0.0)
        unit.plan_ok = False  # both the direct plan and the retry fail
        act = FleetRobotActions("Alpha", unit, cfg, _FakeSearch(), _FakeCarry())
        act.goto((5.0, 0.0))
        self.assertTrue(act.failed())
        self.assertEqual(act.failure_reason(), "goal unreachable")

    def test_failed_false_after_successful_plan(self) -> None:
        cfg = _scene()
        unit = _FakeUnit((-5.0, -5.0), 0.0)
        act = FleetRobotActions("Alpha", unit, cfg, _FakeSearch(), _FakeCarry())
        act.goto((1.0, 2.0))
        self.assertFalse(act.failed())

    def test_start_search_clears_stale_plan_failure(self) -> None:
        # A prior failed navigation must not make a fresh searcher look "failed".
        cfg = _scene()
        unit = _FakeUnit((0.0, 0.0), 0.0)
        unit.plan_ok = False
        act = FleetRobotActions("Alpha", unit, cfg, _FakeSearch(), _FakeCarry())
        act.goto((5.0, 0.0))
        self.assertTrue(act.failed())
        self.assertTrue(act.start_search(ObjectQuery("red", "cube"), "north"))
        self.assertFalse(act.failed())  # stale plan-failure cleared

    def test_start_search_returns_false_when_uncoverable(self) -> None:
        cfg = _scene()
        act = FleetRobotActions("Alpha", _FakeUnit((0, 0), 0), cfg,
                                _FakeSearch(ok=False), _FakeCarry())
        self.assertFalse(act.start_search(ObjectQuery("red", "cube"), "north"))

    def test_pickup_returns_false_when_out_of_range(self) -> None:
        cfg = _scene()
        unit = _FakeUnit((-5.0, 0.5), 0.0)
        act = FleetRobotActions("Alpha", unit, cfg, _FakeSearch(),
                                _FakeCarry(ok=False))
        self.assertFalse(act.pickup(ObjectQuery("red", "cube")))

    def test_pickup_returns_false_when_no_match(self) -> None:
        cfg = _scene()
        unit = _FakeUnit((-5.0, 0.5), 0.0)
        act = FleetRobotActions("Alpha", unit, cfg, _FakeSearch(), _FakeCarry())
        self.assertFalse(act.pickup(ObjectQuery("magenta", "pyramid")))


if __name__ == "__main__":
    unittest.main()
