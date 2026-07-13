"""Unit tests for the pure RobotUnit state machine + status helpers.

The state transitions are exercised through the pure :func:`advance_state`
function; ``status_line``/``distance_to_goal`` are exercised on a bare instance
with a stubbed navigator (no physics stepped here).
"""

from __future__ import annotations

import math
import unittest
from types import SimpleNamespace

from code.fleet.robot_unit import RobotState, RobotUnit, advance_state


class TestAdvanceState(unittest.TestCase):
    def test_walking_continues(self) -> None:
        self.assertEqual(
            advance_state(RobotState.WALKING, fell=False, done=False, paused=False),
            RobotState.WALKING)

    def test_walking_pauses(self) -> None:
        self.assertEqual(
            advance_state(RobotState.WALKING, fell=False, done=False, paused=True),
            RobotState.PAUSED)

    def test_paused_resumes(self) -> None:
        self.assertEqual(
            advance_state(RobotState.PAUSED, fell=False, done=False, paused=False),
            RobotState.WALKING)

    def test_walking_arrives(self) -> None:
        self.assertEqual(
            advance_state(RobotState.WALKING, fell=False, done=True, paused=False),
            RobotState.ARRIVED)

    def test_fall_dominates_arrival(self) -> None:
        self.assertEqual(
            advance_state(RobotState.WALKING, fell=True, done=True, paused=False),
            RobotState.FALLEN)

    def test_arrived_is_sticky(self) -> None:
        self.assertEqual(
            advance_state(RobotState.ARRIVED, fell=True, done=False, paused=True),
            RobotState.ARRIVED)

    def test_fallen_is_sticky(self) -> None:
        self.assertEqual(
            advance_state(RobotState.FALLEN, fell=False, done=True, paused=False),
            RobotState.FALLEN)

    def test_idle_stays_idle(self) -> None:
        self.assertEqual(
            advance_state(RobotState.IDLE, fell=False, done=True, paused=False),
            RobotState.IDLE)

    def test_idle_can_fall(self) -> None:
        self.assertEqual(
            advance_state(RobotState.IDLE, fell=True, done=False, paused=False),
            RobotState.FALLEN)


class TestStatusHelpers(unittest.TestCase):
    def _unit(self, *, xy, goal_xy, state=RobotState.WALKING, name="Alpha"):
        u = object.__new__(RobotUnit)
        u.name = name
        u.state = state
        u.goal_xy = goal_xy
        u._nav = SimpleNamespace(xy=xy)
        return u

    def test_distance_to_goal(self) -> None:
        u = self._unit(xy=(0.0, 0.0), goal_xy=(3.0, 4.0))
        self.assertAlmostEqual(u.distance_to_goal(), 5.0, places=6)

    def test_distance_to_goal_none(self) -> None:
        u = self._unit(xy=(0.0, 0.0), goal_xy=None)
        self.assertEqual(u.distance_to_goal(), float("inf"))

    def test_status_line_has_name_and_state(self) -> None:
        u = self._unit(xy=(0.0, 0.0), goal_xy=(3.0, 4.0),
                       state=RobotState.WALKING, name="Bravo")
        line = u.status_line()
        self.assertIn("Bravo", line)
        self.assertIn("walking", line)
        self.assertIn("5.0m", line)

    def test_status_line_handles_no_goal(self) -> None:
        u = self._unit(xy=(0.0, 0.0), goal_xy=None, state=RobotState.IDLE)
        line = u.status_line()
        self.assertIn("idle", line)
        self.assertNotIn("inf", line)


class TestRobotStateValues(unittest.TestCase):
    def test_all_states_present(self) -> None:
        self.assertEqual(
            {s.value for s in RobotState},
            {"idle", "walking", "paused", "arrived", "fallen"})


if __name__ == "__main__":
    unittest.main()
