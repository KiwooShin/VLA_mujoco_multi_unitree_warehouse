"""Regression tests for close-range fetch-goal refinement (D-14 delivery outlier).

Two mechanisms, both exercised here with scripted fakes (no physics):

* :meth:`RobotProtocol.refine_nav_goal` — the pure gating math that steers an
  in-flight fetch goal onto a fresher detector estimate: rate-limited, a bounded
  *same-object* nudge only, and only while the owner is still outside pickup
  range. This is what recovers the D-14 case (owner sent to an off report, the
  approach continuously refines the goal onto the true object).
* the pickup re-approach forcing a fresh confirm (:meth:`RobotActions.reconfirm_target`)
  before its single retry, so the retry heads for the object's current best
  estimate rather than the stale reported point — and stays byte-identical in
  oracle mode (``reconfirm_target`` returns ``None``).
"""

from __future__ import annotations

import unittest

from code.comms.bus import MessageBus
from code.comms.messages import Performative
from code.comms.protocol import (GOAL_REFINE_INTERVAL, GOAL_REFINE_MAX_DELTA_M,
                                  GOAL_REFINE_MIN_DELTA_M, GOAL_REFINE_MIN_RANGE_M,
                                  RobotProtocol, RobotState)
from code.comms.tests._helpers import FakeActions, FakeClock
from code.comms.tests.test_protocol import LOC, _task
from code.comms.tests.test_protocol_robustness import (_perfs, _reasons,
                                                       _single_owner)

P = Performative

# A found location and a fresh close-range estimate ~1.5 m away (same object).
GOAL = (6.5, 4.7)
FAR_ROBOT = (0.0, 0.0)      # owner still far from GOAL -> refinement allowed
REFINED = (5.3, 4.2)        # ~1.5 m from GOAL: a plausible same-object nudge


def _owner_navigating() -> tuple:
    """A lone owner that already sees its object at GOAL and has begun navigating."""
    act = FakeActions(static_location=GOAL, arrives=False)
    clock, bus, alpha = _single_owner(act)
    alpha.step(clock.now())  # REQUEST_TASK -> can_see(GOAL) -> begin_navigation
    clock.tick()
    return act, clock, bus, alpha


class TestRefineNavGoalMath(unittest.TestCase):
    """The pure gating of :meth:`RobotProtocol.refine_nav_goal`."""

    def test_accepts_plausible_same_object_nudge(self) -> None:
        act, clock, bus, alpha = _owner_navigating()
        self.assertEqual(alpha.state, RobotState.OWNER_NAVIGATING)
        t = GOAL_REFINE_INTERVAL + 5
        self.assertTrue(alpha.refine_nav_goal(REFINED, t, robot_xy=FAR_ROBOT))
        # The goal moved and a fresh navigation was issued to the refined point.
        self.assertEqual(alpha.located_target, REFINED)
        self.assertEqual(act.calls("goto")[-1], REFINED)

    def test_rejects_subthreshold_nudge(self) -> None:
        act, clock, bus, alpha = _owner_navigating()
        tiny = (GOAL[0] + GOAL_REFINE_MIN_DELTA_M * 0.5, GOAL[1])
        t = GOAL_REFINE_INTERVAL + 5
        self.assertFalse(alpha.refine_nav_goal(tiny, t, robot_xy=FAR_ROBOT))
        self.assertEqual(alpha.located_target, GOAL)          # unchanged
        self.assertEqual(act.calls("goto"), [GOAL])           # no re-plan

    def test_rejects_far_jump_different_object(self) -> None:
        act, clock, bus, alpha = _owner_navigating()
        far = (GOAL[0] + GOAL_REFINE_MAX_DELTA_M + 1.0, GOAL[1])
        t = GOAL_REFINE_INTERVAL + 5
        self.assertFalse(alpha.refine_nav_goal(far, t, robot_xy=FAR_ROBOT))
        self.assertEqual(alpha.located_target, GOAL)

    def test_rejects_when_within_pickup_range_of_goal(self) -> None:
        act, clock, bus, alpha = _owner_navigating()
        near_goal = (GOAL[0] + GOAL_REFINE_MIN_RANGE_M * 0.5, GOAL[1])
        t = GOAL_REFINE_INTERVAL + 5
        # Robot is already within the pickup radius of the goal -> don't nudge.
        self.assertFalse(alpha.refine_nav_goal(REFINED, t, robot_xy=near_goal))
        self.assertEqual(alpha.located_target, GOAL)

    def test_rate_limited(self) -> None:
        act, clock, bus, alpha = _owner_navigating()
        t = GOAL_REFINE_INTERVAL + 5
        self.assertTrue(alpha.refine_nav_goal(REFINED, t, robot_xy=FAR_ROBOT))
        # A second refinement inside the interval is rejected...
        second = (REFINED[0] + 0.5, REFINED[1])
        self.assertFalse(
            alpha.refine_nav_goal(second, t + GOAL_REFINE_INTERVAL - 1,
                                  robot_xy=FAR_ROBOT))
        self.assertEqual(alpha.located_target, REFINED)
        # ...but allowed once the interval has elapsed.
        self.assertTrue(
            alpha.refine_nav_goal(second, t + GOAL_REFINE_INTERVAL,
                                  robot_xy=FAR_ROBOT))
        self.assertEqual(alpha.located_target, second)

    def test_no_refine_outside_navigating(self) -> None:
        # Idle owner (task just posted, not yet stepped) never refines.
        act = FakeActions(static_location=GOAL)
        clock, bus, alpha = _single_owner(act)
        self.assertEqual(alpha.state, RobotState.IDLE)
        self.assertFalse(
            alpha.refine_nav_goal(REFINED, GOAL_REFINE_INTERVAL + 5,
                                  robot_xy=FAR_ROBOT))

    def test_no_refine_while_delivering(self) -> None:
        # Drive the owner into OWNER_DELIVERING (arrives + pickup ok), then refine.
        act = FakeActions(static_location=GOAL, arrives=True, can_pickup=True)
        clock, bus, alpha = _single_owner(act)
        alpha.step(clock.now()); clock.tick()   # nav
        alpha.step(clock.now()); clock.tick()   # arrive -> pickup -> delivering
        self.assertEqual(alpha.state, RobotState.OWNER_DELIVERING)
        goto_before = list(act.calls("goto"))
        self.assertFalse(
            alpha.refine_nav_goal(REFINED, clock.now() + GOAL_REFINE_INTERVAL,
                                  robot_xy=FAR_ROBOT))
        self.assertEqual(act.calls("goto"), goto_before)  # no re-plan mid-carry


class TestPickupRetryFreshConfirm(unittest.TestCase):
    """The pickup re-approach forces a fresh confirm before its single retry."""

    def test_retry_reapproaches_refined_estimate(self) -> None:
        # Grasp misses; reconfirm_target yields a fresh, closer estimate.
        act = FakeActions(static_location=GOAL, arrives=True, can_pickup=False,
                          reconfirm_location=REFINED)
        clock, bus, alpha = _single_owner(act)
        alpha.step(clock.now()); clock.tick()   # nav to GOAL
        alpha.step(clock.now()); clock.tick()   # arrive -> miss -> reconfirm+retry
        self.assertEqual(alpha.state, RobotState.OWNER_NAVIGATING)
        # A fresh confirm ran and the retry heads for the REFINED point, not GOAL.
        self.assertEqual(len(act.calls("reconfirm_target")), 1)
        self.assertEqual(act.calls("goto"), [GOAL, REFINED])

    def test_retry_keeps_goal_when_no_fresh_fix(self) -> None:
        # Oracle path: reconfirm_target returns None -> retry re-approaches GOAL
        # (byte-identical to the historical single-retry behaviour).
        act = FakeActions(static_location=GOAL, arrives=True, can_pickup=False)
        clock, bus, alpha = _single_owner(act)
        alpha.step(clock.now()); clock.tick()
        alpha.step(clock.now()); clock.tick()
        self.assertEqual(alpha.state, RobotState.OWNER_NAVIGATING)
        self.assertEqual(act.calls("goto"), [GOAL, GOAL])

    def test_refined_retry_then_pickup_completes(self) -> None:
        act = FakeActions(static_location=GOAL, arrives=True, can_pickup=False,
                          reconfirm_location=REFINED)
        clock, bus, alpha = _single_owner(act)
        alpha.step(clock.now()); clock.tick()   # nav
        alpha.step(clock.now()); clock.tick()   # miss -> reconfirm -> retry to REFINED
        act.can_pickup = True                   # now within reach of the true object
        for _ in range(4):
            alpha.step(clock.now()); clock.tick()
        self.assertEqual(alpha.last_result, "complete")
        self.assertIn(P.TASK_COMPLETE, _perfs(bus))
        self.assertEqual(len(act.calls("pickup")), 2)  # missed once, then held

    def test_far_reconfirm_rejected_keeps_goal(self) -> None:
        # A reconfirm estimate beyond same-object range must not hijack the retry.
        far = (GOAL[0] + GOAL_REFINE_MAX_DELTA_M + 1.0, GOAL[1])
        act = FakeActions(static_location=GOAL, arrives=True, can_pickup=False,
                          reconfirm_location=far)
        clock, bus, alpha = _single_owner(act)
        alpha.step(clock.now()); clock.tick()
        alpha.step(clock.now()); clock.tick()
        self.assertEqual(act.calls("goto"), [GOAL, GOAL])  # stale-far ignored


if __name__ == "__main__":
    unittest.main()
