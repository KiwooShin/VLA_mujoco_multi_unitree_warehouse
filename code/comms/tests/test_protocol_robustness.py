"""Robustness regression tests for code.comms.protocol.

These cover the failure paths that never fire on the scripted happy-path
scenarios but will in live/interactive use:

* an owner that falls or gets an unreachable goal mid-fetch fails FAST (emits
  ``TASK_FAILED``) instead of hanging until the step budget runs out;
* a missed mock pickup is re-approached once, then fails — the owner never
  reports ``TASK_COMPLETE`` without actually holding the object;
* an ASSIST searcher that falls reports (``REJECT``) so the owner reassigns the
  region and recovers gracefully;
* a searcher handed an uncoverable region ``REJECT``\\ s it;
* the default region set is one :mod:`code.fleet.search` understands.
"""

from __future__ import annotations

import unittest

from code.comms.bus import MessageBus
from code.comms.messages import ObjectQuery, Performative
from code.comms.protocol import DEFAULT_REGIONS, RobotProtocol, RobotState
from code.comms.tests._helpers import FakeActions, FakeClock, pump
from code.comms.tests.test_protocol import _FleetFixture, LOC, _task

P = Performative


def _single_owner(actions: FakeActions):
    """Wire a lone owner (no peers) that already sees its object, task posted."""
    clock = FakeClock()
    bus = MessageBus(clock.now)
    alpha = RobotProtocol("Alpha", bus, actions, peers=(),
                          search_regions=DEFAULT_REGIONS)
    bus.post("user", "Alpha", P.REQUEST_TASK, {"task": _task()})
    return clock, bus, alpha


def _perfs(bus: MessageBus):
    return [m.performative for m in bus.transcript]


def _reasons(bus: MessageBus, perf: Performative):
    return [m.payload.get("reason") for m in bus.transcript if m.performative is perf]


class TestOwnerFailsFast(unittest.TestCase):
    """A fall / unreachable goal aborts the task instead of hanging."""

    def test_fall_while_navigating_emits_task_failed(self) -> None:
        act = FakeActions(static_location=LOC, arrives=False)
        clock, bus, alpha = _single_owner(act)
        alpha.step(clock.now()); clock.tick()  # begin nav
        alpha.step(clock.now()); clock.tick()  # still navigating
        self.assertEqual(alpha.state, RobotState.OWNER_NAVIGATING)
        act.nav_fails, act.fail_reason = True, "robot fell"
        alpha.step(clock.now()); clock.tick()
        self.assertEqual(alpha.last_result, "failed")
        self.assertEqual(_reasons(bus, P.TASK_FAILED), ["robot fell"])
        self.assertTrue(alpha.is_idle())
        # The failure went to the requester only.
        failed = [m for m in bus.transcript if m.performative is P.TASK_FAILED]
        self.assertEqual(failed[0].recipient, "user")

    def test_unreachable_goal_emits_task_failed(self) -> None:
        act = FakeActions(static_location=LOC, arrives=False,
                          nav_fails=True, fail_reason="goal unreachable")
        clock, bus, alpha = _single_owner(act)
        for _ in range(3):
            alpha.step(clock.now()); clock.tick()
        self.assertEqual(alpha.last_result, "failed")
        self.assertEqual(_reasons(bus, P.TASK_FAILED), ["goal unreachable"])
        self.assertNotIn(P.TASK_COMPLETE, _perfs(bus))

    def test_fall_while_delivering_never_completes(self) -> None:
        act = FakeActions(static_location=LOC, arrives=True)
        clock, bus, alpha = _single_owner(act)
        alpha.step(clock.now()); clock.tick()  # nav
        alpha.step(clock.now()); clock.tick()  # arrive -> pickup -> delivering
        self.assertEqual(alpha.state, RobotState.OWNER_DELIVERING)
        act.nav_fails, act.fail_reason = True, "robot fell"
        alpha.step(clock.now()); clock.tick()
        self.assertEqual(alpha.last_result, "failed")
        self.assertEqual(_reasons(bus, P.TASK_FAILED), ["robot fell"])
        self.assertNotIn(P.TASK_COMPLETE, _perfs(bus))


class TestPickupRetry(unittest.TestCase):
    """A missed grasp is retried once, then fails; success on retry completes."""

    def test_missed_pickup_retries_then_fails(self) -> None:
        act = FakeActions(static_location=LOC, arrives=True, can_pickup=False)
        clock, bus, alpha = _single_owner(act)
        for _ in range(6):
            alpha.step(clock.now()); clock.tick()
        self.assertEqual(alpha.last_result, "failed")
        self.assertEqual(_reasons(bus, P.TASK_FAILED),
                         ["could not pick up the object"])
        self.assertEqual(len(act.calls("pickup")), 2)  # initial + one retry
        self.assertEqual(act.calls("deliver"), [])
        self.assertNotIn(P.TASK_COMPLETE, _perfs(bus))

    def test_pickup_succeeds_on_retry_completes(self) -> None:
        act = FakeActions(static_location=LOC, arrives=True, can_pickup=False)
        clock, bus, alpha = _single_owner(act)
        alpha.step(clock.now()); clock.tick()  # nav
        alpha.step(clock.now()); clock.tick()  # first (missed) pickup -> retry
        self.assertEqual(alpha.state, RobotState.OWNER_NAVIGATING)
        act.can_pickup = True
        for _ in range(4):
            alpha.step(clock.now()); clock.tick()
        self.assertEqual(alpha.last_result, "complete")
        self.assertEqual(len(act.calls("pickup")), 2)
        self.assertIn(P.TASK_COMPLETE, _perfs(bus))


class TestUncoverableRegionRejects(unittest.TestCase):
    """A region with no reachable patrol is rejected, not silently accepted."""

    def test_reject_when_region_uncoverable(self) -> None:
        clock = FakeClock()
        bus = MessageBus(clock.now)
        act = FakeActions(can_search=False)
        bravo = RobotProtocol("Bravo", bus, act, peers=("Alpha",),
                              search_regions=DEFAULT_REGIONS)
        bus.post("Alpha", "Bravo", P.COMMAND_SEARCH,
                 {"query": ObjectQuery("red", "cube"), "region": "north",
                  "cancel": False})
        pump(clock, [bravo], 3)
        rej = [m for m in bus.transcript if m.performative is P.REJECT]
        self.assertEqual(len(rej), 1)
        self.assertEqual((rej[0].sender, rej[0].recipient), ("Bravo", "Alpha"))
        self.assertEqual(rej[0].payload["reason"], "region not coverable")
        self.assertTrue(bravo.is_idle())
        self.assertEqual(act.calls("start_search"), ["north"])
        self.assertNotIn(P.ACCEPT, _perfs(bus))


class TestSearcherFallReassigns(unittest.TestCase):
    """An ASSIST searcher that falls REJECTs; the owner reassigns and recovers."""

    def test_fallen_searcher_region_goes_to_reserve(self) -> None:
        actions = {
            "Alpha": FakeActions(),  # never sees / delegates
            "Bravo": FakeActions(nav_fails=True, fail_reason="searcher fell"),
            "Charlie": FakeActions(search_location=LOC, find_after_polls=1),
            "Delta": FakeActions()}  # reserve
        fx = _FleetFixture(actions, peers=("Bravo", "Charlie", "Delta"),
                           regions=("north", "middle"))
        fx.bus.post("user", "Alpha", P.REQUEST_TASK, {"task": _task()})
        pump(fx.clock, fx.order(("Alpha", "Bravo", "Charlie", "Delta")), 40)

        # Bravo (north) fell and reported it.
        bravo_rej = [m for m in fx.bus.transcript if m.performative is P.REJECT
                     and m.sender == "Bravo"]
        self.assertEqual(len(bravo_rej), 1)
        self.assertEqual(bravo_rej[0].recipient, "Alpha")
        self.assertEqual(bravo_rej[0].payload["reason"], "searcher fell")
        self.assertEqual(len(fx.actions["Bravo"].calls("abort_search")), 1)
        # Its region was reassigned to the reserve peer Delta.
        delta_cmds = [m for m in fx.bus.transcript
                      if m.performative is P.COMMAND_SEARCH and m.recipient == "Delta"
                      and not m.payload.get("cancel")]
        self.assertEqual(len(delta_cmds), 1)
        self.assertEqual(delta_cmds[0].payload["region"], "north")
        # The mission still completed (Charlie found it).
        self.assertEqual(fx.alpha.last_result, "complete")


class TestDefaultRegions(unittest.TestCase):
    """Finding 6: the default regions must be ones the searcher understands."""

    def test_default_regions_value(self) -> None:
        self.assertEqual(DEFAULT_REGIONS, ("north", "middle", "south"))


if __name__ == "__main__":
    unittest.main()
