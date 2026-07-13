"""Scenario tests for code.comms.protocol.

Each scenario scripts fake perception/navigation and a fake clock, drives the
fleet through the message bus, and asserts the FULL ordered transcript (not just
endpoints) plus the structural need-to-know invariants.
"""

from __future__ import annotations

import unittest
from typing import List, Optional, Tuple

from code.comms.bus import MessageBus
from code.comms.messages import (ObjectQuery, Performative, TaskKind, TaskSpec,
                                 reconstruct_location)
from code.comms.protocol import RobotProtocol, RobotState
from code.comms.tests._helpers import FakeActions, FakeClock, pump

P = Performative
CALLSIGNS = ("Alpha", "Bravo", "Charlie", "Delta")
DELIVERY_XY = (5.8, -1.0)
LOC = (6.5, 4.7)  # where the object turns out to be


def _task(requester: str = "user") -> TaskSpec:
    return TaskSpec(TaskKind.FETCH, ObjectQuery("red", "cube"),
                    "delivery pad", DELIVERY_XY, requester=requester)


def _seq(bus: MessageBus) -> List[Tuple[str, str, Performative]]:
    """The transcript as an ordered (sender, recipient, performative) list."""
    return [(m.sender, m.recipient, m.performative) for m in bus.transcript]


def _region_of(bus: MessageBus, recipient: str, cancel: bool) -> Optional[str]:
    for m in bus.transcript:
        if (m.performative is P.COMMAND_SEARCH and m.recipient == recipient
                and bool(m.payload.get("cancel")) == cancel):
            return m.payload.get("region")
    return None


class _FleetFixture:
    """Owner ``Alpha`` plus peer protocols sharing one bus/clock."""

    def __init__(self, actions: dict, peers=("Bravo", "Charlie", "Delta"),
                 *, regions=None, reply_deadline=50, search_deadline=2000) -> None:
        self.clock = FakeClock()
        self.bus = MessageBus(self.clock.now)
        self.actions = actions
        kw = {"reply_deadline_steps": reply_deadline,
              "search_deadline_steps": search_deadline}
        if regions is not None:
            kw["search_regions"] = regions
        self.alpha = RobotProtocol("Alpha", self.bus, actions["Alpha"], peers, **kw)
        self.protos = {"Alpha": self.alpha}
        for cs in peers:
            others = [c for c in ("Alpha", *peers) if c != cs]
            self.protos[cs] = RobotProtocol(cs, self.bus, actions[cs], others, **kw)

    def order(self, names):
        return [self.protos[n] for n in names if n in self.protos]


class _NeedToKnowMixin:
    """Reusable assertions that helper robots never over-share."""

    def assert_need_to_know(self, bus: MessageBus, owner: str = "Alpha") -> None:
        for m in bus.transcript:
            if m.recipient == "user":
                self.assertEqual(m.sender, owner,
                                 f"only {owner} may message the user: {m}")
            if m.performative in (P.STATUS_UPDATE, P.TASK_COMPLETE, P.TASK_FAILED):
                self.assertEqual(m.sender, owner,
                                 f"only {owner} sends {m.performative.name}: {m}")
            if m.performative is P.REPORT_FOUND:
                self.assertEqual(m.recipient, owner,
                                 f"REPORT_FOUND must go to {owner} only: {m}")


class TestScenarioAOwnerSees(unittest.TestCase, _NeedToKnowMixin):
    """(a) Owner sees the object immediately -> zero robot-robot traffic."""

    def test_no_peer_traffic(self) -> None:
        actions = {"Alpha": FakeActions(static_location=LOC)}
        for cs in ("Bravo", "Charlie", "Delta"):
            actions[cs] = FakeActions()
        fx = _FleetFixture(actions)
        fx.bus.post("user", "Alpha", P.REQUEST_TASK, {"task": _task()})
        pump(fx.clock, fx.order(CALLSIGNS), 15)

        self.assertEqual(_seq(fx.bus), [
            ("user", "Alpha", P.REQUEST_TASK),
            ("Alpha", "user", P.STATUS_UPDATE),   # found / en route
            ("Alpha", "user", P.STATUS_UPDATE),   # picked up / delivering
            ("Alpha", "user", P.TASK_COMPLETE),   # delivered
        ])
        # Not a single peer-directed message.
        for perf in (P.QUERY_VISIBILITY, P.REPORT_VISIBILITY, P.COMMAND_SEARCH,
                     P.ACCEPT, P.REJECT, P.REPORT_FOUND):
            self.assertNotIn(perf, [m.performative for m in fx.bus.transcript])
        self.assertEqual(actions["Alpha"].calls("goto"), [LOC])
        self.assertEqual(actions["Alpha"].calls("deliver"), [DELIVERY_XY])
        self.assertEqual(len(actions["Alpha"].calls("pickup")), 1)
        self.assertEqual(fx.alpha.last_result, "complete")
        self.assert_need_to_know(fx.bus)


class TestScenarioBPeerSees(unittest.TestCase, _NeedToKnowMixin):
    """(b) First peer queried sees it -> one query, one report, no search."""

    def test_peer_visibility_short_circuits(self) -> None:
        actions = {"Alpha": FakeActions(),
                   "Bravo": FakeActions(static_location=LOC),
                   "Charlie": FakeActions(), "Delta": FakeActions()}
        fx = _FleetFixture(actions)
        fx.bus.post("user", "Alpha", P.REQUEST_TASK, {"task": _task()})
        pump(fx.clock, fx.order(CALLSIGNS), 15)

        self.assertEqual(_seq(fx.bus), [
            ("user", "Alpha", P.REQUEST_TASK),
            ("Alpha", "Bravo", P.QUERY_VISIBILITY),
            ("Bravo", "Alpha", P.REPORT_VISIBILITY),
            ("Alpha", "user", P.STATUS_UPDATE),
            ("Alpha", "user", P.STATUS_UPDATE),
            ("Alpha", "user", P.TASK_COMPLETE),
        ])
        perfs = [m.performative for m in fx.bus.transcript]
        self.assertEqual(perfs.count(P.QUERY_VISIBILITY), 1)
        self.assertEqual(perfs.count(P.REPORT_VISIBILITY), 1)
        self.assertNotIn(P.COMMAND_SEARCH, perfs)
        # Charlie and Delta were never queried.
        queried = [m.recipient for m in fx.bus.transcript
                   if m.performative is P.QUERY_VISIBILITY]
        self.assertEqual(queried, ["Bravo"])
        self.assertEqual(actions["Alpha"].calls("goto"), [LOC])
        self.assert_need_to_know(fx.bus)


class TestScenarioCDelegatedSearch(unittest.TestCase, _NeedToKnowMixin):
    """(c) Nobody sees it -> delegate search; Bravo finds and reports to Alpha only."""

    def _run(self) -> _FleetFixture:
        actions = {
            "Alpha": FakeActions(),
            "Bravo": FakeActions(search_location=LOC, find_after_polls=0),
            "Charlie": FakeActions(), "Delta": FakeActions()}
        fx = _FleetFixture(actions, regions=("north", "middle", "south"))
        fx.bus.post("user", "Alpha", P.REQUEST_TASK, {"task": _task()})
        pump(fx.clock, fx.order(CALLSIGNS), 25)
        return fx

    def test_full_transcript(self) -> None:
        fx = self._run()
        self.assertEqual(_seq(fx.bus), [
            ("user", "Alpha", P.REQUEST_TASK),
            ("Alpha", "Bravo", P.QUERY_VISIBILITY),
            ("Bravo", "Alpha", P.REPORT_VISIBILITY),
            ("Alpha", "Charlie", P.QUERY_VISIBILITY),
            ("Charlie", "Alpha", P.REPORT_VISIBILITY),
            ("Alpha", "Delta", P.QUERY_VISIBILITY),
            ("Delta", "Alpha", P.REPORT_VISIBILITY),
            ("Alpha", "Bravo", P.COMMAND_SEARCH),
            ("Alpha", "Charlie", P.COMMAND_SEARCH),
            ("Alpha", "Delta", P.COMMAND_SEARCH),
            ("Bravo", "Alpha", P.ACCEPT),
            ("Charlie", "Alpha", P.ACCEPT),
            ("Delta", "Alpha", P.ACCEPT),
            ("Bravo", "Alpha", P.REPORT_FOUND),
            ("Alpha", "Charlie", P.COMMAND_SEARCH),   # cancel
            ("Alpha", "Delta", P.COMMAND_SEARCH),     # cancel
            ("Alpha", "user", P.STATUS_UPDATE),
            ("Alpha", "user", P.STATUS_UPDATE),
            ("Alpha", "user", P.TASK_COMPLETE),
        ])

    def test_need_to_know_and_aborts(self) -> None:
        fx = self._run()
        self.assert_need_to_know(fx.bus)
        # The one REPORT_FOUND went to Alpha only.
        found = [m for m in fx.bus.transcript if m.performative is P.REPORT_FOUND]
        self.assertEqual(len(found), 1)
        self.assertEqual((found[0].sender, found[0].recipient), ("Bravo", "Alpha"))
        # Charlie and Delta each received a cancel; the finder did not.
        self.assertEqual(_region_of(fx.bus, "Charlie", cancel=True), "middle")
        self.assertEqual(_region_of(fx.bus, "Delta", cancel=True), "south")
        self.assertIsNone(_region_of(fx.bus, "Bravo", cancel=True))
        self.assertEqual(len(fx.actions["Charlie"].calls("abort_search")), 1)
        self.assertEqual(len(fx.actions["Delta"].calls("abort_search")), 1)
        # Helpers never fetch; the owner does.
        self.assertEqual(fx.actions["Bravo"].calls("pickup"), [])
        self.assertEqual(fx.actions["Alpha"].calls("goto"), [LOC])
        self.assertEqual(len(fx.actions["Alpha"].calls("pickup")), 1)
        self.assertEqual(fx.alpha.last_result, "complete")


class TestF3RelativeReports(unittest.TestCase):
    """(F3) Sightings are reported relative to the reporter, then reconstructed."""

    def test_report_found_is_relative_and_reconstructs(self) -> None:
        actions = {
            "Alpha": FakeActions(),
            "Bravo": FakeActions(search_location=LOC, find_after_polls=0,
                                 pose=(-4.0, 2.8), room="storage A"),
            "Charlie": FakeActions(), "Delta": FakeActions()}
        fx = _FleetFixture(actions, regions=("north", "middle", "south"))
        fx.bus.post("user", "Alpha", P.REQUEST_TASK, {"task": _task()})
        pump(fx.clock, fx.order(CALLSIGNS), 25)

        found = [m for m in fx.bus.transcript if m.performative is P.REPORT_FOUND]
        self.assertEqual(len(found), 1)
        pay = found[0].payload
        # No absolute transmitted; relative fields present + reconstructing to LOC.
        self.assertNotIn("location", pay)
        self.assertEqual(pay["reporter_pose"], (-4.0, 2.8))
        self.assertEqual(pay["room"], "storage A")
        self.assertEqual(reconstruct_location(pay), LOC)
        # The owner navigated to the reconstructed absolute position.
        self.assertEqual(fx.actions["Alpha"].calls("goto"), [LOC])
        # The transcript renders the exact F3 sentence in the reporter's voice.
        from code.comms.bus import format_line
        line = format_line(found[0])
        self.assertIn("I am robot Bravo, currently in storage A at position "
                      "(-4.0, 2.8).", line)
        self.assertIn("away from me.", line)


class TestScenarioDBusyReject(unittest.TestCase, _NeedToKnowMixin):
    """(d) A busy peer REJECTs -> owner re-plans onto a reserve peer."""

    def test_replan_transcript(self) -> None:
        actions = {
            "Alpha": FakeActions(),
            "Bravo": FakeActions(search_location=LOC, find_after_polls=1),
            "Charlie": FakeActions(), "Delta": FakeActions()}
        fx = _FleetFixture(actions, regions=("north", "south"))
        # Charlie is already out on another search -> it will REJECT (busy).
        fx.protos["Charlie"]._enter_assist("Base", ObjectQuery("blue", "ball"),
                                           "west", t=0)
        fx.bus.post("user", "Alpha", P.REQUEST_TASK, {"task": _task()})
        pump(fx.clock, fx.order(CALLSIGNS), 25)

        self.assertEqual(_seq(fx.bus), [
            ("user", "Alpha", P.REQUEST_TASK),
            ("Alpha", "Bravo", P.QUERY_VISIBILITY),
            ("Bravo", "Alpha", P.REPORT_VISIBILITY),
            ("Alpha", "Charlie", P.QUERY_VISIBILITY),
            ("Charlie", "Alpha", P.REPORT_VISIBILITY),
            ("Alpha", "Delta", P.QUERY_VISIBILITY),
            ("Delta", "Alpha", P.REPORT_VISIBILITY),
            ("Alpha", "Bravo", P.COMMAND_SEARCH),      # north
            ("Alpha", "Charlie", P.COMMAND_SEARCH),    # south
            ("Bravo", "Alpha", P.ACCEPT),
            ("Charlie", "Alpha", P.REJECT),            # busy
            ("Alpha", "Delta", P.COMMAND_SEARCH),      # re-plan: south -> Delta
            ("Delta", "Alpha", P.ACCEPT),
            ("Bravo", "Alpha", P.REPORT_FOUND),
            ("Alpha", "Delta", P.COMMAND_SEARCH),      # cancel Delta
            ("Alpha", "user", P.STATUS_UPDATE),
            ("Alpha", "user", P.STATUS_UPDATE),
            ("Alpha", "user", P.TASK_COMPLETE),
        ])
        # The reject and the re-plan target its region.
        rej = [m for m in fx.bus.transcript if m.performative is P.REJECT]
        self.assertEqual(rej[0].payload["reason"], "busy")
        self.assertEqual(_region_of(fx.bus, "Charlie", cancel=False), "south")
        self.assertEqual(_region_of(fx.bus, "Delta", cancel=False), "south")
        # Charlie was neither re-commanded nor cancelled by Alpha (it's on Base's job).
        charlie_cmds = [m for m in fx.bus.transcript
                        if m.performative is P.COMMAND_SEARCH and m.recipient == "Charlie"]
        self.assertEqual(len(charlie_cmds), 1)
        self.assertEqual(fx.alpha.last_result, "complete")
        self.assert_need_to_know(fx.bus)


class TestScenarioEReplyTimeout(unittest.TestCase, _NeedToKnowMixin):
    """(e) A peer never answers -> owner times out and moves to the next peer."""

    def test_timeout_advances_query(self) -> None:
        # Bravo is unresponsive (never stepped -> models a dropped/late reply).
        actions = {"Alpha": FakeActions(),
                   "Bravo": FakeActions(),
                   "Charlie": FakeActions(static_location=LOC)}
        fx = _FleetFixture(actions, peers=("Bravo", "Charlie"), reply_deadline=3)
        fx.bus.post("user", "Alpha", P.REQUEST_TASK, {"task": _task()})
        pump(fx.clock, fx.order(("Alpha", "Charlie")), 20)  # Bravo not pumped

        self.assertEqual(_seq(fx.bus), [
            ("user", "Alpha", P.REQUEST_TASK),
            ("Alpha", "Bravo", P.QUERY_VISIBILITY),
            ("Alpha", "Charlie", P.QUERY_VISIBILITY),   # only after Bravo timed out
            ("Charlie", "Alpha", P.REPORT_VISIBILITY),
            ("Alpha", "user", P.STATUS_UPDATE),
            ("Alpha", "user", P.STATUS_UPDATE),
            ("Alpha", "user", P.TASK_COMPLETE),
        ])
        # Bravo produced no reply at all.
        self.assertEqual(
            [m for m in fx.bus.transcript
             if m.sender == "Bravo"], [])
        # Charlie was queried only after the deadline elapsed (>= 3 steps later).
        qs = {m.recipient: m.t_step for m in fx.bus.transcript
              if m.performative is P.QUERY_VISIBILITY}
        self.assertGreaterEqual(qs["Charlie"] - qs["Bravo"], 3)
        self.assertEqual(fx.alpha.last_result, "complete")
        self.assert_need_to_know(fx.bus)


class TestScenarioFFleetRouting(unittest.TestCase):
    """(f) A fleet-addressed request goes to the allocator inbox, not a robot."""

    def test_fleet_request_to_allocator(self) -> None:
        actions = {cs: FakeActions() for cs in CALLSIGNS}
        fx = _FleetFixture(actions)
        fx.bus.post("user", "fleet", P.FLEET_REQUEST, {"task": _task()})
        pump(fx.clock, fx.order(CALLSIGNS), 10)

        # No robot consumed it; every robot stayed idle.
        for cs in CALLSIGNS:
            self.assertEqual(fx.protos[cs].state, RobotState.IDLE)
        self.assertEqual(fx.bus.pending("allocator"), 1)
        drained = fx.bus.drain("fleet")
        self.assertEqual(len(drained), 1)
        self.assertEqual(drained[0].performative, P.FLEET_REQUEST)
        # No STATUS/COMPLETE traffic was generated.
        self.assertEqual(
            [m for m in fx.bus.transcript
             if m.performative in (P.STATUS_UPDATE, P.TASK_COMPLETE)], [])


class TestScenarioGSearchExhausted(unittest.TestCase, _NeedToKnowMixin):
    """Search exhausts with no find -> TASK_FAILED to the requester."""

    def test_task_failed(self) -> None:
        actions = {"Alpha": FakeActions(), "Bravo": FakeActions()}  # Bravo never finds
        fx = _FleetFixture(actions, peers=("Bravo",), regions=("north",),
                           search_deadline=5)
        fx.bus.post("user", "Alpha", P.REQUEST_TASK, {"task": _task()})
        pump(fx.clock, fx.order(("Alpha", "Bravo")), 20)

        self.assertEqual(_seq(fx.bus), [
            ("user", "Alpha", P.REQUEST_TASK),
            ("Alpha", "Bravo", P.QUERY_VISIBILITY),
            ("Bravo", "Alpha", P.REPORT_VISIBILITY),
            ("Alpha", "Bravo", P.COMMAND_SEARCH),      # north
            ("Bravo", "Alpha", P.ACCEPT),
            ("Alpha", "Bravo", P.COMMAND_SEARCH),      # cancel on exhaustion
            ("Alpha", "user", P.TASK_FAILED),
        ])
        failed = [m for m in fx.bus.transcript if m.performative is P.TASK_FAILED]
        self.assertEqual(failed[0].payload["reason"], "search exhausted")
        self.assertEqual(fx.alpha.last_result, "failed")
        self.assertEqual(len(fx.actions["Bravo"].calls("abort_search")), 1)
        self.assert_need_to_know(fx.bus)


class TestProtocolMisc(unittest.TestCase):
    """Reflex / guard behaviours outside the main scenarios."""

    def test_busy_robot_rejects_new_task(self) -> None:
        actions = {"Alpha": FakeActions(), "Bravo": FakeActions()}
        fx = _FleetFixture(actions, peers=("Bravo",))
        fx.protos["Bravo"]._enter_assist("Base", ObjectQuery("red", "cube"),
                                         "west", t=0)
        # A REQUEST_TASK to a busy robot is silently declined (stays searching).
        fx.bus.post("user", "Bravo", P.REQUEST_TASK, {"task": _task()})
        pump(fx.clock, fx.order(("Bravo",)), 5)
        self.assertEqual(fx.protos["Bravo"].state, RobotState.ASSIST_SEARCHING)
        self.assertEqual(
            [m for m in fx.bus.transcript if m.sender == "Bravo"
             and m.recipient == "user"], [])

    def test_query_visibility_is_answered_in_any_state(self) -> None:
        actions = {"Alpha": FakeActions(), "Bravo": FakeActions(static_location=LOC)}
        fx = _FleetFixture(actions, peers=("Bravo",))
        fx.bus.post("Alpha", "Bravo", P.QUERY_VISIBILITY,
                    {"query": ObjectQuery("red", "cube")})
        pump(fx.clock, fx.order(("Bravo",)), 3)
        reports = [m for m in fx.bus.transcript
                   if m.performative is P.REPORT_VISIBILITY]
        self.assertEqual(len(reports), 1)
        self.assertTrue(reports[0].payload["visible"])
        # F3: the report is relative; the absolute is reconstructed by the owner.
        self.assertEqual(reconstruct_location(reports[0].payload), LOC)


if __name__ == "__main__":
    unittest.main()
