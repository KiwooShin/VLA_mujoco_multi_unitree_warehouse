"""Scenario tests for the Demo Set v2 protocol dialogue / concurrency / re-task.

Fake perception + a fake clock drive the protocol through the bus (no MuJoCo) and
assert the FULL ordered transcript plus the structural invariants of the five new
flows: CLARIFY (happy / re-ask / exhausted), sequential multi-goal legs, a
mid-mission re-task (mid-navigation and mid-delegation), and — most important —
that two concurrent owners keep need-to-know intact (owner A never learns owner
B's find).
"""

from __future__ import annotations

import unittest
from typing import Dict, List, Tuple

from code.comms.bus import MessageBus
from code.comms.messages import (Message, ObjectQuery, Performative, TaskKind,
                                 TaskSpec, reconstruct_location)
from code.comms.protocol import RobotProtocol, RobotState
from code.comms.tests._helpers import FakeActions, FakeClock, pump

P = Performative
DELIVERY_XY = (5.8, -1.0)
LOC = (6.5, 4.7)
CUBES = ({"color_name": "red", "shape_name": "cube"},
         {"color_name": "blue", "shape_name": "cube"},
         {"color_name": "yellow", "shape_name": "cube"})


def _task(query: ObjectQuery, requester: str = "user") -> TaskSpec:
    return TaskSpec(TaskKind.FETCH, query, "delivery pad", DELIVERY_XY,
                    requester=requester)


def _seq(bus: MessageBus) -> List[Tuple[str, str, Performative]]:
    return [(m.sender, m.recipient, m.performative) for m in bus.transcript]


class _UserFixture:
    """One owner ``Alpha`` (+ optional peers) whose ``user`` inbox is drained by a
    scripted reply queue, exactly as :class:`~code.fleet.mission.MissionRunner`
    does: every drained CLARIFY is answered by the next reply."""

    def __init__(self, actions: Dict[str, FakeActions], peers=(), *,
                 clarify_deadline=50, replies: List[ObjectQuery] = None) -> None:
        self.clock = FakeClock()
        self.bus = MessageBus(self.clock.now)
        self.replies = list(replies or [])
        self.protos: Dict[str, RobotProtocol] = {}
        self.protos["Alpha"] = RobotProtocol(
            "Alpha", self.bus, actions["Alpha"], peers,
            clarify_deadline_steps=clarify_deadline)
        for cs in peers:
            others = [c for c in ("Alpha", *peers) if c != cs]
            self.protos[cs] = RobotProtocol(cs, self.bus, actions[cs], others,
                                            clarify_deadline_steps=clarify_deadline)
        self.clarify_count = 0

    def order(self, names):
        return [self.protos[n] for n in names]

    def pump_with_user(self, names, ticks: int) -> None:
        """Step the protocols and answer CLARIFYs from the reply queue each tick."""
        protos = self.order(names)
        for _ in range(ticks):
            for proto in protos:
                proto.step(self.clock.now())
            for m in self.bus.drain("user"):
                if m.performative is P.CLARIFY:
                    self.clarify_count += 1
                    if self.replies:
                        q = self.replies.pop(0)
                        self.bus.post("user", m.sender, P.USER_REPLY, {"query": q})
            self.clock.tick()


class TestClarifyHappyPath(unittest.TestCase):
    """Ambiguous "the cube" -> CLARIFY -> user picks -> normal fetch to completion."""

    def test_clarify_then_fetch(self) -> None:
        actions = {"Alpha": FakeActions(static_location=LOC, manifest=CUBES)}
        fx = _UserFixture(actions, replies=[ObjectQuery("red", "cube")])
        fx.bus.post("user", "Alpha", P.REQUEST_TASK,
                    {"task": _task(ObjectQuery(None, "cube"))})
        fx.pump_with_user(["Alpha"], 12)

        self.assertEqual(_seq(fx.bus), [
            ("user", "Alpha", P.REQUEST_TASK),
            ("Alpha", "user", P.CLARIFY),
            ("user", "Alpha", P.USER_REPLY),
            ("Alpha", "user", P.STATUS_UPDATE),   # found / en route
            ("Alpha", "user", P.STATUS_UPDATE),   # picked up / delivering
            ("Alpha", "user", P.TASK_COMPLETE),
        ])
        clar = next(m for m in fx.bus.transcript if m.performative is P.CLARIFY)
        self.assertEqual(clar.payload["options"], ["red cube", "blue cube", "yellow cube"])
        self.assertIn("which one do you mean?", clar.payload["question"])
        self.assertEqual(fx.protos["Alpha"].last_result, "complete")


class TestClarifyReask(unittest.TestCase):
    """A still-ambiguous reply re-clarifies and consumes the next reply."""

    def test_partial_reply_reasks(self) -> None:
        # First reply "cube" stays ambiguous; second "red" disambiguates.
        actions = {"Alpha": FakeActions(static_location=LOC, manifest=CUBES)}
        fx = _UserFixture(actions,
                          replies=[ObjectQuery(None, "cube"), ObjectQuery("red", None)])
        fx.bus.post("user", "Alpha", P.REQUEST_TASK,
                    {"task": _task(ObjectQuery(None, "cube"))})
        fx.pump_with_user(["Alpha"], 16)

        perfs = [m.performative for m in fx.bus.transcript]
        self.assertEqual(perfs.count(P.CLARIFY), 2)      # asked twice
        self.assertEqual(perfs.count(P.USER_REPLY), 2)   # answered twice
        self.assertEqual(fx.protos["Alpha"].last_result, "complete")


class TestClarifyExhausted(unittest.TestCase):
    """No reply left -> deadline -> TASK_FAILED('need clarification')."""

    def test_unanswered_clarify_fails(self) -> None:
        actions = {"Alpha": FakeActions(static_location=LOC, manifest=CUBES)}
        fx = _UserFixture(actions, clarify_deadline=6, replies=[])
        fx.bus.post("user", "Alpha", P.REQUEST_TASK,
                    {"task": _task(ObjectQuery(None, "cube"))})
        fx.pump_with_user(["Alpha"], 12)

        failed = [m for m in fx.bus.transcript if m.performative is P.TASK_FAILED]
        self.assertEqual(len(failed), 1)
        self.assertEqual(failed[0].payload["reason"], "need clarification")
        self.assertTrue(fx.protos["Alpha"].is_idle())
        self.assertEqual(fx.protos["Alpha"].last_result, "failed")


class TestUnambiguousNoClarify(unittest.TestCase):
    """A specific referent against the manifest never clarifies (byte-identical)."""

    def test_specific_query_skips_clarify(self) -> None:
        actions = {"Alpha": FakeActions(static_location=LOC, manifest=CUBES)}
        fx = _UserFixture(actions, replies=[])
        fx.bus.post("user", "Alpha", P.REQUEST_TASK,
                    {"task": _task(ObjectQuery("red", "cube"))})
        fx.pump_with_user(["Alpha"], 12)

        self.assertNotIn(P.CLARIFY, [m.performative for m in fx.bus.transcript])
        self.assertEqual(fx.protos["Alpha"].last_result, "complete")


class TestSequentialLegs(unittest.TestCase):
    """Two legs in one mission: per-leg milestones, TASK_COMPLETE only after leg B."""

    def test_two_legs_one_completion(self) -> None:
        actions = {"Alpha": FakeActions(static_location=LOC)}
        clock = FakeClock()
        bus = MessageBus(clock.now)
        alpha = RobotProtocol("Alpha", bus, actions["Alpha"], ())
        leg_a = _task(ObjectQuery("red", "cube"))
        leg_b = _task(ObjectQuery("yellow", "cylinder"))
        bus.post("user", "Alpha", P.REQUEST_TASK, {"task": leg_a, "legs": (leg_b,)})
        pump(clock, [alpha], 20)

        perfs = [m.performative for m in bus.transcript]
        self.assertEqual(perfs.count(P.TASK_COMPLETE), 1)   # only the last leg
        statuses = [m.payload["text"] for m in bus.transcript
                    if m.performative is P.STATUS_UPDATE]
        # A transition milestone names both objects; the final completion is leg B.
        self.assertTrue(any("now fetching the yellow cylinder" in s for s in statuses))
        complete = next(m for m in bus.transcript if m.performative is P.TASK_COMPLETE)
        self.assertIn("yellow cylinder", complete.payload["text"])
        # Both objects were actually picked up (one per leg).
        self.assertEqual(len(actions["Alpha"].calls("pickup")), 2)
        self.assertEqual(actions["Alpha"].calls("deliver"), [DELIVERY_XY, DELIVERY_XY])


class TestRetaskMidNavigation(unittest.TestCase):
    """Re-task while navigating to object A redirects the owner onto object B."""

    def test_retask_switches_target(self) -> None:
        actions = {"Alpha": FakeActions(static_location=LOC, arrives=False)}
        clock = FakeClock()
        bus = MessageBus(clock.now)
        alpha = RobotProtocol("Alpha", bus, actions["Alpha"], ())
        bus.post("user", "Alpha", P.REQUEST_TASK,
                 {"task": _task(ObjectQuery("red", "cube"))})
        pump(clock, [alpha], 3)
        self.assertEqual(alpha.state, RobotState.OWNER_NAVIGATING)
        self.assertEqual(alpha.current_task.query, ObjectQuery("red", "cube"))

        NEW = (1.0, 2.0)
        actions["Alpha"].static_location = NEW
        self.assertTrue(alpha.retask(_task(ObjectQuery("yellow", "cylinder")),
                                     clock.now()))
        pump(clock, [alpha], 2)
        self.assertEqual(alpha.current_task.query, ObjectQuery("yellow", "cylinder"))
        # A re-task milestone was announced and the new goal was navigated to.
        self.assertTrue(any(m.performative is P.STATUS_UPDATE
                            and "switching to the yellow cylinder"
                            in m.payload.get("text", "") for m in bus.transcript))
        self.assertIn(NEW, actions["Alpha"].calls("goto"))


class TestRetaskMidDelegation(unittest.TestCase):
    """Re-task while delegating a search stands the helpers down (cancel each)."""

    def test_retask_stands_down_searchers(self) -> None:
        peers = ("Bravo", "Charlie", "Delta")
        actions = {"Alpha": FakeActions(),  # sees nothing -> delegates
                   "Bravo": FakeActions(), "Charlie": FakeActions(),
                   "Delta": FakeActions()}
        clock = FakeClock()
        bus = MessageBus(clock.now)
        protos = {"Alpha": RobotProtocol("Alpha", bus, actions["Alpha"], peers,
                                         search_regions=("north", "middle", "south"))}
        for cs in peers:
            others = [c for c in ("Alpha", *peers) if c != cs]
            protos[cs] = RobotProtocol(cs, bus, actions[cs], others,
                                       search_regions=("north", "middle", "south"))
        order = [protos[n] for n in ("Alpha", *peers)]
        bus.post("user", "Alpha", P.REQUEST_TASK,
                 {"task": _task(ObjectQuery("red", "cube"))})
        pump(clock, order, 10)
        self.assertEqual(protos["Alpha"].state, RobotState.OWNER_DELEGATING)

        protos["Alpha"].retask(_task(ObjectQuery("blue", "ball")), clock.now())
        pump(clock, order, 6)
        # Every commanded searcher received a stand-down (cancel) COMMAND_SEARCH.
        cancels = {m.recipient for m in bus.transcript
                   if m.performative is P.COMMAND_SEARCH and m.payload.get("cancel")}
        self.assertTrue({"Bravo", "Charlie", "Delta"}.issubset(cancels))
        self.assertEqual(protos["Alpha"].current_task.query, ObjectQuery("blue", "ball"))


class TestConcurrentNeedToKnow(unittest.TestCase):
    """Two owners, shared searchers: owner A never learns owner B's find."""

    def _run(self):
        red, blue = (6.5, 4.7), (-6.5, -4.7)
        actions = {
            "Alpha": FakeActions(),   # owner of the red cube (sees nothing at start)
            "Bravo": FakeActions(),   # owner of the blue ball
            # Searchers find their own commander's object once searching.
            "Charlie": FakeActions(search_location=red, find_after_polls=0),
            "Delta": FakeActions(search_location=blue, find_after_polls=0),
        }
        clock = FakeClock()
        bus = MessageBus(clock.now)
        protos: Dict[str, RobotProtocol] = {}
        allcs = ("Alpha", "Bravo", "Charlie", "Delta")
        for cs in allcs:
            peers = [c for c in allcs if c != cs]
            protos[cs] = RobotProtocol(cs, bus, actions[cs], peers,
                                       search_regions=("north",))
        order = [protos[n] for n in allcs]
        bus.post("user", "Alpha", P.REQUEST_TASK,
                 {"task": _task(ObjectQuery("red", "cube"))})
        bus.post("user", "Bravo", P.REQUEST_TASK,
                 {"task": _task(ObjectQuery("blue", "ball"))})
        pump(clock, order, 40)
        return bus, protos, red, blue

    def test_finds_are_partitioned(self) -> None:
        bus, protos, red, blue = self._run()
        founds = [m for m in bus.transcript if m.performative is P.REPORT_FOUND]
        self.assertGreaterEqual(len(founds), 2)  # each owner got at least one find
        for m in founds:
            # REPORT_FOUND only ever goes to an OWNER, never a searcher.
            self.assertIn(m.recipient, ("Alpha", "Bravo"))
            loc = reconstruct_location(m.payload)
            # Need-to-know: each owner ONLY learns its OWN object's location.
            if m.recipient == "Alpha":
                self.assertEqual(loc, red)
            else:
                self.assertEqual(loc, blue)

    def test_owner_a_never_sees_b_find(self) -> None:
        bus, protos, red, blue = self._run()
        # Every message Alpha can observe (addressed to Alpha) that carries a
        # location reconstructs to Alpha's object — never Bravo's.
        for m in bus.transcript:
            if m.recipient == "Alpha" and "rel_offset" in m.payload:
                self.assertEqual(reconstruct_location(m.payload), red)
            if m.recipient == "Bravo" and "rel_offset" in m.payload:
                self.assertEqual(reconstruct_location(m.payload), blue)


if __name__ == "__main__":
    unittest.main()
