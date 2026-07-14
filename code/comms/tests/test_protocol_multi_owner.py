"""Cross-owner searcher-budget tests (concurrent two-owner missions).

Regression coverage for the searcher-starvation gap: two owners running the
delegated-search protocol at the same time must *partition* the free peers
instead of the first owner to delegate grabbing them all. A ``recruitable_hook``
(``None`` -> no limit, byte-identical single-mission behaviour; a set -> this
owner's disjoint share) is the cross-owner budget. These fakes exercise the
protocol mechanics directly:

* two owners + two free peers -> each owner recruits exactly ITS one peer, never
  the other's, and each searcher's find reaches only its commander (need-to-know);
* two owners + one free peer -> one owner searches, the short-handed owner WAITS
  in OWNER_DELEGATING (it does not fail) and recruits the freed robots the moment
  the first mission finishes;
* no hook -> recruitment is unchanged (the historical single-owner path).
"""

from __future__ import annotations

import unittest

from code.comms.bus import MessageBus
from code.comms.messages import ObjectQuery, Performative, TaskKind, TaskSpec
from code.comms.protocol import RobotProtocol, RobotState
from code.comms.tests._helpers import FakeActions, FakeClock, pump

P = Performative
ROSTER = ("Alpha", "Bravo", "Charlie", "Delta")
REGIONS = ("roomA", "roomB", "roomC")
DEST = (4.0, -2.0)
LOC_A = (-9.0, 2.6)   # red cube, deep storage A
LOC_B = (2.5, -2.6)   # blue ball, deep storage B


def _task(color: str, shape: str) -> TaskSpec:
    return TaskSpec(TaskKind.FETCH, ObjectQuery(color, shape),
                    "delivery pad", DEST, requester="user")


def _commands(bus: MessageBus):
    """(sender, recipient) of every non-cancel COMMAND_SEARCH on the bus."""
    return [(m.sender, m.recipient) for m in bus.transcript
            if m.performative is P.COMMAND_SEARCH and not m.payload.get("cancel")]


def _proto(cs: str, bus: MessageBus, actions, *, hook=None) -> RobotProtocol:
    peers = [c for c in ROSTER if c != cs]
    return RobotProtocol(cs, bus, actions, peers, search_regions=REGIONS,
                         reply_deadline_steps=5, search_deadline_steps=4000,
                         recruitable_hook=hook)


class TestTwoOwnersTwoPeers(unittest.TestCase):
    """2 owners / 2 free peers -> one searcher EACH, never the sibling's."""

    def _fixture(self):
        clock = FakeClock()
        bus = MessageBus(clock.now)
        acts = {cs: FakeActions() for cs in ROSTER}
        # Each searcher finds its commander's object once it has searched a while.
        acts["Charlie"] = FakeActions(search_location=LOC_A, find_after_polls=2)
        acts["Delta"] = FakeActions(search_location=LOC_B, find_after_polls=2)
        # Disjoint static partition of the two free peers across the two owners.
        hooks = {"Alpha": lambda: {"Charlie"}, "Bravo": lambda: {"Delta"}}
        protos = {cs: _proto(cs, bus, acts[cs], hook=hooks.get(cs))
                  for cs in ROSTER}
        return clock, bus, protos

    def test_each_owner_recruits_only_its_share(self) -> None:
        clock, bus, protos = self._fixture()
        bus.post("user", "Alpha", P.REQUEST_TASK, {"task": _task("red", "cube")})
        bus.post("user", "Bravo", P.REQUEST_TASK, {"task": _task("blue", "ball")})
        pump(clock, [protos[c] for c in ROSTER], 40)

        cmds = _commands(bus)
        # Alpha only ever commanded Charlie; Bravo only ever commanded Delta.
        self.assertEqual({r for s, r in cmds if s == "Alpha"}, {"Charlie"})
        self.assertEqual({r for s, r in cmds if s == "Bravo"}, {"Delta"})
        # The starvation bug would have Alpha command BOTH free peers.
        self.assertNotIn(("Alpha", "Delta"), cmds)
        self.assertNotIn(("Bravo", "Charlie"), cmds)

    def test_need_to_know_finds_reach_only_commander(self) -> None:
        clock, bus, protos = self._fixture()
        bus.post("user", "Alpha", P.REQUEST_TASK, {"task": _task("red", "cube")})
        bus.post("user", "Bravo", P.REQUEST_TASK, {"task": _task("blue", "ball")})
        pump(clock, [protos[c] for c in ROSTER], 60)

        found = [m for m in bus.transcript if m.performative is P.REPORT_FOUND]
        self.assertTrue(found)
        by_sender = {m.sender: m.recipient for m in found}
        self.assertEqual(by_sender.get("Charlie"), "Alpha")  # Charlie -> Alpha only
        self.assertEqual(by_sender.get("Delta"), "Bravo")    # Delta -> Bravo only
        # No searcher find ever addressed the other owner or the user.
        for m in found:
            self.assertIn(m.recipient, ("Alpha", "Bravo"))
        self.assertNotEqual(by_sender.get("Charlie"), "Bravo")
        self.assertNotEqual(by_sender.get("Delta"), "Alpha")


class TestTwoOwnersOnePeer(unittest.TestCase):
    """2 owners / 1 free peer -> one searches, the other waits then recruits freed."""

    def test_short_handed_owner_waits_then_recovers(self) -> None:
        clock = FakeClock()
        bus = MessageBus(clock.now)
        acts = {cs: FakeActions() for cs in ROSTER[:3]}   # Alpha, Bravo, Charlie
        acts["Charlie"] = FakeActions(search_location=LOC_A, find_after_polls=8)
        roster3 = ROSTER[:3]

        def alpha_hook():
            return {"Charlie"}

        def bravo_hook():
            # Mirrors the runner's recovery: Bravo's pool is empty while Alpha's
            # mission is live, then opens to the freed robots once Alpha finishes.
            if protos["Alpha"].is_idle() and protos["Alpha"].last_result == "complete":
                return {"Charlie", "Alpha"}
            return set()

        def _p(cs, hook=None):
            peers = [c for c in roster3 if c != cs]
            return RobotProtocol(cs, bus, acts[cs], peers, search_regions=REGIONS,
                                 reply_deadline_steps=5, search_deadline_steps=9000,
                                 recruitable_hook=hook)
        protos = {"Alpha": _p("Alpha", alpha_hook),
                  "Bravo": _p("Bravo", bravo_hook), "Charlie": _p("Charlie")}
        order = [protos[c] for c in roster3]

        bus.post("user", "Alpha", P.REQUEST_TASK, {"task": _task("red", "cube")})
        bus.post("user", "Bravo", P.REQUEST_TASK, {"task": _task("blue", "ball")})

        # Phase 1: both delegate. Alpha recruits Charlie; Bravo, budgeted to an
        # empty pool, WAITS (no searcher, but must not have failed).
        pump(clock, order, 10)
        self.assertIs(protos["Bravo"].state, RobotState.OWNER_DELEGATING)
        self.assertIs(protos["Charlie"].state, RobotState.ASSIST_SEARCHING)
        bravo_fail = [m for m in bus.transcript if m.performative is P.TASK_FAILED
                      and m.sender == "Bravo"]
        self.assertEqual(bravo_fail, [])   # it waited, it did not give up
        # Only Alpha ever commanded Charlie so far.
        self.assertEqual({s for s, r in _commands(bus)}, {"Alpha"})

        # Phase 2: let Alpha finish; its freed robots flow into Bravo's pool and
        # Bravo recruits and completes too.
        pump(clock, order, 80)
        self.assertEqual(protos["Alpha"].last_result, "complete")
        completes = {m.sender for m in bus.transcript
                     if m.performative is P.TASK_COMPLETE}
        self.assertEqual(completes, {"Alpha", "Bravo"})
        # Bravo did recruit once freed (a Bravo->Charlie command exists).
        self.assertIn(("Bravo", "Charlie"), _commands(bus))


class TestNoHookUnchanged(unittest.TestCase):
    """No recruitable hook -> the single-owner delegation path is unchanged."""

    def test_owner_commands_every_region_peer(self) -> None:
        clock = FakeClock()
        bus = MessageBus(clock.now)
        acts = {cs: FakeActions() for cs in ROSTER}
        protos = {cs: _proto(cs, bus, acts[cs]) for cs in ROSTER}  # hook=None
        bus.post("user", "Alpha", P.REQUEST_TASK, {"task": _task("red", "cube")})
        pump(clock, [protos[c] for c in ROSTER], 40)

        # Unbudgeted: Alpha zips all three peers onto the three regions (historical).
        cmds = {r for s, r in _commands(bus) if s == "Alpha"}
        self.assertEqual(cmds, {"Bravo", "Charlie", "Delta"})
        for cs in ("Bravo", "Charlie", "Delta"):
            self.assertIs(protos[cs].state, RobotState.ASSIST_SEARCHING)


if __name__ == "__main__":
    unittest.main()
