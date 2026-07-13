"""Six-robot coordination-protocol tests (callsign-agnostic delegation).

Proves the delegated-search protocol scales past four robots: with FIVE idle
peers and only THREE searchable rooms, the owner commands exactly three peers
(one per room) and keeps the surplus two in reserve; when a commanded searcher
declines, a reserve peer picks up the freed region. No behaviour here is
hard-wired to a four-robot roster — everything derives from the peer list and
the region labels handed in.
"""

from __future__ import annotations

import unittest

from code.comms.bus import MessageBus
from code.comms.messages import (ObjectQuery, Performative, TaskKind, TaskSpec)
from code.comms.protocol import RobotProtocol, RobotState
from code.comms.tests._helpers import FakeActions, FakeClock, pump

P = Performative
# Six-robot roster; three searchable rooms (rooms6 minus the loading room bays).
CALLSIGNS6 = ("Alpha", "Bravo", "Charlie", "Delta", "Echo", "Foxtrot")
PEERS5 = ("Bravo", "Charlie", "Delta", "Echo", "Foxtrot")
ROOMS3 = ("storage A", "storage B", "back room")
DELIVERY_XY = (4.0, -2.0)
LOC = (10.5, 6.5)  # deep back-room corner


def _task() -> TaskSpec:
    return TaskSpec(TaskKind.FETCH, ObjectQuery("red", "cube"),
                    "delivery pad", DELIVERY_XY, requester="user")


def _commands(bus: MessageBus, *, cancel: bool):
    """(recipient, region) of every COMMAND_SEARCH with the given cancel flag."""
    return [(m.recipient, m.payload.get("region")) for m in bus.transcript
            if m.performative is P.COMMAND_SEARCH
            and bool(m.payload.get("cancel")) == cancel]


class _Fixture:
    """Owner Alpha + five peers on one bus, given three searchable regions."""

    def __init__(self, actions: dict) -> None:
        self.clock = FakeClock()
        self.bus = MessageBus(self.clock.now)
        kw = dict(reply_deadline_steps=50, search_deadline_steps=4000,
                  search_regions=ROOMS3)
        self.protos = {
            "Alpha": RobotProtocol("Alpha", self.bus, actions["Alpha"],
                                   list(PEERS5), **kw)}
        for cs in PEERS5:
            others = [c for c in CALLSIGNS6 if c != cs]
            self.protos[cs] = RobotProtocol(cs, self.bus, actions[cs], others, **kw)

    def order(self):
        return [self.protos[n] for n in CALLSIGNS6]


class TestFivePeersThreeRooms(unittest.TestCase):
    def test_three_commanded_two_reserve(self) -> None:
        # Nobody sees the object -> Alpha queries all five peers then delegates.
        actions = {cs: FakeActions() for cs in CALLSIGNS6}
        fx = _Fixture(actions)
        fx.bus.post("user", "Alpha", P.REQUEST_TASK, {"task": _task()})
        pump(fx.clock, fx.order(), 60)

        commanded = _commands(fx.bus, cancel=False)
        # Exactly three initial COMMAND_SEARCHes, one region each, in order.
        self.assertEqual(len(commanded), 3)
        self.assertEqual(commanded,
                         [("Bravo", "storage A"), ("Charlie", "storage B"),
                          ("Delta", "back room")])
        # The two surplus peers are reserve: never commanded.
        commanded_names = {r for r, _ in commanded}
        self.assertNotIn("Echo", commanded_names)
        self.assertNotIn("Foxtrot", commanded_names)
        # Owner is delegating; the three commanded peers accepted and search.
        self.assertIs(fx.protos["Alpha"].state, RobotState.OWNER_DELEGATING)
        for cs in ("Bravo", "Charlie", "Delta"):
            self.assertIs(fx.protos[cs].state, RobotState.ASSIST_SEARCHING)

    def test_reserve_peer_covers_a_declined_region(self) -> None:
        # Charlie's region has no coverable patrol -> Charlie REJECTs; the owner
        # re-plans that region onto the first reserve peer (Echo).
        actions = {cs: FakeActions() for cs in CALLSIGNS6}
        actions["Charlie"] = FakeActions(can_search=False)
        fx = _Fixture(actions)
        fx.bus.post("user", "Alpha", P.REQUEST_TASK, {"task": _task()})
        pump(fx.clock, fx.order(), 80)

        # Charlie declined; a reserve peer took over storage B.
        rejects = [m.sender for m in fx.bus.transcript
                   if m.performative is P.REJECT]
        self.assertIn("Charlie", rejects)
        commanded = _commands(fx.bus, cancel=False)
        # storage B is re-commanded to Echo (first reserve peer).
        echo_cmds = [(r, reg) for r, reg in commanded if r == "Echo"]
        self.assertEqual(echo_cmds, [("Echo", "storage B")])

    def test_found_by_reserve_takeover_completes_delegation(self) -> None:
        # The reserve peer that takes over the freed region finds the object,
        # reporting it straight to the owner (need-to-know intact at N=6).
        actions = {cs: FakeActions() for cs in CALLSIGNS6}
        actions["Delta"] = FakeActions(can_search=False)  # back room declines
        actions["Echo"] = FakeActions(search_location=LOC, find_after_polls=1)
        fx = _Fixture(actions)
        fx.bus.post("user", "Alpha", P.REQUEST_TASK, {"task": _task()})
        pump(fx.clock, fx.order(), 200)

        found = [m for m in fx.bus.transcript
                 if m.performative is P.REPORT_FOUND]
        self.assertTrue(found)
        for m in found:
            self.assertEqual(m.recipient, "Alpha")  # only the owner is told
        # Owner advanced past delegation on the reserve peer's find.
        self.assertNotIn(fx.protos["Alpha"].state,
                         (RobotState.OWNER_QUERYING, RobotState.OWNER_DELEGATING))


if __name__ == "__main__":
    unittest.main()
