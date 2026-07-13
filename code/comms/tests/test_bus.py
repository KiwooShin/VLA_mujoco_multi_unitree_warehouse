"""Unit tests for code.comms.bus (ordering, drain, transcript, formatting)."""

from __future__ import annotations

import unittest

from code.comms.bus import MessageBus, format_line
from code.comms.messages import ObjectQuery, Performative, TaskKind, TaskSpec
from code.comms.tests._helpers import FakeClock


class TestBusRouting(unittest.TestCase):
    def setUp(self) -> None:
        self.clock = FakeClock()
        self.bus = MessageBus(self.clock.now, allocator_inbox="allocator")

    def test_msg_ids_monotonic_and_stamped(self) -> None:
        self.clock.t = 5
        a = self.bus.post("Alpha", "Bravo", Performative.ACCEPT, {})
        self.clock.t = 9
        b = self.bus.post("Alpha", "Bravo", Performative.ACCEPT, {})
        self.assertEqual((a.msg_id, a.t_step), (0, 5))
        self.assertEqual((b.msg_id, b.t_step), (1, 9))

    def test_drain_is_fifo_and_clears(self) -> None:
        self.bus.post("Alpha", "Bravo", Performative.REJECT, {"reason": "busy"})
        self.bus.post("Charlie", "Bravo", Performative.ACCEPT, {})
        drained = self.bus.drain("Bravo")
        self.assertEqual([m.sender for m in drained], ["Alpha", "Charlie"])
        self.assertEqual(self.bus.drain("Bravo"), [])  # inbox now empty

    def test_per_recipient_isolation(self) -> None:
        self.bus.post("Alpha", "Bravo", Performative.ACCEPT, {})
        self.bus.post("Alpha", "Charlie", Performative.ACCEPT, {})
        self.assertEqual(len(self.bus.drain("Bravo")), 1)
        self.assertEqual(len(self.bus.drain("Charlie")), 1)
        self.assertEqual(self.bus.drain("Delta"), [])

    def test_fleet_routes_to_allocator_inbox(self) -> None:
        task = TaskSpec(TaskKind.FETCH, ObjectQuery("red", "cube"),
                        "delivery pad", (5.8, -1.0))
        self.bus.post("user", "fleet", Performative.FLEET_REQUEST, {"task": task})
        # No robot receives it; it lands in the allocator inbox.
        self.assertEqual(self.bus.drain("Alpha"), [])
        self.assertEqual(self.bus.pending("allocator"), 1)
        via_fleet = self.bus.drain("fleet")  # "fleet" alias -> allocator inbox
        self.assertEqual(len(via_fleet), 1)
        self.assertEqual(via_fleet[0].performative, Performative.FLEET_REQUEST)

    def test_transcript_is_ordered_and_readonly(self) -> None:
        self.bus.post("Alpha", "Bravo", Performative.ACCEPT, {})
        self.bus.post("Bravo", "Alpha", Performative.REJECT, {"reason": "busy"})
        tx = self.bus.transcript
        self.assertEqual([m.msg_id for m in tx], [0, 1])
        self.assertIsInstance(tx, tuple)  # snapshot, not the live list


class TestFormatting(unittest.TestCase):
    def setUp(self) -> None:
        self.clock = FakeClock()
        self.bus = MessageBus(self.clock.now)
        self.q = ObjectQuery("red", "cube")

    def _line(self, sender, recipient, perf, payload) -> str:
        self.clock.t = 1200
        return format_line(self.bus.post(sender, recipient, perf, payload))

    def test_query_line(self) -> None:
        line = self._line("Alpha", "Bravo", Performative.QUERY_VISIBILITY,
                          {"query": self.q})
        self.assertEqual(
            line, "t=1200 Alpha->Bravo QUERY_VISIBILITY: can you see the red cube?")

    def test_report_visibility_lines(self) -> None:
        # F3: a positive sighting renders the fixed relative-position sentence.
        yes = self._line("Bravo", "Alpha", Performative.REPORT_VISIBILITY,
                         {"query": self.q, "visible": True,
                          "reporter_pose": (-2.0, 0.5), "room": "storage A",
                          "rel_offset": (3.5, 4.2)})
        self.assertEqual(
            yes, "t=1200 Bravo->Alpha REPORT_VISIBILITY: "
                 "I am robot Bravo, currently in storage A at position "
                 "(-2.0, 0.5). The object is located 3.5 m and 4.2 m away from me.")
        no = self._line("Charlie", "Alpha", Performative.REPORT_VISIBILITY,
                        {"query": self.q, "visible": False})
        self.assertEqual(
            no, "t=1200 Charlie->Alpha REPORT_VISIBILITY: no, I can't see the red cube")

    def test_command_search_and_cancel_lines(self) -> None:
        start = self._line("Alpha", "Charlie", Performative.COMMAND_SEARCH,
                           {"query": self.q, "region": "north", "cancel": False})
        self.assertIn("search the north area for the red cube", start)
        cancel = self._line("Alpha", "Charlie", Performative.COMMAND_SEARCH,
                            {"query": self.q, "region": "north", "cancel": True})
        self.assertIn("stand down — the red cube has been found", cancel)

    def test_report_found_line(self) -> None:
        # F3: REPORT_FOUND also renders the fixed relative-position sentence.
        line = self._line("Bravo", "Alpha", Performative.REPORT_FOUND,
                          {"object": self.q, "reporter_pose": (5.0, 1.0),
                           "room": "back room", "rel_offset": (1.5, 3.7)})
        self.assertEqual(
            line, "t=1200 Bravo->Alpha REPORT_FOUND: "
                  "I am robot Bravo, currently in back room at position "
                  "(5.0, 1.0). The object is located 1.5 m and 3.7 m away from me.")

    def test_request_and_fleet_lines(self) -> None:
        task = TaskSpec(TaskKind.FETCH, self.q, "delivery pad", (5.8, -1.0))
        req = self._line("user", "Alpha", Performative.REQUEST_TASK, {"task": task})
        self.assertEqual(
            req, "t=1200 user->Alpha REQUEST_TASK: "
                 "fetch the red cube to the delivery pad")
        fleet = self._line("user", "fleet", Performative.FLEET_REQUEST, {"task": task})
        self.assertIn("any robot: fetch the red cube to the delivery pad", fleet)

    def test_transcript_lines_last_n(self) -> None:
        for _ in range(3):
            self.bus.post("Alpha", "Bravo", Performative.ACCEPT, {})
        self.assertEqual(len(self.bus.transcript_lines()), 3)
        self.assertEqual(len(self.bus.transcript_lines(last_n=2)), 2)

    def test_formatting_is_deterministic(self) -> None:
        payload = {"query": self.q, "visible": True, "reporter_pose": (1.5, 0.0),
                   "room": "middle area", "rel_offset": (2.0, 1.0)}
        a = self._line("Bravo", "Alpha", Performative.REPORT_VISIBILITY, payload)
        b = self._line("Bravo", "Alpha", Performative.REPORT_VISIBILITY, payload)
        self.assertEqual(a, b)


if __name__ == "__main__":
    unittest.main()
