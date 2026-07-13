"""Tests for FleetService: queueing, busy-notice, snapshots, transcript, loop."""

from __future__ import annotations

import time
import unittest
from typing import Callable

from code.apps.fleet_web.service import FleetService
from code.apps.fleet_web.tests.fakes import FakeEngine


def _wait_until(pred: Callable[[], bool], timeout: float = 5.0) -> bool:
    """Poll ``pred`` until true or ``timeout`` (seconds); return the last value."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        if pred():
            return True
        time.sleep(0.01)
    return pred()


class ServiceCommandLogicTest(unittest.TestCase):
    """Deterministic (no-thread) tests of intake/queueing/snapshots."""

    def setUp(self):
        self.svc = FleetService(FakeEngine())

    def test_reject_paths(self):
        self.assertFalse(self.svc.submit_command("")["ok"])
        self.assertFalse(self.svc.submit_command("Zulu, fetch red cube")["ok"])
        bad = self.svc.submit_command("Alpha, dance around")
        self.assertFalse(bad["ok"])
        self.assertIn("error", bad)

    def test_immediate_accept_when_idle(self):
        res = self.svc.submit_command("Alpha, fetch the red cube to the pad")
        self.assertTrue(res["ok"])
        self.assertFalse(res["queued"])
        self.assertIn("Order sent", res["message"])

    def test_busy_queues_with_notice(self):
        # Simulate a mission already in flight.
        self.svc._active = True
        res = self.svc.submit_command("Bravo, bring the blue ball to the pad")
        self.assertTrue(res["ok"])
        self.assertTrue(res["queued"])
        self.assertIn("queued", res["message"].lower())
        # A second one is now two orders deep.
        res2 = self.svc.submit_command("Delta, fetch the green cylinder")
        self.assertIn("2 ahead", res2["message"])
        # Both queue notices were logged to the transcript.
        lines = self.svc.snapshot_state(0)["transcript"]
        systems = [x for x in lines if x["kind"] == "system"]
        self.assertEqual(len(systems), 2)

    def test_snapshot_state_shape(self):
        state = self.svc.snapshot_state(0)
        self.assertEqual(set(state),
                         {"robots", "transcript", "next_id", "status", "mission"})
        self.assertEqual(set(state["mission"]),
                         {"active", "recipient", "target", "phase", "outcome",
                          "on_pad", "queued"})

    def test_transcript_incremental_fetch(self):
        self.svc._active = True
        self.svc.submit_command("Alpha, fetch the red cube")
        self.svc.submit_command("Bravo, fetch the blue ball")
        first_id = self.svc.snapshot_state(0)["transcript"][0]["id"]
        newer = self.svc.snapshot_state(first_id)["transcript"]
        self.assertTrue(all(m["id"] > first_id for m in newer))
        self.assertEqual(len(newer), 1)


class ServiceLoopTest(unittest.TestCase):
    """End-to-end drive of the worker loop with the scripted fake engine."""

    def setUp(self):
        self.engine = FakeEngine(mission_steps=6)
        self.svc = FleetService(self.engine, target_fps=200.0,
                                steps_per_frame=1, max_steps=50)
        self.svc.start()
        self.addCleanup(self.svc.stop)
        self.assertTrue(_wait_until(
            lambda: "ready" in self.svc.snapshot_state(0)["status"]))

    def test_full_mission_cycle(self):
        self.assertTrue(self.svc.submit_command(
            "Alpha, fetch the red cube to the delivery pad")["ok"])
        self.assertTrue(_wait_until(
            lambda: self.svc.snapshot_state(0)["mission"]["outcome"] == "complete"))
        state = self.svc.snapshot_state(0)
        senders = {m["sender"] for m in state["transcript"]}
        # The raw order is echoed as "you"; the runner's own user-request line
        # is filtered out; the owner's chatter + a system summary are present.
        self.assertIn("you", senders)
        self.assertNotIn("user", senders)
        self.assertIn("Alpha", senders)
        self.assertTrue(any(m["kind"] == "system" and "Delivered" in m["text"]
                            for m in state["transcript"]))
        self.assertEqual(len(state["robots"]), 4)
        self.assertTrue(state["mission"]["on_pad"])
        self.assertIsNotNone(self.svc.latest_jpeg())

    def test_sequential_missions_accumulate_transcript(self):
        self.svc.submit_command("Alpha, fetch the red cube to the pad")
        self.assertTrue(_wait_until(
            lambda: self.svc.snapshot_state(0)["mission"]["outcome"] == "complete"))
        after_first = self.svc.snapshot_state(0)["next_id"]

        self.svc.submit_command("someone bring me the blue ball")
        self.assertTrue(_wait_until(
            lambda: self.svc.snapshot_state(0)["next_id"] > after_first
            and self.svc.snapshot_state(0)["mission"]["outcome"] == "complete"
            and not self.svc.snapshot_state(0)["mission"]["active"]))
        # A rebuild happened for the 2nd order (initial reset + 1 rebuild).
        self.assertGreaterEqual(self.engine.reset_count, 2)
        # Incremental fetch returns only the 2nd mission's lines.
        newer = self.svc.snapshot_state(after_first)["transcript"]
        self.assertTrue(newer)
        self.assertTrue(all(m["id"] > after_first for m in newer))
        self.assertTrue(any("blue ball" in m["text"] for m in newer))


if __name__ == "__main__":
    unittest.main()
