"""Unit tests for the persistent transcript log (incremental fetch + kinds)."""

from __future__ import annotations

import unittest

from code.apps.fleet_web.transcript import TranscriptLog, kind_for


class KindForTest(unittest.TestCase):
    def test_sender_categories(self):
        self.assertEqual(kind_for("you"), "user")
        self.assertEqual(kind_for("user"), "user")
        self.assertEqual(kind_for("allocator"), "allocator")
        self.assertEqual(kind_for("system"), "system")
        self.assertEqual(kind_for("Alpha"), "robot")


class TranscriptLogTest(unittest.TestCase):
    def setUp(self):
        self.log = TranscriptLog()

    def test_append_assigns_monotonic_ids(self):
        a = self.log.append("you", "Alpha", "hi", "user")
        b = self.log.append("Alpha", "user", "ok", "robot")
        self.assertEqual(a.id, 1)
        self.assertEqual(b.id, 2)
        self.assertEqual(self.log.last_id, 2)
        self.assertEqual(len(self.log), 2)

    def test_since_returns_only_newer(self):
        for i in range(5):
            self.log.append("Alpha", "user", f"line{i}", "robot")
        newer = self.log.since(2)
        self.assertEqual([e.id for e in newer], [3, 4, 5])

    def test_since_zero_returns_all(self):
        self.log.append("Alpha", "user", "x", "robot")
        self.assertEqual(len(self.log.since(0)), 1)

    def test_dicts_since_shape(self):
        self.log.append("Bravo", "user", "found it", "robot")
        d = self.log.dicts_since(0)[0]
        self.assertEqual(set(d), {"id", "sender", "recipient", "text", "kind"})
        self.assertEqual(d["sender"], "Bravo")
        self.assertEqual(d["kind"], "robot")

    def test_default_kind_inferred_from_sender(self):
        e = self.log.append("allocator", "user", "all busy", "")
        self.assertEqual(e.kind, "allocator")


if __name__ == "__main__":
    unittest.main()
