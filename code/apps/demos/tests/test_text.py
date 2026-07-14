"""Transcript sanitising, word-wrap and generic (clarify-aware) flattening."""

from __future__ import annotations

import types
import unittest

from code.apps.demos import style
from code.apps.demos.text import (ascii_sanitize, flatten_transcript,
                                  message_phrase, tail_lines, wrap)
from code.comms.bus import MessageBus
from code.comms.messages import Performative


def _bus_with_status():
    bus = MessageBus(lambda: 42)
    bus.post("Alpha", "user", Performative.STATUS_UPDATE,
             {"text": "delivered the red cube"})
    return bus


class TestText(unittest.TestCase):
    def test_ascii_sanitize_maps_and_drops(self):
        self.assertEqual(ascii_sanitize("a—b’s “q” →x"),
                         'a-b\'s "q" ->x')
        self.assertEqual(ascii_sanitize("café ☺"), "caf ")

    def test_wrap_respects_width_and_splits_long_tokens(self):
        lines = wrap("the quick brown fox jumps", 9)
        self.assertTrue(all(len(x) <= 9 for x in lines))
        self.assertEqual(" ".join(lines).split(),
                         "the quick brown fox jumps".split())
        long = wrap("x" * 25, 10)
        self.assertTrue(all(len(x) <= 10 for x in long))
        self.assertEqual("".join(long), "x" * 25)

    def test_known_performative_uses_project_phrasing(self):
        msg = _bus_with_status().transcript[0]
        self.assertEqual(message_phrase(msg), "delivered the red cube")

    def test_unknown_performative_falls_back_to_payload_text(self):
        # A new clarify flow the composer has never seen: unknown performative,
        # question carried in the payload — must still surface in the panel.
        clarify = types.SimpleNamespace(
            sender="Alpha", recipient="user",
            performative=types.SimpleNamespace(name="CLARIFY_QUERY"),
            payload={"question": "which cube: red, blue, or yellow?"}, t_step=5)
        self.assertEqual(message_phrase(clarify),
                         "which cube: red, blue, or yellow?")

    def test_flatten_transcript_colors_and_wraps(self):
        bus = _bus_with_status()
        bus.post("user", "Alpha", Performative.REQUEST_TASK,
                 {"task": types.SimpleNamespace(describe=lambda: "fetch the red cube")})
        lines = flatten_transcript(bus.transcript, width_chars=40)
        kinds = [k for k, _, _ in lines]
        self.assertEqual(kinds.count("head"), 2)     # one head per message
        self.assertIn("body", kinds)
        head_alpha = next(l for l in lines
                          if l[0] == "head" and l[2].startswith("Alpha"))
        self.assertEqual(head_alpha[1], style.accent_bgr("Alpha"))

    def test_tail_lines_keeps_newest(self):
        lines = [("body", (0, 0, 0), str(i)) for i in range(10)]
        self.assertEqual([t[2] for t in tail_lines(lines, 3)], ["7", "8", "9"])
        self.assertEqual(tail_lines(lines, 0), [])


if __name__ == "__main__":
    unittest.main()
