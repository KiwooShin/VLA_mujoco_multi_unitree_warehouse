"""Unit tests for code.comms.messages (validation, ObjectQuery, TaskSpec)."""

from __future__ import annotations

import unittest

from code.comms.messages import (
    Message,
    ObjectQuery,
    Performative,
    TaskKind,
    TaskSpec,
)


def _msg(performative: Performative, payload=None, **kw) -> Message:
    base = dict(msg_id=0, t_step=0, sender="Alpha", recipient="Bravo",
                performative=performative, payload=payload or {})
    base.update(kw)
    return Message(**base)


class TestObjectQuery(unittest.TestCase):
    def test_matches_color_and_shape(self) -> None:
        q = ObjectQuery("red", "cube")
        self.assertTrue(q.matches({"color_name": "red", "shape_name": "cube"}))
        self.assertFalse(q.matches({"color_name": "blue", "shape_name": "cube"}))
        self.assertFalse(q.matches({"color_name": "red", "shape_name": "ball"}))

    def test_wildcards(self) -> None:
        self.assertTrue(ObjectQuery(None, "cube").matches(
            {"color_name": "green", "shape_name": "cube"}))
        self.assertTrue(ObjectQuery("red", None).matches(
            {"color_name": "red", "shape_name": "cone"}))
        self.assertTrue(ObjectQuery(None, None).matches(
            {"color_name": "x", "shape_name": "y"}))

    def test_describe(self) -> None:
        self.assertEqual(ObjectQuery("red", "cube").describe(), "red cube")
        self.assertEqual(ObjectQuery(None, "cube").describe(), "cube")
        self.assertEqual(ObjectQuery("red", None).describe(), "red object")
        self.assertEqual(ObjectQuery(None, None).describe(), "object")

    def test_frozen(self) -> None:
        q = ObjectQuery("red", "cube")
        with self.assertRaises(Exception):
            q.color_name = "blue"  # type: ignore[misc]


class TestTaskSpec(unittest.TestCase):
    def test_describe(self) -> None:
        task = TaskSpec(TaskKind.FETCH, ObjectQuery("red", "cube"),
                        "delivery pad", (5.8, -1.0))
        self.assertEqual(task.describe(), "fetch the red cube to the delivery pad")

    def test_default_requester_is_user(self) -> None:
        task = TaskSpec(TaskKind.FETCH, ObjectQuery("red", "cube"),
                        "delivery pad", (5.8, -1.0))
        self.assertEqual(task.requester, "user")


class TestMessageValidation(unittest.TestCase):
    def test_report_found_requires_object_and_location(self) -> None:
        with self.assertRaises(ValueError):
            _msg(Performative.REPORT_FOUND, {"object": ObjectQuery("red", "cube")})
        with self.assertRaises(ValueError):
            _msg(Performative.REPORT_FOUND, {"location": (1.0, 2.0)})
        # Both present -> valid.
        _msg(Performative.REPORT_FOUND,
             {"object": ObjectQuery("red", "cube"), "location": (1.0, 2.0)})

    def test_required_keys_per_performative(self) -> None:
        cases = {
            Performative.QUERY_VISIBILITY: {"query": ObjectQuery("red", "cube")},
            Performative.REPORT_VISIBILITY: {
                "query": ObjectQuery("red", "cube"), "visible": False},
            Performative.COMMAND_SEARCH: {
                "query": ObjectQuery("red", "cube"), "region": "north"},
            Performative.REJECT: {"reason": "busy"},
            Performative.STATUS_UPDATE: {"text": "hi"},
            Performative.TASK_COMPLETE: {"text": "done"},
            Performative.TASK_FAILED: {"reason": "exhausted"},
        }
        for perf, payload in cases.items():
            _msg(perf, payload)  # valid
            with self.assertRaises(ValueError, msg=f"{perf} should require keys"):
                _msg(perf, {})

    def test_accept_needs_no_payload(self) -> None:
        _msg(Performative.ACCEPT, {})  # no required keys

    def test_bad_performative_type_raises(self) -> None:
        with self.assertRaises(TypeError):
            _msg("QUERY_VISIBILITY")  # type: ignore[arg-type]

    def test_empty_sender_or_recipient_raises(self) -> None:
        with self.assertRaises(ValueError):
            _msg(Performative.ACCEPT, {}, sender="")
        with self.assertRaises(ValueError):
            _msg(Performative.ACCEPT, {}, recipient="")

    def test_payload_is_read_only(self) -> None:
        msg = _msg(Performative.STATUS_UPDATE, {"text": "hi"})
        with self.assertRaises(TypeError):
            msg.payload["text"] = "mutated"  # type: ignore[index]

    def test_in_reply_to_optional(self) -> None:
        msg = _msg(Performative.ACCEPT, {}, in_reply_to=7)
        self.assertEqual(msg.in_reply_to, 7)
        self.assertIsNone(_msg(Performative.ACCEPT, {}).in_reply_to)


if __name__ == "__main__":
    unittest.main()
