"""Unit tests for code.comms.messages (validation, ObjectQuery, TaskSpec)."""

from __future__ import annotations

import unittest

from code.comms.messages import (
    Message,
    ObjectQuery,
    Performative,
    TaskKind,
    TaskSpec,
    reconstruct_location,
    relative_report_payload,
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

    def test_is_generic(self) -> None:  # F4
        self.assertTrue(ObjectQuery(None, None).is_generic)
        self.assertTrue(ObjectQuery().is_generic)
        self.assertFalse(ObjectQuery("red", None).is_generic)
        self.assertFalse(ObjectQuery(None, "cube").is_generic)
        # A generic query matches every object (first found wins).
        self.assertTrue(ObjectQuery(None, None).matches(
            {"color_name": "orange", "shape_name": "cone"}))


class TestRelativeReports(unittest.TestCase):  # F3
    def test_roundtrip_reconstructs_absolute(self) -> None:
        payload = relative_report_payload((2.0, -1.0), "storage A", (6.5, 4.7),
                                          extra={"object": ObjectQuery("red", "cube")})
        self.assertEqual(payload["reporter_pose"], (2.0, -1.0))
        self.assertEqual(payload["room"], "storage A")
        self.assertEqual(payload["rel_offset"], (4.5, 5.7))
        self.assertNotIn("location", payload)  # no transmitted absolute
        x, y = reconstruct_location(payload)
        self.assertAlmostEqual(x, 6.5)
        self.assertAlmostEqual(y, 4.7)


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
    def test_report_found_requires_relative_fields(self) -> None:
        # F3: REPORT_FOUND carries a relative report (object + reporter_pose +
        # rel_offset), never a transmitted absolute location.
        with self.assertRaises(ValueError):
            _msg(Performative.REPORT_FOUND, {"object": ObjectQuery("red", "cube")})
        with self.assertRaises(ValueError):
            _msg(Performative.REPORT_FOUND,
                 {"reporter_pose": (0.0, 0.0), "rel_offset": (1.0, 2.0)})
        # All three present -> valid.
        _msg(Performative.REPORT_FOUND,
             {"object": ObjectQuery("red", "cube"),
              "reporter_pose": (0.0, 0.0), "rel_offset": (1.0, 2.0)})

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
