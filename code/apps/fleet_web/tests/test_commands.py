"""Unit tests for command validation (code.apps.fleet_web.commands)."""

from __future__ import annotations

import unittest

from code.apps.fleet_web.commands import (CommandCheck, example_commands,
                                          validate_command)

_CALLSIGNS = ("Alpha", "Bravo", "Charlie", "Delta")


class ValidateCommandTest(unittest.TestCase):
    def _check(self, text: str) -> CommandCheck:
        return validate_command(text, _CALLSIGNS)

    def test_named_callsign_with_object(self):
        c = self._check("Alpha, fetch the red cube to the delivery pad")
        self.assertTrue(c.ok)
        self.assertFalse(c.is_fleet)
        self.assertEqual(c.recipient, "Alpha")
        self.assertEqual(c.recipient_label, "Alpha")
        self.assertEqual(c.target_desc, "red cube")

    def test_fleet_word_routes_to_allocator(self):
        c = self._check("someone bring me the blue ball")
        self.assertTrue(c.ok)
        self.assertTrue(c.is_fleet)
        self.assertEqual(c.recipient, "fleet")
        self.assertEqual(c.recipient_label, "the fleet")
        self.assertEqual(c.target_desc, "blue ball")

    def test_bare_imperative_is_fleet(self):
        c = self._check("bring the green cylinder")
        self.assertTrue(c.ok)
        self.assertTrue(c.is_fleet)

    def test_color_only_object(self):
        c = self._check("Bravo, get the red one")
        self.assertTrue(c.ok)
        self.assertEqual(c.target_desc, "red")

    def test_shape_only_object(self):
        c = self._check("Delta, grab the cube")
        self.assertTrue(c.ok)
        self.assertEqual(c.target_desc, "cube")

    def test_empty_is_rejected(self):
        c = self._check("   ")
        self.assertFalse(c.ok)
        self.assertIn("Type a command", c.error)

    def test_unknown_callsign_is_rejected(self):
        c = self._check("Zulu, fetch the red cube")
        self.assertFalse(c.ok)
        self.assertIn("Zulu", c.error)
        self.assertIn("Alpha", c.error)

    def test_greeting_before_callsign_is_ok(self):
        c = self._check("hey Alpha, fetch the red cube")
        self.assertTrue(c.ok)
        self.assertEqual(c.recipient, "Alpha")

    def test_unresolvable_object_is_rejected(self):
        c = self._check("Alpha, do a barrel roll")
        self.assertFalse(c.ok)
        self.assertIn("colour", c.error)

    def test_greeting_lead_not_treated_as_callsign(self):
        # "Hey," opens with a comma but is a greeting, not an unknown callsign.
        c = self._check("Hey, someone fetch the blue ball")
        self.assertTrue(c.ok)
        self.assertTrue(c.is_fleet)

    def test_examples_are_all_valid(self):
        for ex in example_commands():
            with self.subTest(ex=ex["text"]):
                self.assertTrue(validate_command(ex["text"], _CALLSIGNS).ok)


if __name__ == "__main__":
    unittest.main()
