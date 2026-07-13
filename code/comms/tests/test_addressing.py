"""Unit tests for code.comms.addressing (callsign / fleet parse table)."""

from __future__ import annotations

import unittest

from code.comms.addressing import FLEET, parse_addressed_instruction

CALLSIGNS = ("Alpha", "Bravo", "Charlie", "Delta")


class TestAddressingTable(unittest.TestCase):
    def _parse(self, text: str):
        return parse_addressed_instruction(text, CALLSIGNS)

    def test_parse_table(self) -> None:
        # (text) -> (expected recipient, expected body)
        table = [
            ("Alpha, fetch the red cube to the delivery pad",
             "Alpha", "fetch the red cube to the delivery pad"),
            ("hey bravo find the blue ball", "Bravo", "find the blue ball"),
            ("someone bring me the yellow cone", FLEET, "bring me the yellow cone"),
            ("everyone stop", FLEET, "stop"),
            ("Charlie go to the delivery pad", "Charlie", "go to the delivery pad"),
            ("DELTA, PICK UP THE GREEN CUBE", "Delta", "PICK UP THE GREEN CUBE"),
            ("bring me the orange cylinder", FLEET, "bring me the orange cylinder"),
            ("ok alpha please deliver the purple ball",
             "Alpha", "deliver the purple ball"),
            ("anyone see the red ball?", FLEET, "see the red ball?"),
            ("robots, return to base", FLEET, "return to base"),
            ("everybody stop moving", FLEET, "stop moving"),
            ("alpha! fetch the red ball.", "Alpha", "fetch the red ball."),
            ("hi Delta, search the north aisle", "Delta", "search the north aisle"),
            ("find the cyan cube", FLEET, "find the cyan cube"),
            ("whoever is free, grab the blue cone",
             FLEET, "is free, grab the blue cone"),
            ("bravo", "Bravo", ""),
            ("hey delta", "Delta", ""),
        ]
        for text, recipient, body in table:
            with self.subTest(text=text):
                res = self._parse(text)
                self.assertEqual(res.recipient, recipient)
                self.assertEqual(res.body, body)
                self.assertEqual(res.is_fleet, recipient == FLEET)

    def test_negative_substring_is_not_a_callsign(self) -> None:
        # "alphabet" must not match the "Alpha" callsign (word boundary).
        res = self._parse("alphabet soup is on the floor")
        self.assertEqual(res.recipient, FLEET)
        self.assertEqual(res.matched_callsign, None)

    def test_empty_text_routes_to_fleet(self) -> None:
        res = self._parse("")
        self.assertEqual(res.recipient, FLEET)
        self.assertEqual(res.body, "")

    def test_first_callsign_wins_when_several(self) -> None:
        res = self._parse("charlie and delta hold position")
        self.assertEqual(res.recipient, "Charlie")
        self.assertIn("delta", res.body.lower())

    def test_matched_callsign_is_canonical(self) -> None:
        res = self._parse("BRAVO, advance")
        self.assertEqual(res.recipient, "Bravo")
        self.assertEqual(res.matched_callsign, "Bravo")

    def test_is_fleet_flag(self) -> None:
        self.assertTrue(self._parse("someone help").is_fleet)
        self.assertFalse(self._parse("Alpha help").is_fleet)


if __name__ == "__main__":
    unittest.main()
