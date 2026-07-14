"""Comm-glow decay + title-card alpha ramp (pure time-based effects)."""

from __future__ import annotations

import dataclasses
import unittest

from code.apps.demos.effects import glow_levels, title_card_alpha


@dataclasses.dataclass
class _Msg:
    sender: str
    recipient: str
    t_step: int


class TestGlow(unittest.TestCase):
    def test_glow_peaks_at_send_then_decays_linearly(self):
        msgs = [_Msg("Alpha", "Bravo", 100)]
        names = ["Alpha", "Bravo", "Charlie"]
        self.assertEqual(glow_levels(msgs, 100, names, 100),
                         {"Alpha": 1.0, "Bravo": 1.0, "Charlie": 0.0})
        half = glow_levels(msgs, 150, names, 100)
        self.assertAlmostEqual(half["Alpha"], 0.5)
        self.assertAlmostEqual(half["Bravo"], 0.5)
        self.assertEqual(glow_levels(msgs, 201, names, 100)["Alpha"], 0.0)
        self.assertEqual(glow_levels(msgs, 50, names, 100)["Alpha"], 0.0)

    def test_glow_takes_freshest_of_multiple_messages(self):
        msgs = [_Msg("Alpha", "Bravo", 100), _Msg("Charlie", "Alpha", 180)]
        lvl = glow_levels(msgs, 190, ["Alpha", "Bravo", "Charlie"], 100)
        self.assertAlmostEqual(lvl["Alpha"], 0.9)   # t=180 touch, not t=100
        self.assertAlmostEqual(lvl["Bravo"], 0.1)

    def test_glow_rejects_nonpositive_window(self):
        with self.assertRaises(ValueError):
            glow_levels([], 0, ["Alpha"], 0)


class TestTitleCardAlpha(unittest.TestCase):
    def test_holds_then_fades(self):
        self.assertEqual(title_card_alpha(0.0, 2.0, 0.6), 1.0)
        self.assertEqual(title_card_alpha(1.4, 2.0, 0.6), 1.0)   # solid to hold-fade
        self.assertAlmostEqual(title_card_alpha(1.7, 2.0, 0.6), 0.5)
        self.assertEqual(title_card_alpha(2.0, 2.0, 0.6), 0.0)
        self.assertEqual(title_card_alpha(3.0, 2.0, 0.6), 0.0)

    def test_monotone_non_increasing(self):
        prev = 1.1
        for i in range(0, 25):
            a = title_card_alpha(i * 0.1, 2.0, 0.6)
            self.assertLessEqual(a, prev + 1e-9)
            prev = a

    def test_disabled_when_hold_zero(self):
        self.assertEqual(title_card_alpha(0.0, 0.0, 0.6), 0.0)


if __name__ == "__main__":
    unittest.main()
