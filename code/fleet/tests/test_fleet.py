"""Unit tests for the pure proximity-pause logic + fleet constants.

No simulator is stepped here (see test_smoke.py for the 4-robot sim smoke).
"""

from __future__ import annotations

import unittest

from code.fleet.fleet import ENGAGE_M, RELEASE_M, compute_pauses
from code.fleet.viz import ACCENT_RGBA
from code.warehouse.layout import CALLSIGNS

_PRIO = {"Alpha": 0, "Bravo": 1, "Charlie": 2, "Delta": 3}
_ACTIVE_ALL = {c: True for c in CALLSIGNS}


class TestComputePauses(unittest.TestCase):
    def test_far_apart_no_pause(self) -> None:
        pos = {"Alpha": (-5.0, 0.0), "Bravo": (5.0, 0.0),
               "Charlie": (-5.0, 5.0), "Delta": (5.0, 5.0)}
        self.assertEqual(
            compute_pauses(pos, _ACTIVE_ALL, _PRIO, set()), set())

    def test_lower_priority_yields(self) -> None:
        # Alpha (highest) and Bravo within engage: only Bravo pauses.
        pos = {"Alpha": (0.0, 0.0), "Bravo": (0.0, 0.5),
               "Charlie": (-5.0, 0.0), "Delta": (5.0, 0.0)}
        paused = compute_pauses(pos, _ACTIVE_ALL, _PRIO, set())
        self.assertEqual(paused, {"Bravo"})

    def test_highest_priority_never_pauses(self) -> None:
        pos = {"Alpha": (0.0, 0.0), "Bravo": (0.0, 0.4),
               "Charlie": (0.0, 0.8), "Delta": (5.0, 0.0)}
        paused = compute_pauses(pos, _ACTIVE_ALL, _PRIO, set())
        self.assertNotIn("Alpha", paused)

    def test_engage_threshold_starts_pause(self) -> None:
        # 0.9 m apart (< engage 1.0): pause starts.
        pos = {"Alpha": (0.0, 0.0), "Bravo": (0.0, 0.9),
               "Charlie": (-5.0, 0.0), "Delta": (5.0, 0.0)}
        self.assertIn("Bravo", compute_pauses(pos, _ACTIVE_ALL, _PRIO, set()))

    def test_engage_not_tripped_when_between_bands(self) -> None:
        # 1.1 m apart (> engage 1.0) and NOT already paused: no pause.
        pos = {"Alpha": (0.0, 0.0), "Bravo": (0.0, 1.1),
               "Charlie": (-5.0, 0.0), "Delta": (5.0, 0.0)}
        self.assertNotIn("Bravo", compute_pauses(pos, _ACTIVE_ALL, _PRIO, set()))

    def test_hysteresis_stays_paused_in_band(self) -> None:
        # Already paused, now 1.1 m apart (between engage 1.0 and release 1.2):
        # must stay paused (harder to leave the paused state).
        pos = {"Alpha": (0.0, 0.0), "Bravo": (0.0, 1.1),
               "Charlie": (-5.0, 0.0), "Delta": (5.0, 0.0)}
        paused = compute_pauses(pos, _ACTIVE_ALL, _PRIO, {"Bravo"})
        self.assertIn("Bravo", paused)

    def test_hysteresis_releases_beyond_release(self) -> None:
        # Already paused, now 1.3 m apart (> release 1.2): resumes.
        pos = {"Alpha": (0.0, 0.0), "Bravo": (0.0, 1.3),
               "Charlie": (-5.0, 0.0), "Delta": (5.0, 0.0)}
        paused = compute_pauses(pos, _ACTIVE_ALL, _PRIO, {"Bravo"})
        self.assertNotIn("Bravo", paused)

    def test_inactive_blocker_does_not_pause_others(self) -> None:
        # Alpha inactive (arrived/idle): its proximity must not pause Bravo.
        pos = {"Alpha": (0.0, 0.0), "Bravo": (0.0, 0.5),
               "Charlie": (-5.0, 0.0), "Delta": (5.0, 0.0)}
        active = dict(_ACTIVE_ALL, Alpha=False)
        self.assertEqual(compute_pauses(pos, active, _PRIO, set()), set())

    def test_inactive_self_never_pauses(self) -> None:
        # An inactive (arrived) robot is never itself paused.
        pos = {"Alpha": (0.0, 0.0), "Bravo": (0.0, 0.5),
               "Charlie": (-5.0, 0.0), "Delta": (5.0, 0.0)}
        active = dict(_ACTIVE_ALL, Bravo=False)
        self.assertEqual(compute_pauses(pos, active, _PRIO, set()), set())

    def test_release_at_least_engage(self) -> None:
        self.assertGreaterEqual(RELEASE_M, ENGAGE_M)


class TestAccentColours(unittest.TestCase):
    def test_every_callsign_has_accent(self) -> None:
        for cs in CALLSIGNS:
            self.assertIn(cs, ACCENT_RGBA)
            self.assertEqual(len(ACCENT_RGBA[cs]), 4)


if __name__ == "__main__":
    unittest.main()
