"""Unit tests for per-robot status chip derivation."""

from __future__ import annotations

import unittest

from code.apps.fleet_web.status import (ACCENT_HEX, RobotSnap, accent,
                                        robot_view, state_label)


def _snap(**kw) -> RobotSnap:
    base = dict(name="Alpha", coord_state="IDLE", motion="idle",
                dist_to_goal=None, carrying=False, is_owner=False, task_desc="")
    base.update(kw)
    return RobotSnap(**base)


class StateLabelTest(unittest.TestCase):
    def test_idle_standing_by(self):
        self.assertEqual(state_label(_snap()), "Standing by")

    def test_carrying_wins(self):
        s = _snap(coord_state="OWNER_DELIVERING", carrying=True)
        self.assertEqual(state_label(s), "Carrying to pad")

    def test_coordination_states(self):
        cases = {
            "OWNER_QUERYING": "Asking peers",
            "OWNER_DELEGATING": "Coordinating search",
            "OWNER_NAVIGATING": "Fetching",
            "OWNER_DELIVERING": "Delivering",
            "ASSIST_SEARCHING": "Searching",
        }
        for st, label in cases.items():
            self.assertEqual(state_label(_snap(coord_state=st)), label)

    def test_arrived_and_fallen(self):
        self.assertEqual(state_label(_snap(motion="arrived")), "Arrived")
        self.assertEqual(state_label(_snap(motion="fallen")), "Fallen")


class RobotViewTest(unittest.TestCase):
    def test_owner_task_line(self):
        s = _snap(coord_state="OWNER_NAVIGATING", motion="walking",
                  dist_to_goal=3.2, is_owner=True, task_desc="red cube")
        v = robot_view(s)
        self.assertEqual(v["name"], "Alpha")
        self.assertEqual(v["color"], ACCENT_HEX["Alpha"])
        self.assertEqual(v["state"], "Fetching")
        self.assertEqual(v["task"], "red cube → delivery pad")
        self.assertTrue(v["busy"])
        self.assertEqual(v["dist"], "3.2 m")

    def test_idle_view_has_no_dist_or_task(self):
        v = robot_view(_snap())
        self.assertEqual(v["task"], "—")
        self.assertEqual(v["dist"], "")
        self.assertFalse(v["busy"])

    def test_searcher_task_text(self):
        v = robot_view(_snap(coord_state="ASSIST_SEARCHING", motion="walking",
                             dist_to_goal=1.0))
        self.assertEqual(v["task"], "assisting the search")
        self.assertTrue(v["busy"])

    def test_accent_fallback(self):
        self.assertEqual(accent("Echo"), "#9aa0a6")


if __name__ == "__main__":
    unittest.main()
