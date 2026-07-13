"""Fast unit tests for the mission runner's pure helpers (no physics)."""

from __future__ import annotations

import unittest

from code.comms.messages import ObjectQuery
from code.comms.protocol import RobotState
from code.fleet.mission import _OWNER_PHASE, delivery_xy, resolve_query
from code.warehouse.layout import hero_layout


class TestResolveQuery(unittest.TestCase):
    def test_colour_and_shape(self) -> None:
        self.assertEqual(resolve_query("fetch the red cube to the delivery pad"),
                         ObjectQuery("red", "cube"))

    def test_shape_only(self) -> None:
        self.assertEqual(resolve_query("bring the cone over"),
                         ObjectQuery(None, "cone"))

    def test_colour_only(self) -> None:
        self.assertEqual(resolve_query("grab the blue thing"),
                         ObjectQuery("blue", None))

    def test_unresolvable(self) -> None:
        self.assertIsNone(resolve_query("do something useful"))


class TestDeliveryXy(unittest.TestCase):
    def test_delivery_pad_from_layout(self) -> None:
        name, xy = delivery_xy(hero_layout())
        self.assertEqual(name, "delivery pad")
        self.assertEqual(xy, (5.8, -1.0))


class TestPhaseLabels(unittest.TestCase):
    def test_owner_phase_covers_owner_states(self) -> None:
        for st in (RobotState.OWNER_QUERYING, RobotState.OWNER_DELEGATING,
                   RobotState.OWNER_NAVIGATING, RobotState.OWNER_DELIVERING):
            self.assertIn(st, _OWNER_PHASE)
            self.assertTrue(_OWNER_PHASE[st].isupper())


if __name__ == "__main__":
    unittest.main()
