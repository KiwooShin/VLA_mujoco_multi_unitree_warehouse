"""Scenario config tests: every demo constructs and its ambiguity precondition holds.

Pure-data (no simulator): builds each scenario's layout + object placement and
checks the manifest-ambiguity contract — that an ambiguous order's CLARIFY lists
exactly the intended options, and that unambiguous demos leave every planted target
uniquely resolvable. Guards the production contract without a GPU/MuJoCo run.
"""

from __future__ import annotations

import unittest

from code.comms.messages import ObjectQuery, clarify_options
from code.apps.demos.scenarios import REGISTRY, get, names
from code.apps.demos.scenarios.core import Scenario, build_objects, manifest_of


class TestRegistry(unittest.TestCase):
    def test_six_distinct_named_scenarios(self):
        self.assertEqual(len(REGISTRY), 6)
        self.assertEqual(len(set(names())), 6)
        for n in names():
            self.assertIsInstance(get(n), Scenario)

    def test_expected_names_present(self):
        self.assertEqual(set(names()), {
            "clarify_fetch", "unseen_map", "dual_fetch", "relay_multigoal",
            "retask", "six_robot_flagship"})


class TestConstructsAndPlaces(unittest.TestCase):
    def test_each_scenario_builds_layout_and_objects(self):
        for sc in REGISTRY:
            with self.subTest(sc.name):
                layout = sc.make_layout()
                objs = sc.make_objects(layout)
                # One object per spot, all with a resolvable colour rgb.
                self.assertEqual(len(objs), len(layout.object_spots))
                for o in objs:
                    self.assertIn("color_rgb", o)
                    self.assertIsNotNone(o["color_name"])
                    self.assertIsNotNone(o["shape_name"])

    def test_learned_stack_is_the_default(self):
        # Prefer the learned stack; any fallback is explicit + justified in report.
        learned = [s.name for s in REGISTRY
                   if s.perception_mode == "groundnet" and s.locomotion == "vla"]
        self.assertGreaterEqual(len(learned), 5,
                                f"learned-stack demos: {learned}")


class TestAmbiguityPrecondition(unittest.TestCase):
    def test_precondition_holds_for_all(self):
        for sc in REGISTRY:
            with self.subTest(sc.name):
                sc.check_precondition()  # raises on violation

    def test_ambiguous_demos_list_exactly_the_options(self):
        # clarify_fetch -> exactly three cubes; flagship -> exactly three balls.
        cf = get("clarify_fetch")
        self.assertEqual(sorted(cf.clarify_preview()),
                         ["blue cube", "red cube", "yellow cube"])
        fl = get("six_robot_flagship")
        self.assertEqual(sorted(fl.clarify_preview()),
                         ["blue ball", "green ball", "red ball"])

    def test_unambiguous_demos_have_unique_planted_targets(self):
        # Every non-clarify demo's planted objects must each be unique in the
        # manifest, so the order resolves without a clarification.
        for name in ("unseen_map", "dual_fetch", "relay_multigoal", "retask"):
            sc = get(name)
            layout = sc.make_layout()
            planted = (sc.planted_fn(layout) if sc.planted_fn else sc.planted)
            manifest = manifest_of(sc.make_objects(layout))
            for _spot, (color, shape) in planted.items():
                opts = clarify_options(ObjectQuery(color, shape), manifest)
                self.assertLessEqual(
                    len(opts), 1,
                    f"{name}: planted {color} {shape} is ambiguous ({opts})")


class TestObjectPlacementHelper(unittest.TestCase):
    def test_planted_overrides_fillers(self):
        class _L:
            object_spots = [(0, 0), (1, 1), (2, 2)]
        objs = build_objects(_L(), {1: ("red", "cube")})
        self.assertEqual(objs[1]["color_name"], "red")
        self.assertEqual(objs[1]["shape_name"], "cube")
        # Fillers fill the rest and are never the ambiguity shapes by default.
        for o in (objs[0], objs[2]):
            self.assertNotIn(o["shape_name"], ("cube", "ball"))


if __name__ == "__main__":
    unittest.main()
