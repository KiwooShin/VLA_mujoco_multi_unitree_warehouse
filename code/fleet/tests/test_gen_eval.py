"""Fast unit tests for the randomized-layout generalization eval CLI.

Covers the layout-parameterized mission plans (shape + class mix), the
deterministic family sampler dispatch, the metrics summary and the BEV render —
all without stepping any physics (the mission-running path is exercised by the
full-suite mission tests + the eval run itself).
"""

from __future__ import annotations

import os
import tempfile
import unittest

import numpy as np

from code.fleet.gen_eval import (STACKS, build_plan, hero_plan, render_layout_bev,
                                 rooms_plan, sample_family_layout, summarize)
from code.fleet.search import region_name_for_xy, search_regions_for_layout
from code.warehouse.layout import hero_layout, rooms_layout


class TestPlans(unittest.TestCase):
    def test_hero_plan_shape_and_classes(self) -> None:
        seeds = 3
        plan = hero_plan(hero_layout(), seeds)
        self.assertEqual(len(plan), 3 * seeds + 3)
        klasses = [k for k, _s, _spot in plan]
        self.assertEqual(klasses.count("D"), 3)
        self.assertEqual(len(klasses) - 3, 3 * seeds)  # A/B/C fetches
        n_spots = len(hero_layout().object_spots)
        for _k, _s, spot in plan:
            self.assertTrue(0 <= spot < n_spots)

    def test_rooms_plan_shape_and_searchable(self) -> None:
        seeds = 3
        layout = rooms_layout()
        plan = rooms_plan(layout, seeds)
        self.assertEqual(len(plan), 3 * seeds + 3)
        klasses = [k for k, _s, _spot in plan]
        self.assertEqual(klasses.count("C"), 3 * seeds)
        self.assertEqual(klasses.count("D"), 3)
        # Every fetched spot lies in a searchable (non-spawn) room.
        regions = search_regions_for_layout(layout)
        for _k, _s, spot in plan:
            xy = layout.object_spots[spot]
            self.assertIn(region_name_for_xy(layout, xy), regions)

    def test_build_plan_dispatch(self) -> None:
        self.assertEqual(build_plan("hero", hero_layout(), 2),
                         hero_plan(hero_layout(), 2))
        self.assertEqual(build_plan("rooms", rooms_layout(), 2),
                         rooms_plan(rooms_layout(), 2))


class TestSampleFamilyLayout(unittest.TestCase):
    def test_deterministic_dispatch(self) -> None:
        a = sample_family_layout("rooms", 4)
        b = sample_family_layout("rooms", 4)
        self.assertEqual(a.object_spots, b.object_spots)
        self.assertEqual(a.hall_x, 20.0)
        h = sample_family_layout("hero", 1)
        self.assertEqual(h.hall_x, 16.0)
        self.assertEqual(len(h.rooms), 0)


class TestStacks(unittest.TestCase):
    def test_stack_definitions(self) -> None:
        self.assertEqual(STACKS["learned"],
                         {"perception_mode": "groundnet", "locomotion": "vla"})
        self.assertEqual(STACKS["baseline"],
                         {"perception_mode": "oracle", "locomotion": "teacher"})


class TestSummarize(unittest.TestCase):
    def _m(self, klass, success, outcome="complete", fell=False, steps=100,
           found=50, alloc_correct=None):
        r = {"class": klass, "seed": 0, "target_spot": 0, "success": success,
             "outcome": outcome, "any_fell": fell, "steps": steps,
             "found_step": found, "mean_vla_infer_ms": 1.5, "n_confirmations": 2}
        if klass == "D":
            r["alloc_correct"] = alloc_correct
        return r

    def test_counts_and_distributions(self) -> None:
        results = [
            self._m("A", True, steps=80, found=40),
            self._m("B", True, steps=120, found=60),
            self._m("C", False, outcome="timeout", steps=9000, found=None),
            self._m("D", False, alloc_correct=True),
            self._m("D", False, alloc_correct=False),
        ]
        s = summarize(results)
        self.assertEqual((s["ac_success"], s["ac_total"]), (2, 3))
        self.assertEqual((s["d_correct"], s["d_total"]), (1, 2))
        self.assertEqual(s["plan_failures"], 1)  # the C timeout
        self.assertEqual(s["steps"]["max"], 9000)
        self.assertEqual(s["per_class"]["A"], {"n": 1, "ok": 1})
        self.assertGreater(s["mean_vla_infer_ms"], 0.0)


class TestBev(unittest.TestCase):
    def test_renders_png(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            for layout in (hero_layout(), rooms_layout()):
                p = os.path.join(d, f"{layout.name}.png")
                render_layout_bev(layout, p, title="test")
                self.assertTrue(os.path.exists(p))
                self.assertGreater(os.path.getsize(p), 0)


if __name__ == "__main__":
    unittest.main()
