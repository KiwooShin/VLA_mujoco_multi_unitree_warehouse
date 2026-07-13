"""Robustness regression tests for the mission runner + region defaults.

* Finding 6 (no physics): the comms default region set is exactly the one the
  searcher's :func:`~code.fleet.search.region_bounds` understands, so a
  default-constructed protocol can delegate a search without crashing.
* Finding 4 (needs WBC/MuJoCo/EGL): a fleet request that arrives with no idle
  robot is queued (with a one-time notice) and assigned once a robot frees up,
  instead of vanishing and timing the mission out silently.
* Finding 5 (needs WBC/MuJoCo/EGL): submitting a second order while a mission is
  already in flight is rejected rather than silently clobbering the active task.
"""

from __future__ import annotations

import unittest

from code.comms.messages import ObjectQuery, Performative
from code.comms.protocol import DEFAULT_REGIONS
from code.fleet.search import SEARCH_REGIONS, region_bounds
from code.sim.arena_build import COLORS

_CMAP = dict(COLORS)


class TestDefaultRegionsCoverable(unittest.TestCase):
    """Finding 6: DEFAULT_REGIONS must be valid searcher regions."""

    def test_default_regions_match_search_regions(self) -> None:
        self.assertEqual(DEFAULT_REGIONS, SEARCH_REGIONS)

    def test_region_bounds_accepts_every_default_region(self) -> None:
        for region in DEFAULT_REGIONS:
            y_lo, y_hi = region_bounds(region, 20.0)  # must not raise
            self.assertLess(y_lo, y_hi)


def _visible_scene():
    """Red cube at spot 6 (-5, 0.5), visible to Alpha at its south bay."""
    from code.warehouse.layout import hero_layout

    fill = [("orange", "cube", 0.24), ("blue", "cylinder", 0.22),
            ("green", "ball", 0.24), ("yellow", "cone", 0.26),
            ("purple", "cube", 0.24), ("cyan", "cylinder", 0.22),
            ("blue", "ball", 0.24)]
    objs, fi = [], 0
    for i, (x, y) in enumerate(hero_layout().object_spots):
        if i == 6:
            c, s, sz = "red", "cube", 0.24
        else:
            c, s, sz = fill[fi]
            fi += 1
        objs.append({"color_name": c, "color_rgb": _CMAP[c], "shape_name": s,
                     "size": float(sz), "x": float(x), "y": float(y)})
    return objs


class _RunnerCase(unittest.TestCase):
    """Base that builds shared teachers once and a fresh runner per test."""

    @classmethod
    def setUpClass(cls) -> None:
        try:
            from code.sim.teacher import WBCTeacher
            from code.warehouse.layout import CALLSIGNS

            cls.teachers = {cs: WBCTeacher(use_gpu=True) for cs in CALLSIGNS}
        except unittest.SkipTest:
            raise
        except Exception as e:  # pragma: no cover - environment-dependent
            raise unittest.SkipTest(f"WBC/MuJoCo/EGL unavailable: {e}")

    def setUp(self) -> None:
        from code.fleet.mission import MissionRunner

        self.mr = MissionRunner(objects=_visible_scene(), teachers=self.teachers,
                                use_gpu=True)

    def tearDown(self) -> None:
        if getattr(self, "mr", None) is not None:
            self.mr.close()


def _queued_notices(mr):
    return [m for m in mr.bus.transcript
            if m.performative is Performative.STATUS_UPDATE
            and "queued" in m.payload.get("text", "")]


class TestFleetQueue(_RunnerCase):
    """Finding 4: no idle robot -> queue + one notice -> assign on free."""

    def test_queued_then_assigned_when_robot_frees(self) -> None:
        mr = self.mr
        # Make every robot busy so the fleet request cannot be allocated.
        for cs in mr.callsigns:
            self.assertTrue(mr.protocols[cs]._enter_assist(
                "Base", ObjectQuery("red", "cube"), "middle", 0))
            self.assertFalse(mr.protocols[cs].is_idle())

        mr.submit("someone bring me the red cube")
        mr._run_allocator()
        self.assertEqual(len(_queued_notices(mr)), 1)
        self.assertEqual(len(mr._pending_fleet), 1)

        # Retrying while still busy neither re-notifies nor drops the request.
        mr._run_allocator()
        self.assertEqual(len(_queued_notices(mr)), 1)
        self.assertEqual(len(mr._pending_fleet), 1)

        # Free one robot; the queued order is now assigned to a real robot.
        mr.protocols["Bravo"]._actions.abort_search()
        mr.protocols["Bravo"]._end_assist()
        mr._run_allocator()
        self.assertEqual(len(mr._pending_fleet), 0)
        assigned = [m for m in mr.bus.transcript
                    if m.performative is Performative.REQUEST_TASK
                    and m.sender == "allocator"]
        self.assertEqual(len(assigned), 1)
        self.assertIn(assigned[0].recipient, mr.callsigns)


class TestSubmitGuard(_RunnerCase):
    """Finding 5: a second submit while a mission runs is rejected."""

    def test_second_submit_raises(self) -> None:
        self.mr.submit("Alpha, fetch the red cube to the delivery pad")
        with self.assertRaises(RuntimeError):
            self.mr.submit("Bravo, fetch the red cube to the delivery pad")


if __name__ == "__main__":
    unittest.main()
