"""End-to-end mission smoke (skip without WBC/MuJoCo/EGL).

A fast reaction check plus a full scenario-A fetch: the owner sees the red cube
at its bay, walks over, mock-picks it, delivers it to the pad and reports
``TASK_COMPLETE`` — with nobody falling.
"""

from __future__ import annotations

import unittest

from code.sim.arena_build import COLORS

_CMAP = dict(COLORS)


def _visible_scene():
    """Red cube at spot 6 (-5, 0.5), visible to Alpha at its south bay."""
    from code.warehouse.layout import hero_layout

    objs = []
    fill = [("orange", "cube", 0.24), ("blue", "cylinder", 0.22),
            ("green", "ball", 0.24), ("yellow", "cone", 0.26),
            ("purple", "cube", 0.24), ("cyan", "cylinder", 0.22),
            ("blue", "ball", 0.24)]
    fi = 0
    for i, (x, y) in enumerate(hero_layout().object_spots):
        if i == 6:
            c, s, sz = "red", "cube", 0.24
        else:
            c, s, sz = fill[fi]
            fi += 1
        objs.append({"color_name": c, "color_rgb": _CMAP[c], "shape_name": s,
                     "size": float(sz), "x": float(x), "y": float(y)})
    return objs


def _make_runner():
    from code.fleet.mission import MissionRunner

    return MissionRunner(objects=_visible_scene(), use_gpu=True)


class TestMissionReacts(unittest.TestCase):
    """Fast: the owner reacts to a REQUEST_TASK within a few steps."""

    def test_owner_navigates_quickly(self) -> None:
        try:
            mr = _make_runner()
        except Exception as e:  # pragma: no cover - environment-dependent
            raise unittest.SkipTest(f"WBC/MuJoCo/EGL unavailable: {e}")
        try:
            from code.comms.messages import Performative
            from code.comms.protocol import RobotState

            mr.submit("Alpha, fetch the red cube to the delivery pad")
            mr.run(60)
            self.assertFalse(mr.fleet.any_fell)
            self.assertEqual(mr.protocols["Alpha"].state,
                             RobotState.OWNER_NAVIGATING)
            perfs = [m.performative for m in mr.bus.transcript]
            self.assertIn(Performative.REQUEST_TASK, perfs)
            self.assertIn(Performative.STATUS_UPDATE, perfs)
        finally:
            mr.close()


class TestMissionEndToEnd(unittest.TestCase):
    """Full scenario-A fetch to completion."""

    def test_fetch_completes_on_pad(self) -> None:
        try:
            mr = _make_runner()
        except Exception as e:  # pragma: no cover - environment-dependent
            raise unittest.SkipTest(f"WBC/MuJoCo/EGL unavailable: {e}")
        try:
            mr.submit("Alpha, fetch the red cube to the delivery pad")
            res = mr.run(2600)
            self.assertEqual(res.outcome, "complete")
            self.assertTrue(res.object_on_pad)
            self.assertTrue(res.task_complete_sent)
            self.assertFalse(res.any_fell)
        finally:
            mr.close()


if __name__ == "__main__":
    unittest.main()
