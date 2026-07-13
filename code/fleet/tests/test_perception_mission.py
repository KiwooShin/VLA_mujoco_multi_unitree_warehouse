"""End-to-end groundnet-mode mission smoke (skip without WBC/MuJoCo/EGL/ckpt).

Mirrors ``test_mission_smoke``'s conventions for environment-dependent assets:
skips (does not fail) when the walk policies / EGL / the GROUND_NET checkpoint
are unavailable. Verifies that the real learned detector is wired into the loop
(the owner completes a fetch with the detector confirming its target during the
approach) and that the detector's world-xy estimates land within the pickup
radius (the fetch endgame absorbs the detector error — item 3).
"""

from __future__ import annotations

import math
import unittest

from code.sim.arena_build import COLORS

_CMAP = dict(COLORS)


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


def _require_ckpt():
    """Skip the whole case if the GROUND_NET checkpoint cannot be loaded."""
    try:
        from code.fleet.perception_bridge import load_shared_detector
    except Exception as e:  # pragma: no cover - environment-dependent
        raise unittest.SkipTest(f"perception bridge unavailable: {e}")
    if load_shared_detector() is None:
        raise unittest.SkipTest("GROUND_NET checkpoint unavailable")


def _make_runner(mode):
    from code.fleet.mission import MissionRunner

    return MissionRunner(objects=_visible_scene(), use_gpu=True,
                         perception_mode=mode, search_deadline_steps=2600)


class TestGroundnetMission(unittest.TestCase):
    """Full scenario-A fetch with the learned detector confirming in the loop."""

    def test_groundnet_fetch_completes_with_confirmations(self) -> None:
        _require_ckpt()
        try:
            mr = _make_runner("groundnet")
        except unittest.SkipTest:
            raise
        except Exception as e:  # pragma: no cover - environment-dependent
            raise unittest.SkipTest(f"WBC/MuJoCo/EGL unavailable: {e}")
        try:
            self.assertEqual(mr.perception_mode, "groundnet")
            self.assertTrue(mr.perceptions["Alpha"].has_detector)
            mr.submit("Alpha, fetch the red cube to the delivery pad")
            res = mr.run(2600)
            # The mission still completes (oracle gate preserved) with nobody
            # falling — the detector confirms, it does not gate.
            self.assertEqual(res.outcome, "complete")
            self.assertTrue(res.object_on_pad)
            self.assertFalse(res.any_fell)
            # The learned detector actually confirmed the target in the loop...
            self.assertGreater(len(mr.confirmations), 0)
            # ...and its world-xy estimates land within the pickup radius, so the
            # fetch endgame absorbs the detector error (item 3).
            from code.fleet.carry import PICKUP_RADIUS_M
            errs = [math.hypot(ev.world_xy[0] + 5.0, ev.world_xy[1] - 0.5)
                    for _, ev in mr.confirmations]
            self.assertLess(min(errs), PICKUP_RADIUS_M)
        finally:
            mr.close()


class TestModeParity(unittest.TestCase):
    """Oracle and groundnet modes both complete the same visible fetch."""

    def test_both_modes_complete(self) -> None:
        _require_ckpt()
        for mode in ("oracle", "groundnet"):
            try:
                mr = _make_runner(mode)
            except unittest.SkipTest:
                raise
            except Exception as e:  # pragma: no cover - environment-dependent
                raise unittest.SkipTest(f"WBC/MuJoCo/EGL unavailable: {e}")
            try:
                mr.submit("Alpha, fetch the red cube to the delivery pad")
                res = mr.run(2600)
                self.assertEqual(res.outcome, "complete", f"mode={mode}")
                self.assertTrue(res.object_on_pad, f"mode={mode}")
            finally:
                mr.close()


if __name__ == "__main__":
    unittest.main()
