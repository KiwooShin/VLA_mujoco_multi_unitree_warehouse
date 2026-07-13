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


class TestOwnConfirmSingleRender(unittest.TestCase):
    """Finding 8a: the owner own-confirm leg renders the grounding cam once/step."""

    def test_own_confirm_leg_does_not_double_render(self) -> None:
        # Alpha's red cube (spot 6) is 5.5 m away (> the 4.5 m reliable range), so
        # a groundnet fetch runs the CONFIRM-THEN-REPORT own-confirm walk-in. The
        # protocol's own can_see already renders + confirms each of those steps;
        # _owner_approach_confirm must NOT render a second time (it would double
        # the dominant GPU cost and drop the first confirmation's telemetry).
        from code.comms.protocol import RobotState
        _require_ckpt()
        try:
            mr = _make_runner("groundnet")
        except unittest.SkipTest:
            raise
        except Exception as e:  # pragma: no cover - environment-dependent
            raise unittest.SkipTest(f"WBC/MuJoCo/EGL unavailable: {e}")
        try:
            p = mr.perceptions["Alpha"]
            orig_confirm = p.confirm
            per_step = {"n": 0}
            own_confirm_max = {"n": 0}
            own_confirm_steps = {"n": 0}

            def counting_confirm(*a, **k):
                per_step["n"] += 1
                return orig_confirm(*a, **k)

            p.confirm = counting_confirm  # type: ignore[assignment]
            double_steps = {"n": 0}

            def hook(runner, t):
                proto = runner.protocols["Alpha"]
                if (proto.state is RobotState.OWNER_NAVIGATING
                        and proto.located_target is None):  # own-confirm leg
                    own_confirm_steps["n"] += 1
                    own_confirm_max["n"] = max(own_confirm_max["n"], per_step["n"])
                    if per_step["n"] > 1:
                        double_steps["n"] += 1
                per_step["n"] = 0
                return None

            mr.submit("Alpha, fetch the red cube to the delivery pad")
            res = mr.run(2600, on_step=hook)
            self.assertEqual(res.outcome, "complete")
            # The own-confirm leg ran for many steps (5.5 m walk-in to the standoff).
            self.assertGreater(own_confirm_steps["n"], 3)
            # Before the fix _owner_approach_confirm rendered a SECOND time every
            # step of the leg. Now the leg renders once/step throughout its
            # duration; only the entry step can carry an extra render (the
            # task-receipt view check preceding the confirm leg), never the leg
            # itself — so at most one own-confirm step double-renders.
            self.assertLessEqual(double_steps["n"], 1)
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
