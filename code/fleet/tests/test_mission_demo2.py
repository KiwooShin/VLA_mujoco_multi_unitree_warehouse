"""Real-sim smokes for the Demo Set v2 capabilities (skip without WBC/MuJoCo/EGL).

One short oracle+teacher mission per new capability — clarification dialogue,
random spawns, concurrent (two-owner) missions, sequential multi-goal, and a
mid-mission re-task — exercised end to end through the real :class:`MissionRunner`
(federated physics + shared viz + carry). Each asserts the new flow actually
engaged in the transcript AND that the world outcome is a clean, no-fall delivery.
"""

from __future__ import annotations

import unittest

from code.comms.messages import ObjectQuery, Performative
from code.sim.arena_build import COLORS

_CMAP = dict(COLORS)
P = Performative

# Hero object-spot visibility (computed from the fixed layout): spot 6 (-5, 0.5)
# is visible to Alpha; spot 7 (3, -0.5) to Charlie/Delta; all others hidden.
_ALPHA_SPOT = 6
_PEER_SPOT = 7

# Non-cube, non-target fillers so a "the cube" order has exactly the intended
# cube types in the manifest.
_FILLER = (("green", "ball", 0.24), ("cyan", "cylinder", 0.22),
           ("green", "cone", 0.26), ("purple", "ball", 0.24),
           ("cyan", "cone", 0.26))


def _scene(placements):
    """Hero scene with ``{spot_idx: (color, shape, size)}`` and non-cube fillers."""
    from code.warehouse.layout import hero_layout

    objs, fi = [], 0
    for i, (x, y) in enumerate(hero_layout().object_spots):
        if i in placements:
            c, s, sz = placements[i]
        else:
            c, s, sz = _FILLER[fi % len(_FILLER)]
            fi += 1
        objs.append({"color_name": c, "color_rgb": _CMAP[c], "shape_name": s,
                     "size": float(sz), "x": float(x), "y": float(y)})
    return objs


class _Demo2Case(unittest.TestCase):
    """Builds shared walk teachers once; each test gets a fresh runner."""

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

    def _runner(self, **kw):
        from code.fleet.mission import MissionRunner

        return MissionRunner(teachers=self.teachers, use_gpu=True, **kw)


class TestClarifySmoke(_Demo2Case):
    """Ambiguous "the cube" -> CLARIFY -> scripted "the red one" -> fetch & deliver."""

    def test_clarify_fetch_completes(self) -> None:
        mr = self._runner(objects=_scene({_ALPHA_SPOT: ("red", "cube", 0.24),
                                          1: ("blue", "cube", 0.24),
                                          3: ("yellow", "cube", 0.24)}))
        try:
            mr.submit("Alpha, bring the cube to the delivery pad",
                      replies=["the red one"])
            res = mr.run(3200)
            perfs = [m.performative for m in mr.bus.transcript]
            self.assertIn(P.CLARIFY, perfs)
            self.assertIn(P.USER_REPLY, perfs)
            clar = next(m for m in mr.bus.transcript if m.performative is P.CLARIFY)
            self.assertEqual(sorted(clar.payload["options"]),
                             ["blue cube", "red cube", "yellow cube"])
            self.assertEqual(res.outcome, "complete")
            self.assertTrue(res.object_on_pad)
            self.assertFalse(res.any_fell)
        finally:
            mr.close()


class TestRandomSpawnSmoke(_Demo2Case):
    """Seeded random start poses (not the home bays) still complete a fleet fetch."""

    def test_random_spawn_fetch_completes(self) -> None:
        from code.warehouse.layout import hero_layout

        bays = hero_layout().spawn_poses
        mr = self._runner(objects=_scene({_ALPHA_SPOT: ("red", "cube", 0.24)}),
                          spawn_seed=7)
        try:
            # The robots really did start away from their home bays.
            moved = [cs for cs in mr.callsigns
                     if mr.fleet.units[cs].xy != bays[cs][:2]]
            self.assertEqual(len(moved), len(mr.callsigns))
            mr.submit("someone bring me the red cube")
            res = mr.run(3500)
            self.assertEqual(res.outcome, "complete")
            self.assertTrue(res.object_on_pad)
            self.assertFalse(res.any_fell)
        finally:
            mr.close()


class TestConcurrentSmoke(_Demo2Case):
    """Two conjuncts -> two owners in parallel -> both delivered (need-to-know kept)."""

    def test_dual_fetch_both_delivered(self) -> None:
        mr = self._runner(objects=_scene({_ALPHA_SPOT: ("red", "cube", 0.24),
                                          _PEER_SPOT: ("blue", "ball", 0.24)}))
        try:
            tasks = mr.submit_multi(
                "bring the red cube and the blue ball to the delivery pad")
            self.assertEqual(len(tasks), 2)
            res = mr.run(3000)
            self.assertEqual(mr.mission_outcomes(), ["complete", "complete"])
            self.assertTrue(res.object_on_pad)          # both objects on the pad
            self.assertFalse(res.any_fell)
            # Two distinct owners each sent their own completion (need-to-know).
            completes = [m for m in mr.bus.transcript
                         if m.performative is P.TASK_COMPLETE]
            self.assertEqual(len({m.sender for m in completes}), 2)
            # No searcher's REPORT_FOUND ever reached the user.
            for m in mr.bus.transcript:
                if m.performative is P.REPORT_FOUND:
                    self.assertNotEqual(m.recipient, "user")
        finally:
            mr.close()


class TestSequentialSmoke(_Demo2Case):
    """One owner fetches red cube THEN yellow cylinder; one TASK_COMPLETE at the end."""

    def test_two_legs_complete_in_order(self) -> None:
        mr = self._runner(objects=_scene({_ALPHA_SPOT: ("red", "cube", 0.24),
                                          _PEER_SPOT: ("yellow", "cylinder", 0.22)}))
        try:
            mr.submit("Alpha, bring the red cube, then the yellow cylinder")
            res = mr.run(5200)
            completes = [m for m in mr.bus.transcript
                         if m.performative is P.TASK_COMPLETE]
            self.assertEqual(len(completes), 1)                 # only after leg B
            self.assertIn("yellow cylinder", completes[0].payload["text"])
            self.assertTrue(any("now fetching the yellow cylinder"
                                in m.payload.get("text", "")
                                for m in mr.bus.transcript))     # per-leg milestone
            self.assertEqual(res.outcome, "complete")
            self.assertFalse(res.any_fell)
        finally:
            mr.close()


class TestRetaskSmoke(_Demo2Case):
    """Mid-navigation re-task redirects the owner; the FINAL task is delivered."""

    def test_retask_redirects_owner(self) -> None:
        mr = self._runner(objects=_scene({_ALPHA_SPOT: ("red", "cube", 0.24),
                                          _PEER_SPOT: ("yellow", "cylinder", 0.22)}))
        try:
            mr.submit("Alpha, fetch the red cube to the delivery pad")
            mr.retask("actually, bring the yellow cylinder instead", at_step=40)
            res = mr.run(3000)
            # The owner announced the switch and delivered the NEW target only.
            self.assertTrue(any("switching to the yellow cylinder"
                                in m.payload.get("text", "")
                                for m in mr.bus.transcript))
            complete = next(m for m in mr.bus.transcript
                            if m.performative is P.TASK_COMPLETE)
            self.assertIn("yellow cylinder", complete.payload["text"])
            self.assertEqual(res.outcome, "complete")
            self.assertFalse(res.any_fell)
            # The abandoned red cube never reached the pad.
            self.assertFalse(mr._query_on_pad(ObjectQuery("red", "cube"), 1.0))
        finally:
            mr.close()


if __name__ == "__main__":
    unittest.main()
