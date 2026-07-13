"""Unit tests for the GROUND_NET perception bridge (code.fleet.perception_bridge).

These exercise the confirmer's wiring, the world-xy conversion math, the
oracle/groundnet mode switch and — crucially — per-robot GroundNetState
ISOLATION (two robots' hysteresis track state must not interact; the regression
guard for the process-wide-singleton bug the baseline survey flagged). A scripted
fake detector + fake viz/renderer stand in for the real checkpoint, so nothing
here needs the weights, torch or a GPU.
"""

from __future__ import annotations

import math
import unittest

import numpy as np

from code.comms.messages import ObjectQuery
from code.fleet.actions import FleetRobotActions
from code.fleet.perception_bridge import (CONFIRM_BEARING_GATE_DEG,
                                          CONFIRM_DIST_GATE_M, DetectionResult,
                                          RobotPerception, detection_world_xy,
                                          oracle_range_bearing)

_CLASS = ["ball", "cube", "cylinder", "cone"]
_COLOR = ["red", "yellow", "blue", "green", "orange", "purple", "cyan"]


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------
class _FakeDetector:
    """Scripted HeatmapDetector: returns queued (present, conf, dist, bearing)."""

    def __init__(self, script):
        self._script = list(script)
        self._i = 0
        self.last_heat_prob = np.zeros((144, 192), dtype=np.float32)
        self.calls = []

    def infer(self, rgb, depth, class_name, color_name, cam_type, conf_thresh=0.5):
        self.calls.append((class_name, color_name, cam_type))
        present, conf, dist, bearing = self._script[min(self._i, len(self._script) - 1)]
        self._i += 1
        self.last_heat_prob = np.full((144, 192), conf, dtype=np.float32)
        return dict(present=present, confidence=conf, dist_m=dist,
                    bearing_deg=bearing, peak_px=(96.0, 72.0))


class _FakeViz:
    """Minimal FleetViz stand-in: a fixed pelvis xpos and a dummy data handle."""

    def __init__(self, pelvis=(0.0, 0.0, 0.744)):
        self._pelvis = np.array(pelvis, dtype=float)
        self.data = object()

    def pelvis_xpos(self, name):
        return self._pelvis.copy()


class _FakeRenderer:
    def render(self, data, pelvis_xyz, yaw):
        rgb = np.zeros((360, 480, 3), dtype=np.uint8)
        depth = np.full((360, 480), 3.0, dtype=np.float32)
        return rgb, depth, {"is_proximity": False, "is_widefov": False}


def _perception(callsign, script, *, detector=None, tau=0.5, hysteresis=False,
                tau_track=None, pelvis=(0.0, 0.0, 0.744)):
    det = detector if detector is not None else _FakeDetector(script)
    return RobotPerception(callsign, _FakeViz(pelvis), detector=det,
                           renderer=_FakeRenderer(), tau=tau,
                           hysteresis=hysteresis, tau_track=tau_track,
                           class_names=_CLASS, color_names=_COLOR), det


# ---------------------------------------------------------------------------
# World-xy conversion math
# ---------------------------------------------------------------------------
class TestWorldXyMath(unittest.TestCase):
    def test_straight_ahead(self):
        self.assertAlmostEqual(detection_world_xy((0, 0), 0.0, 2.0, 0.0)[0], 2.0)
        self.assertAlmostEqual(detection_world_xy((0, 0), 0.0, 2.0, 0.0)[1], 0.0)

    def test_facing_plus_y(self):
        x, y = detection_world_xy((0, 0), math.pi / 2, 2.0, 0.0)
        self.assertAlmostEqual(x, 0.0, places=6)
        self.assertAlmostEqual(y, 2.0, places=6)

    def test_bearing_is_ccw_left(self):
        # +bearing = target to the LEFT (CCW) of heading.
        x, y = detection_world_xy((1.0, 1.0), 0.0, 1.0, 90.0)
        self.assertAlmostEqual(x, 1.0, places=6)
        self.assertAlmostEqual(y, 2.0, places=6)

    def test_oracle_range_bearing_roundtrip(self):
        d, b = oracle_range_bearing((0, 0), 0.0, (0.0, 3.0))
        self.assertAlmostEqual(d, 3.0)
        self.assertAlmostEqual(b, 90.0)
        # A detection at that (d, b) back-projects to the object.
        x, y = detection_world_xy((0, 0), 0.0, d, b)
        self.assertAlmostEqual(x, 0.0, places=6)
        self.assertAlmostEqual(y, 3.0, places=6)


# ---------------------------------------------------------------------------
# confirm() with a fake detector
# ---------------------------------------------------------------------------
class TestConfirm(unittest.TestCase):
    def test_confirms_above_tau(self):
        p, det = _perception("Alpha", [(True, 0.9, 3.0, 0.0)], tau=0.5)
        res = p.confirm(ObjectQuery("red", "cube"), (0.0, 0.0, 0.0))
        self.assertIsInstance(res, DetectionResult)
        self.assertAlmostEqual(res.dist_m, 3.0)
        self.assertAlmostEqual(res.world_xy[0], 3.0, places=3)
        self.assertEqual(res.source, "groundnet")
        self.assertEqual(det.calls[0], ("cube", "red", "grounding"))

    def test_none_below_tau(self):
        p, _ = _perception("Alpha", [(False, 0.2, 3.0, 0.0)], tau=0.5)
        self.assertIsNone(p.confirm(ObjectQuery("red", "cube"), (0.0, 0.0, 0.0)))

    def test_none_for_wildcard_query(self):
        p, det = _perception("Alpha", [(True, 0.9, 3.0, 0.0)], tau=0.5)
        self.assertIsNone(p.confirm(ObjectQuery("red", None), (0.0, 0.0, 0.0)))
        self.assertIsNone(p.confirm(ObjectQuery(None, "cube"), (0.0, 0.0, 0.0)))
        self.assertEqual(det.calls, [])  # never rendered/ran the detector

    def test_geometry_gate_accepts_consistent(self):
        # Detector agrees with the oracle hypothesis (object dead ahead at 3 m).
        p, _ = _perception("Alpha", [(True, 0.9, 3.0, 0.0)], tau=0.5)
        res = p.confirm(ObjectQuery("red", "cube"), (0.0, 0.0, 0.0),
                        oracle_xy=(3.0, 0.0))
        self.assertIsNotNone(res)

    def test_geometry_gate_rejects_inconsistent_distance(self):
        p, _ = _perception("Alpha", [(True, 0.9, 3.0, 0.0)], tau=0.5)
        far = 3.0 + CONFIRM_DIST_GATE_M + 1.0
        self.assertIsNone(p.confirm(ObjectQuery("red", "cube"), (0.0, 0.0, 0.0),
                                    oracle_xy=(far, 0.0)))

    def test_geometry_gate_rejects_inconsistent_bearing(self):
        # Detector peaks dead ahead but the oracle hypothesis is off to the side.
        p, _ = _perception("Alpha", [(True, 0.9, 3.0, 0.0)], tau=0.5)
        ang = math.radians(CONFIRM_BEARING_GATE_DEG + 15.0)
        oracle_xy = (3.0 * math.cos(ang), 3.0 * math.sin(ang))
        self.assertIsNone(p.confirm(ObjectQuery("red", "cube"), (0.0, 0.0, 0.0),
                                    oracle_xy=oracle_xy))

    def test_confirmation_is_popped_once(self):
        p, _ = _perception("Alpha", [(True, 0.9, 3.0, 0.0)], tau=0.5)
        p.confirm(ObjectQuery("red", "cube"), (0.0, 0.0, 0.0))
        self.assertIsNotNone(p.pop_confirmation())
        self.assertIsNone(p.pop_confirmation())  # drained

    def test_reset_clears_state(self):
        p, _ = _perception("Alpha", [(True, 0.9, 3.0, 0.0)], tau=0.5,
                           hysteresis=True)
        p.confirm(ObjectQuery("red", "cube"), (0.0, 0.0, 0.0))
        self.assertIsNotNone(p.last_confirmation)
        p.reset()
        self.assertIsNone(p.last_confirmation)
        self.assertIsNone(p._state.track_dist_m)


# ---------------------------------------------------------------------------
# Per-robot GroundNetState isolation (singleton-bug regression)
# ---------------------------------------------------------------------------
class TestPerRobotIsolation(unittest.TestCase):
    def test_shared_weights_separate_track_state(self):
        # ONE shared detector object, TWO robots, hysteresis on (tau_track < tau).
        shared = _FakeDetector([(True, 0.9, 3.0, 0.0)])
        a = RobotPerception("Alpha", _FakeViz(), detector=shared,
                            renderer=_FakeRenderer(), tau=0.6, hysteresis=True,
                            tau_track=0.3, class_names=_CLASS, color_names=_COLOR)
        b = RobotPerception("Bravo", _FakeViz(), detector=shared,
                            renderer=_FakeRenderer(), tau=0.6, hysteresis=True,
                            tau_track=0.3, class_names=_CLASS, color_names=_COLOR)
        # Same shared model object underneath both.
        self.assertIs(a._state.detector, b._state.detector)
        # A acquires a track (high confidence); B never has.
        shared._script = [(True, 0.9, 3.0, 0.0)]; shared._i = 0
        self.assertIsNotNone(a.confirm(ObjectQuery("red", "cube"), (0, 0, 0)))
        self.assertAlmostEqual(a._state.track_dist_m, 3.0)
        self.assertIsNone(b._state.track_dist_m)  # ISOLATED

    def test_track_continuation_does_not_leak_across_robots(self):
        shared = _FakeDetector([(True, 0.9, 3.0, 0.0)])
        a = RobotPerception("Alpha", _FakeViz(), detector=shared,
                            renderer=_FakeRenderer(), tau=0.6, hysteresis=True,
                            tau_track=0.3, class_names=_CLASS, color_names=_COLOR)
        b = RobotPerception("Bravo", _FakeViz(), detector=shared,
                            renderer=_FakeRenderer(), tau=0.6, hysteresis=True,
                            tau_track=0.3, class_names=_CLASS, color_names=_COLOR)
        q = ObjectQuery("red", "cube")
        # A: high-conf acquire, then a marginal (tau_track) spatially-continuous
        # continuation -> accepted BECAUSE A holds a track.
        shared._script = [(True, 0.9, 3.0, 0.0)]; shared._i = 0
        self.assertIsNotNone(a.confirm(q, (0, 0, 0)))
        shared._script = [(False, 0.4, 3.1, 1.0)]; shared._i = 0
        self.assertIsNotNone(a.confirm(q, (0, 0, 0)))  # track continuation
        # B: the SAME marginal detection as its FIRST look -> rejected, because
        # B has no track of its own (it did NOT inherit A's). This is the exact
        # cross-contamination the singleton bug caused.
        shared._script = [(False, 0.4, 3.1, 1.0)]; shared._i = 0
        self.assertIsNone(b.confirm(q, (0, 0, 0)))


# ---------------------------------------------------------------------------
# Oracle / groundnet mode switch in can_see
# ---------------------------------------------------------------------------
class _FakeUnit:
    def __init__(self, xy, yaw):
        self._xy, self._yaw = xy, yaw
        self.done = self.fell = False

    @property
    def xy(self):
        return self._xy

    @property
    def yaw(self):
        return self._yaw

    @property
    def base_height(self):
        return 0.74

    def assign_goal(self, xy):
        return True

    def halt(self):
        pass


class _FakeCarry:
    def is_carried(self, i):
        return False


class _FakePerception:
    """Returns a scripted DetectionResult (or None) and records the oracle_xy."""

    def __init__(self, result):
        self._result = result
        self.seen = []

    def confirm(self, query, robot_pose, viz=None, *, oracle_xy=None):
        self.seen.append(oracle_xy)
        return self._result


def _open_scene():
    from code.warehouse.arena import warehouse_scene_cfg
    from code.warehouse.layout import hero_layout
    from code.sim.arena_build import COLORS
    cmap = dict(COLORS)
    objs = [{"color_name": "red", "color_rgb": cmap["red"], "shape_name": "cube",
             "size": 0.24, "x": -5.0, "y": 0.5}]
    return warehouse_scene_cfg(hero_layout(), objects=objs,
                               rng=np.random.default_rng(0))


class TestModeSwitch(unittest.TestCase):
    def _actions(self, perception):
        return FleetRobotActions("Alpha", _FakeUnit((-5.0, -5.0), math.pi / 2),
                                 _open_scene(), None, _FakeCarry(),
                                 perception=perception)

    def test_oracle_mode_returns_exact_oracle_xy(self):
        act = FleetRobotActions("Alpha", _FakeUnit((-5.0, -5.0), math.pi / 2),
                                _open_scene(), None, _FakeCarry())  # no perception
        self.assertEqual(act.can_see(ObjectQuery("red", "cube")), (-5.0, 0.5))
        self.assertEqual(act.last_see_source, "oracle")

    def test_groundnet_mode_reports_detector_xy(self):
        det = DetectionResult(world_xy=(-4.8, 0.55), dist_m=5.5, bearing_deg=1.0,
                              confidence=0.3, source="groundnet")
        act = self._actions(_FakePerception(det))
        loc = act.can_see(ObjectQuery("red", "cube"))
        self.assertEqual(loc, (-4.8, 0.55))       # DETECTOR xy, not oracle xy
        self.assertEqual(act.last_see_source, "detector")

    def test_groundnet_miss_falls_back_to_oracle_xy(self):
        act = self._actions(_FakePerception(None))  # detector misses
        loc = act.can_see(ObjectQuery("red", "cube"))
        self.assertEqual(loc, (-5.0, 0.5))         # oracle fallback
        self.assertEqual(act.last_see_source, "oracle_fallback")

    def test_confirm_receives_oracle_hypothesis(self):
        fp = _FakePerception(None)
        self._actions(fp).can_see(ObjectQuery("red", "cube"))
        self.assertEqual(fp.seen, [(-5.0, 0.5)])   # oracle_xy passed to confirm


# ---------------------------------------------------------------------------
# Fetch endgame absorbs detector xy error (item 3)
# ---------------------------------------------------------------------------
class TestEndgameTolerance(unittest.TestCase):
    def test_typical_detector_error_within_pickup_radius(self):
        from code.fleet.carry import PICKUP_RADIUS_M
        # A ~0.3 m detector estimate error is absorbed by the 0.6 m pickup radius.
        obj = (6.5, 4.7)
        est = detection_world_xy((6.5, 1.4), math.pi / 2, 3.30, 0.0)  # ~0.0 m off
        self.assertLess(math.hypot(est[0] - obj[0], est[1] - obj[1]),
                        PICKUP_RADIUS_M)
        # A deliberately 0.3 m-offset estimate is still inside the radius.
        offset = (obj[0] + 0.3, obj[1])
        self.assertLess(math.hypot(offset[0] - obj[0], offset[1] - obj[1]),
                        PICKUP_RADIUS_M)

    def test_large_error_needs_reapproach(self):
        from code.fleet.carry import PICKUP_RADIUS_M
        # A 0.7 m error exceeds the radius -> pickup misses, the retry+approach
        # path (protocol MAX_PICKUP_RETRIES) is what recovers it.
        obj = (6.5, 4.7)
        offset = (obj[0] + 0.7, obj[1])
        self.assertGreater(math.hypot(offset[0] - obj[0], offset[1] - obj[1]),
                           PICKUP_RADIUS_M)


if __name__ == "__main__":
    unittest.main()
