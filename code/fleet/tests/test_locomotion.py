"""Tests for the F5 VLA locomotion threading through the fleet.

Three layers, cheapest first:

* Backend-parameter validation (no sim, no torch): the ``locomotion`` default
  stays ``"teacher"`` on ``RobotUnit`` / ``Fleet`` / ``MissionRunner`` (existing
  evals untouched), a bad ``locomotion`` string fails fast before any physics,
  and ``resolve_vla_ckpt`` honours explicit > env > default.
* Checkpoint validation: ``load_shared_vla_policy`` / ``make_unit_vla_backend``
  surface a missing checkpoint as ``FileNotFoundError``.
* A short closed-loop fleet-on-VLA smoke (skip-if-no-ckpt / no-assets): all four
  robots SHARE one loaded policy model, each keeps its OWN proprio window, and a
  brief run stays upright.
"""

from __future__ import annotations

import inspect
import os
import unittest
from pathlib import Path

from code.fleet.fleet import Fleet
from code.fleet.locomotion import (DEFAULT_VLA_CKPT, clear_shared_policies,
                                   make_unit_vla_backend, resolve_vla_ckpt)
from code.fleet.mission import MissionRunner
from code.fleet.robot_unit import RobotUnit

_REPO = Path(__file__).resolve().parents[3]
_CKPT = _REPO / "runs" / "warehouse_dart_ft_A" / "model_best.pt"


# ---------------------------------------------------------------------------
# Parameter validation (no simulator / no torch).
# ---------------------------------------------------------------------------
class TestLocomotionDefaults(unittest.TestCase):
    def test_robot_unit_default_is_teacher(self) -> None:
        d = inspect.signature(RobotUnit.__init__).parameters["locomotion"].default
        self.assertEqual(d, "teacher")

    def test_fleet_default_is_teacher(self) -> None:
        d = inspect.signature(Fleet.__init__).parameters["locomotion"].default
        self.assertEqual(d, "teacher")

    def test_mission_runner_default_is_teacher(self) -> None:
        d = inspect.signature(MissionRunner.__init__).parameters["locomotion"].default
        self.assertEqual(d, "teacher")

    def test_robot_unit_rejects_bad_locomotion_before_sim(self) -> None:
        """A bad locomotion string raises before any scene_cfg/teacher is touched."""
        with self.assertRaises(ValueError):
            RobotUnit("Alpha", None, (0.0, 0.0), 0.0, locomotion="bogus")


class TestResolveVlaCkpt(unittest.TestCase):
    def test_explicit_wins(self) -> None:
        self.assertEqual(resolve_vla_ckpt("/x/y.pt"), "/x/y.pt")

    def test_env_over_default(self) -> None:
        old = os.environ.get("VLA_CKPT")
        os.environ["VLA_CKPT"] = "/env/ckpt.pt"
        try:
            self.assertEqual(resolve_vla_ckpt(None), "/env/ckpt.pt")
        finally:
            if old is None:
                os.environ.pop("VLA_CKPT", None)
            else:
                os.environ["VLA_CKPT"] = old

    def test_default_when_unset(self) -> None:
        old = os.environ.pop("VLA_CKPT", None)
        try:
            self.assertEqual(resolve_vla_ckpt(None), DEFAULT_VLA_CKPT)
        finally:
            if old is not None:
                os.environ["VLA_CKPT"] = old


class TestCkptValidation(unittest.TestCase):
    def setUp(self) -> None:
        clear_shared_policies()

    def tearDown(self) -> None:
        clear_shared_policies()

    def test_missing_ckpt_raises(self) -> None:
        with self.assertRaises(FileNotFoundError):
            make_unit_vla_backend("/no/such/checkpoint.pt")


# ---------------------------------------------------------------------------
# Short closed-loop fleet-on-VLA smoke.
# ---------------------------------------------------------------------------
@unittest.skipUnless(_CKPT.exists(), f"VLA checkpoint absent: {_CKPT}")
class TestFleetVlaSharedModel(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        clear_shared_policies()
        try:
            from code.warehouse.layout import rooms_layout
            cls.layout = rooms_layout()
        except Exception as e:  # pragma: no cover - environment-dependent
            raise unittest.SkipTest(f"MuJoCo assets unavailable: {e}")

    @classmethod
    def tearDownClass(cls) -> None:
        clear_shared_policies()

    def _goals(self):
        s = self.layout.object_spots
        return {"Alpha": (float(s[5][0]), float(s[5][1])),
                "Delta": (float(s[2][0]), float(s[2][1]))}

    def test_fleet_shares_one_model_with_per_unit_windows(self) -> None:
        fleet = Fleet(self.layout, self._goals(), build_viz=False,
                      locomotion="vla", seed=0)
        try:
            units = list(fleet.units.values())
            # Every robot's VLA backend references the SAME loaded model object.
            models = [u._nav.vla.model for u in units]
            self.assertTrue(all(m is models[0] for m in models),
                            "fleet robots must share ONE loaded policy model")
            # But each robot owns a distinct backend + proprio window.
            self.assertEqual(len({id(u._nav.vla) for u in units}), len(units))
            windows = [id(u._nav.vla.proprio_hist) for u in units]
            self.assertEqual(len(set(windows)), len(units),
                             "each robot must keep its own proprio window")
            # A short run steps the shared policy; robots stay upright.
            for _ in range(120):
                fleet.step_all()
            self.assertFalse(fleet.any_fell)
            self.assertGreater(fleet.mean_vla_infer_ms(), 0.0)
            self.assertTrue(all(u.base_height > 0.5 for u in units))
        finally:
            fleet.close()

    def test_teacher_fleet_loads_no_vla(self) -> None:
        """A default (teacher) fleet never builds a VLA backend."""
        fleet = Fleet(self.layout, self._goals(), build_viz=False, seed=0)
        try:
            self.assertEqual(fleet.locomotion, "teacher")
            self.assertTrue(all(getattr(u._nav, "vla", None) is None
                                for u in fleet.units.values()))
            self.assertEqual(fleet.mean_vla_infer_ms(), 0.0)
        finally:
            fleet.close()


if __name__ == "__main__":
    unittest.main()
