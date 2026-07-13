"""Tests for the F5 VLA locomotion backend.

Three layers, each isolated so the cheap ones always run:

* Backend-switch validation (no simulator, no torch): the ``StepwiseNav``
  constructor rejects a bad backend string, requires a checkpoint for
  ``backend="vla"``, surfaces a missing-checkpoint file as ``FileNotFoundError``
  (all fail-fast, before any arena is compiled), and still defaults to
  ``"teacher"`` (nothing existing changes behaviour).
* Proprio/phase window assembly against a golden slice from the real warehouse
  DART dataset: the pure helpers (:func:`build_phase_frame`,
  :func:`stack_proprio_window`) must reproduce, bit-for-bit, the ``(K, 57)``
  window that ``PhaseParquetDataset`` feeds the model in training.
* A short closed-loop VLA sim smoke (skip-if-no-ckpt / no-assets).
"""

from __future__ import annotations

import inspect
import os
import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np

from code.apps.warehouse_demo.nav_core import StepwiseNav
from code.apps.warehouse_demo.vla_backend import (
    VlaBackend,
    build_phase_frame,
    stack_proprio_window,
)

_REPO = Path(__file__).resolve().parents[4]
_DATASET = _REPO / "runs" / "warehouse_dart" / "2026-07-12"
_EP0 = _DATASET / "data" / "chunk-000" / "episode_000000.parquet"
_BEST_CKPT = _REPO / "runs" / "warehouse_dart_ft_A" / "model_best.pt"
_PROPRIO_K = 6


# ---------------------------------------------------------------------------
# Backend-switch validation (no simulator / no torch model built).
# ---------------------------------------------------------------------------
class TestBackendValidation(unittest.TestCase):
    def test_default_backend_is_teacher(self) -> None:
        """The constructor default is 'teacher' — existing callers unchanged."""
        default = inspect.signature(StepwiseNav.__init__).parameters["backend"].default
        self.assertEqual(default, "teacher")

    def test_bad_backend_raises_before_any_sim(self) -> None:
        """A bad backend string fails fast (scene_cfg/teacher never touched)."""
        with self.assertRaises(ValueError):
            StepwiseNav(None, (0.0, 0.0), 0.0, teacher=None, backend="bogus")

    def test_vla_requires_ckpt(self) -> None:
        """backend='vla' without a checkpoint is a ValueError (before sim)."""
        with self.assertRaises(ValueError):
            StepwiseNav(None, (0.0, 0.0), 0.0, teacher=None,
                        backend="vla", vla_ckpt=None)

    def test_vla_missing_ckpt_file_raises(self) -> None:
        """A non-existent checkpoint path surfaces as FileNotFoundError."""
        with self.assertRaises(FileNotFoundError):
            StepwiseNav(None, (0.0, 0.0), 0.0, teacher=None,
                        backend="vla", vla_ckpt="/no/such/checkpoint.pt")

    def test_vla_backend_missing_ckpt_direct(self) -> None:
        """VlaBackend itself rejects a missing checkpoint (no torch load)."""
        with self.assertRaises(FileNotFoundError):
            VlaBackend("/no/such/checkpoint.pt")


# ---------------------------------------------------------------------------
# Pure window-assembly helpers.
# ---------------------------------------------------------------------------
class TestPhaseFrameHelper(unittest.TestCase):
    def test_concat_shape_and_order(self) -> None:
        p55 = np.arange(55, dtype=np.float32)
        ph = np.array([0.3, 0.9], dtype=np.float32)
        f = build_phase_frame(p55, ph)
        self.assertEqual(f.shape, (57,))
        self.assertEqual(f.dtype, np.float32)
        np.testing.assert_array_equal(f[:55], p55)
        np.testing.assert_array_equal(f[55:], ph)

    def test_missing_phase_falls_back_to_zeros(self) -> None:
        f = build_phase_frame(np.ones(55, dtype=np.float32), None)
        np.testing.assert_array_equal(f[55:], np.zeros(2, dtype=np.float32))

    def test_stack_window_shape_and_order(self) -> None:
        frames = [np.full(57, i, dtype=np.float32) for i in range(_PROPRIO_K)]
        w = stack_proprio_window(frames)
        self.assertEqual(w.shape, (_PROPRIO_K, 57))
        # Oldest first, most-recent last (matches the dataset stack order).
        self.assertEqual(w[0, 0], 0.0)
        self.assertEqual(w[-1, 0], float(_PROPRIO_K - 1))


@unittest.skipUnless(_EP0.exists(), "warehouse DART dataset not present")
class TestProprioWindowGolden(unittest.TestCase):
    """Reproduce PhaseParquetDataset's proprio_h window from the raw parquet."""

    def test_matches_dataset_window(self) -> None:
        import pandas as pd
        from code.data.dataset_phase import PhaseParquetDataset

        # Build a 1-episode temp repo so the dataset instantiates fast.
        tmp = tempfile.mkdtemp(prefix="vla_golden_")
        try:
            chunk = Path(tmp) / "data" / "chunk-000"
            chunk.mkdir(parents=True)
            shutil.copy(_EP0, chunk / "episode_000000.parquet")

            ds = PhaseParquetDataset(repo_paths=tmp, split="train",
                                     train_fraction=0.9, proprio_K=_PROPRIO_K)
            sample = ds[0]                       # flat index 0 -> (ep 0, t=K)
            got = sample["proprio_h"].numpy()    # (K, 57)

            # Golden reconstruction directly from the parquet rows [t-K:t] with
            # the module-under-test's own helpers.
            df = pd.read_parquet(_EP0)
            t = _PROPRIO_K
            frames = [
                build_phase_frame(np.asarray(df["proprio"].iloc[i], dtype=np.float32),
                                  np.asarray(df["phase"].iloc[i], dtype=np.float32))
                for i in range(t - _PROPRIO_K, t)
            ]
            expected = stack_proprio_window(frames)

            self.assertEqual(got.shape, (_PROPRIO_K, 57))
            np.testing.assert_allclose(got, expected, rtol=0, atol=1e-6)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


# ---------------------------------------------------------------------------
# Short closed-loop VLA sim smoke (skip-if-no-ckpt / no-assets).
# ---------------------------------------------------------------------------
class TestVlaSimSmoke(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        if not _BEST_CKPT.exists():
            raise unittest.SkipTest(f"VLA checkpoint absent: {_BEST_CKPT}")
        try:
            from code.sim.teacher import WBCTeacher
            from code.warehouse.arena import warehouse_scene_cfg
            from code.warehouse.layout import hero_layout

            cls.teacher = WBCTeacher(use_gpu=False)
            cls.cfg = warehouse_scene_cfg(hero_layout(), robot="Bravo",
                                          rng=np.random.default_rng(0))
        except Exception as e:  # pragma: no cover - environment-dependent
            raise unittest.SkipTest(f"WBC/MuJoCo assets unavailable: {e}")

    def test_short_vla_rollout(self) -> None:
        from code.apps.warehouse_demo.nav_rollout import NavParams, NavResult, run_nav_rollout

        res = run_nav_rollout(
            self.cfg, (-2.0, -3.5), max_steps=80, teacher=self.teacher,
            params=NavParams(), backend="vla", vla_ckpt=str(_BEST_CKPT),
        )
        self.assertIsInstance(res, NavResult)
        self.assertEqual(res.backend, "vla")
        self.assertGreater(res.steps, 0)
        self.assertLessEqual(res.steps, 80)
        self.assertIn(res.outcome, {"success", "timeout"})
        self.assertFalse(res.fell, "policy fell during a short clean rollout")
        # The policy actually ran a forward pass each step.
        self.assertGreater(res.vla_infer_ms, 0.0)


if __name__ == "__main__":
    unittest.main()
