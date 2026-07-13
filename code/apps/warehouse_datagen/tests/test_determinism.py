"""Determinism test: the same (seed, episode_index) yields byte-identical labels.

Runs one short warehouse DART episode twice (render off for speed) with the same
plan + per-episode noise RNG and asserts the recorded proprio/action/goal/vel_cmd
are identical — the property episode-level idempotency relies on.
"""

from __future__ import annotations

import unittest

import numpy as np

from code.scene import derive_rng
from code.teacher import WBCTeacher
from code.apps.warehouse_datagen.scene import sample_episode_plan
from code.apps.warehouse_datagen.rollout import run_warehouse_dart_episode

_SEED = 5
_EP = 0
_NOISE_OFFSET = 10000
_MAXSTEPS = 40


def _run_once(teacher: WBCTeacher) -> list:
    plan = sample_episode_plan(_SEED, _EP, primitive_frac=1.0)  # primitive = fast, no follower
    noise_rng = derive_rng(_SEED + _NOISE_OFFSET, _EP)
    result = run_warehouse_dart_episode(
        teacher, plan, episode_idx=_EP, global_frame_offset=0,
        noise_sigma=0.07, hard_maxsteps=_MAXSTEPS, rng_noise=noise_rng, render=False,
    )
    assert result is not None
    return result["rows"]


class DeterminismTest(unittest.TestCase):
    def test_identical_across_two_runs(self) -> None:
        teacher = WBCTeacher()
        rows0 = _run_once(teacher)
        rows1 = _run_once(teacher)
        self.assertEqual(len(rows0), len(rows1))
        self.assertGreater(len(rows0), 0)
        for r0, r1 in zip(rows0, rows1):
            for key in ("action", "proprio", "goal", "vel_cmd", "phase"):
                np.testing.assert_allclose(
                    np.array(r0[key]), np.array(r1[key]), atol=1e-6,
                    err_msg=f"mismatch in {key} at frame {r0['frame_index']}")
            self.assertEqual(r0["done"], r1["done"])


if __name__ == "__main__":
    unittest.main()
