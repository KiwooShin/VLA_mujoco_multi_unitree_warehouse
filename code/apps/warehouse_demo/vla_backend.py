"""vla_backend.py — Trained-VLA locomotion backend for :class:`StepwiseNav` (F5).

The teacher backend drives each 50 Hz control step through
``WBCTeacher.step(vel_cmd)`` (the Unitree whole-body controller). This module
provides the alternative demanded by ``docs/final_demo_spec.md`` §F5: locomotion
must come from the TRAINED distilled GroundedNav policy lineage, not from driving
the WBC directly.

:class:`VlaBackend` replicates the baseline closed-loop deploy path
(``code/runtime/rollout_step.py`` / ``rollout_state.py``) one control step at a
time, for the ``goal_source='gt' + vel_source='gt'`` injection lineage this
fine-tune was trained on:

  * proprio is built exactly as the runtime does (``_build_proprio`` → 55-d) and,
    for a phase-conditioned checkpoint, the gait phase ``[sin φ, cos φ]`` is
    appended (→ 57-d) from a :class:`_GaitPhaseTracker`;
  * a ``PROPRIO_K``-frame rolling window (current frame last) feeds the GRU;
  * ego RGB is fed as ZEROS and the language embedding as ZEROS — matching the
    phase trainer (``code/train/gaitfix_epoch.py`` zeroes vision; the phase
    dataset feeds a zero lang emb), because locomotion is vision-independent
    for this lineage;
  * the steer velocity (``vel_cmd``) and the egocentric goal (``goal_vec``) are
    INJECTED as ``gt_vel`` / ``gt_goal`` (teacher-forced), exactly the two
    quantities the warehouse DART datagen logged per frame
    (``code/apps/warehouse_datagen/rollout.py``: ``vel`` and
    ``goal_vec(dist, yaw_err)``);
  * the model's raw action is de-standardised back to absolute joint targets
    (residual mode: ``default + raw·std + mean``) and executed through the SAME
    student PD path the runtime uses (``_apply_student_pd`` for
    ``CONTROL_DECIMATION`` substeps).

The backend owns NO simulator: it steps the ``MjData`` that :class:`StepwiseNav`
(hence the ``WBCTeacher``) already owns, so all baseline qpos slicing stays
valid. Checkpoint parsing / model construction is delegated verbatim to
``code.runtime.io.build_model`` so the ``arch`` / ``proprio_dim=57`` /
``action_stats`` / teacher-forcing wiring can never drift from the baseline.
"""

from __future__ import annotations

import collections
import os
import time
from typing import Optional, Sequence

import mujoco
import numpy as np
import torch

from code.control.steer import goal_vec
from code.runtime.constants import IMG_SIZE, PROPRIO_DIM, PROPRIO_DIM_PHASE, PROPRIO_K
from code.runtime.gait_phase import _GaitPhaseTracker
from code.runtime.helpers import _apply_student_pd, _build_proprio
from code.runtime.io import build_model
from code.sim.teacher import CONTROL_DECIMATION

__all__ = ["VlaBackend", "build_phase_frame", "stack_proprio_window"]


# ---------------------------------------------------------------------------
# Pure window-assembly helpers (unit-tested against a golden dataset slice).
# ---------------------------------------------------------------------------
def build_phase_frame(proprio55: np.ndarray, phase2: np.ndarray | None) -> np.ndarray:
    """Concatenate a 55-d proprio vector with a 2-d gait phase → 57-d frame.

    Mirrors ``code/data/dataset_phase.py``'s per-row assembly
    (``np.concatenate([p55, ph])``): when ``phase2`` is None a zero phase
    ``[0, 0]`` is appended (the dataset's missing-phase fallback).

    Args:
        proprio55: (55,) proprio vector.
        phase2: (2,) ``[sin φ, cos φ]`` gait phase, or None.

    Returns:
        (57,) float32 frame ``[proprio55, phase2]``.
    """
    p55 = np.asarray(proprio55, dtype=np.float32)
    ph = (np.zeros(2, dtype=np.float32) if phase2 is None
          else np.asarray(phase2, dtype=np.float32))
    return np.concatenate([p55, ph]).astype(np.float32)


def stack_proprio_window(frames: Sequence[np.ndarray]) -> np.ndarray:
    """Stack a sequence of per-frame proprio vectors into a ``(K, D)`` window.

    Mirrors the dataset's ``torch.from_numpy(np.stack(frames))`` (oldest frame
    first, most-recent last), so the deploy window and the training window are
    assembled identically.

    Args:
        frames: Sequence of (D,) float32 frames.

    Returns:
        (K, D) float32 array.
    """
    return np.stack([np.asarray(f, dtype=np.float32) for f in frames], axis=0)


# ---------------------------------------------------------------------------
# VLA locomotion backend
# ---------------------------------------------------------------------------
class VlaBackend:
    """Runs the trained GroundedNav policy as a stepwise locomotion backend.

    One instance drives ONE robot. Construct once (loads the checkpoint), call
    :meth:`reset` after the settle phase to prime the proprio window, then call
    :meth:`step` once per 50 Hz control step with the steer velocity + goal that
    :class:`StepwiseNav` already computes for the teacher backend.

    Args:
        ckpt_path: Path to a GroundedNav ``.pt`` fine-tune checkpoint (arch A,
            phase-conditioned, residual action_stats embedded).
        device: 'cuda' | 'cpu' | None. None → 'cuda' if available else 'cpu'.
        arch: Model architecture ('A' — the only lineage with gt goal/vel heads).

    Raises:
        FileNotFoundError: If ``ckpt_path`` does not exist.
    """

    def __init__(
        self,
        ckpt_path: str,
        device: Optional[str] = None,
        arch: str = "A",
    ) -> None:
        if not ckpt_path or not os.path.isfile(ckpt_path):
            raise FileNotFoundError(f"VLA checkpoint not found: {ckpt_path!r}")
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.ckpt_path = ckpt_path

        # Delegate all checkpoint parsing + model wiring to the baseline loader.
        # goal_source='gt' + vel_source='gt' → teacher_forcing=True so the
        # injected gt_goal / gt_vel replace the grounding/velocity heads, exactly
        # as this lineage was trained (RGB + lang zeroed, goal + vel supplied).
        lr = build_model(
            checkpoint_path=ckpt_path, arch=arch, chunk_H=1,
            device=self.device, goal_source="gt", vel_source="gt",
        )
        if not lr.checkpoint_loaded:
            raise ValueError(f"VLA checkpoint could not be loaded: {ckpt_path!r}")
        self.model = lr.model            # already .eval()
        self.arch = lr.arch
        self.use_phase = lr.use_phase
        self.eff_proprio_dim = PROPRIO_DIM_PHASE if self.use_phase else PROPRIO_DIM

        # Residual de-normalisation arrays (Fix 1): abs = default + raw·std + mean.
        self._use_residual = lr.action_stats is not None
        if self._use_residual:
            self._da_mean = np.asarray(lr.action_stats["mean"], dtype=np.float32)
            self._da_std = np.asarray(lr.action_stats["std"], dtype=np.float32)
            self._da_deflt = np.asarray(lr.action_stats["default_angles"], dtype=np.float32)

        # Constant zero inputs (vision + language off for this lineage).
        self._img_zeros = torch.zeros(1, 3, IMG_SIZE, IMG_SIZE,
                                      dtype=torch.float32, device=self.device)
        self._lang_zeros = torch.zeros(1, 2048, dtype=torch.float32, device=self.device)

        # Per-episode state (populated by reset()).
        self.phase_tracker: Optional[_GaitPhaseTracker] = None
        self.prev_action: Optional[np.ndarray] = None
        self.proprio_hist: Optional[collections.deque] = None
        self._infer_ms: list = []
        self._step_ms: list = []

    # ------------------------------------------------------------------
    def reset(self, data_mj: mujoco.MjData, init_target_dof: np.ndarray) -> None:
        """Prime the proprio window from the settled physics state.

        Mirrors ``code/runtime/rollout_state.py``: seed a ``PROPRIO_K``-deep
        deque with the current proprio frame so the GRU sees a full window from
        step 0 (no zero-padding transient).

        Args:
            data_mj: The robot's ``MjData`` (post-settle standing pose).
            init_target_dof: (15,) last settle joint targets, used as the initial
                ``prev_action`` fed into the proprio vector.
        """
        self.phase_tracker = _GaitPhaseTracker() if self.use_phase else None
        self.prev_action = np.asarray(init_target_dof, dtype=np.float32).copy()
        self.proprio_hist = collections.deque(
            [np.zeros(self.eff_proprio_dim, dtype=np.float32)] * PROPRIO_K,
            maxlen=PROPRIO_K,
        )
        prop_now = self._assemble_proprio(data_mj)
        for _ in range(PROPRIO_K):
            self.proprio_hist.append(prop_now.copy())
        self._infer_ms = []
        self._step_ms = []

    # ------------------------------------------------------------------
    def _assemble_proprio(self, data_mj: mujoco.MjData) -> np.ndarray:
        """Build this step's proprio (+phase) frame from the physics state."""
        p = _build_proprio(data_mj, self.prev_action)   # (55,)
        if self.use_phase:
            ph = self.phase_tracker.update(data_mj.qpos[7:22].copy())  # (2,)
            p = build_phase_frame(p, ph)                # (57,)
        return p

    # ------------------------------------------------------------------
    def step(
        self,
        model_mj: mujoco.MjModel,
        data_mj: mujoco.MjData,
        nj: int,
        goal_vec3: np.ndarray,
        vel_cmd3: np.ndarray,
    ) -> np.ndarray:
        """Advance one 50 Hz control step under the trained policy.

        Runs the student forward pass with the injected egocentric goal + steer
        velocity, de-standardises the raw action to absolute joint targets, and
        executes them through the student PD path for ``CONTROL_DECIMATION``
        physics substeps (mutating ``data_mj`` in place).

        Args:
            model_mj: The robot's ``MjModel``.
            data_mj: The robot's ``MjData`` (stepped in place).
            nj: Total actuated joints (``teacher._nj``).
            goal_vec3: (3,) injected ``gt_goal`` ``[dist, cos θ, sin θ]``.
            vel_cmd3: (3,) injected ``gt_vel`` ``[vx, vy, ωz]`` (steer command).

        Returns:
            (15,) absolute joint targets commanded this step.
        """
        if self.proprio_hist is None:
            raise RuntimeError("VlaBackend.step called before reset()")

        # ---- Assemble proprio window (current frame appended last) ----
        prop_now = self._assemble_proprio(data_mj)
        self.proprio_hist.append(prop_now)
        prop_arr = stack_proprio_window(self.proprio_hist)          # (K, 57)
        prop_t = torch.from_numpy(prop_arr).unsqueeze(0).to(self.device)
        goal_t = torch.from_numpy(
            np.asarray(goal_vec3, dtype=np.float32)).unsqueeze(0).to(self.device)
        vel_t = torch.from_numpy(
            np.asarray(vel_cmd3, dtype=np.float32)).unsqueeze(0).to(self.device)

        # ---- Student forward pass (vision + lang zeroed; goal/vel injected) ----
        t0 = time.perf_counter()
        with torch.no_grad():
            out = self.model(
                ego_rgb=self._img_zeros,
                lang_emb=self._lang_zeros,
                proprio_h=prop_t,
                gt_goal=goal_t,
                gt_vel=vel_t,
            )
        raw_action = out["action"][0, 0].detach().cpu().numpy()     # (15,)
        self._infer_ms.append((time.perf_counter() - t0) * 1000.0)

        # ---- Raw output → absolute joint targets ----
        if self._use_residual:
            target_dof = self._da_deflt + raw_action * self._da_std + self._da_mean
        else:
            target_dof = raw_action
        target_dof = target_dof.astype(np.float32)

        # ---- Student PD → physics substeps (no teacher in the loop) ----
        for _ in range(CONTROL_DECIMATION):
            _apply_student_pd(data_mj, target_dof, nj)
            mujoco.mj_step(model_mj, data_mj)

        self.prev_action = target_dof.copy()
        self._step_ms.append((time.perf_counter() - t0) * 1000.0)
        return target_dof

    # ------------------------------------------------------------------
    @property
    def mean_infer_ms(self) -> float:
        """Mean policy forward-pass time (ms) over the episode (0.0 if unused)."""
        return float(np.mean(self._infer_ms)) if self._infer_ms else 0.0

    @property
    def mean_step_ms(self) -> float:
        """Mean full control-step time (forward + PD substeps) in ms."""
        return float(np.mean(self._step_ms)) if self._step_ms else 0.0
