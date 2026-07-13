"""locomotion.py — Fleet-shared VLA locomotion policy (F5 threading).

The fleet's four robots each drive their OWN federated physics through a
:class:`~code.apps.warehouse_demo.nav_core.StepwiseNav`, but when
``locomotion="vla"`` they must all run the SAME trained GroundedNav policy —
loaded ONCE per process and shared across every robot, exactly like the shared
GROUND_NET detector (:func:`code.fleet.perception_bridge.load_shared_detector`).
The policy weights are read-only during inference and the fleet steps robots
sequentially inside :meth:`~code.fleet.fleet.Fleet.step_all`, so a single CUDA
module is safe to share; the ONLY per-robot state is the proprio window (each
unit's own gait history), which stays isolated in a per-robot backend clone.

``code.apps.warehouse_demo.vla_backend.VlaBackend`` (read-only; its per-step API
is complete) couples the checkpoint load with the per-episode proprio state. To
share one loaded model without reloading it four times per fleet — or once per
mission across a whole eval — this module:

* :func:`load_shared_vla_policy` builds a single "template" ``VlaBackend`` and
  caches it process-wide (keyed by resolved checkpoint + device), so the ~90 MB
  GroundedNav weights hit the GPU exactly once;
* :func:`make_unit_vla_backend` hands each robot a lightweight *clone* that
  SHARES the template's ``model`` / de-norm arrays / zero-input tensors but owns
  a fresh, per-robot proprio window (populated by its own ``reset``);
* :func:`attach_vla_to_nav` swaps a settled teacher-mode ``StepwiseNav`` over to
  the VLA backend — reproducing ``StepwiseNav``'s own vla-init (the WBC teacher
  runs the shared settle phase in both modes; the trained policy takes over only
  once the robot is standing, matching the warehouse DART datagen recipe).

Nothing here mutates ``code/apps/warehouse_demo``: it only constructs the public
``VlaBackend`` and assigns its documented public attributes.
"""

from __future__ import annotations

import copy
import os
from typing import Dict, Optional, Tuple

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))

# F5 default checkpoint: the warehouse-domain DART fine-tune (arch A, phase-
# conditioned, residual action_stats). Overridable via the ``VLA_CKPT`` env var
# or an explicit path, mirroring perception_bridge.resolve_ckpt_path.
DEFAULT_VLA_CKPT: str = os.path.join(_REPO_ROOT, "runs", "warehouse_dart_ft_A",
                                     "model_best.pt")

VALID_BACKENDS: Tuple[str, ...] = ("teacher", "vla")

# Process-wide shared-policy cache: (resolved_ckpt, device_str) -> template
# VlaBackend. Loaded once; every per-unit clone references the template's model.
_SHARED_POLICIES: Dict[Tuple[str, str], object] = {}


def resolve_vla_ckpt(ckpt: Optional[str] = None, *,
                     default_ckpt: str = DEFAULT_VLA_CKPT,
                     env_var: str = "VLA_CKPT") -> str:
    """Resolve which GroundedNav checkpoint the fleet should load for VLA.

    Resolution order (highest priority first):

      1. An explicit ``ckpt`` argument (used as-is).
      2. The ``VLA_CKPT`` environment variable (used as-is — a bad path surfaces
         loudly rather than silently falling back).
      3. :data:`DEFAULT_VLA_CKPT` — ``runs/warehouse_dart_ft_A/model_best.pt``.

    Args:
        ckpt: Explicit checkpoint path, or None to fall through to env/default.
        default_ckpt: The default checkpoint (overridable for testing).
        env_var: Environment variable consulted before the default.

    Returns:
        The resolved checkpoint path (not validated for existence here; the
        loader raises :class:`FileNotFoundError` for a missing file).
    """
    if ckpt:
        return ckpt
    env = os.environ.get(env_var)
    if env:
        return env
    return default_ckpt


def _resolve_device(device: Optional[str]) -> str:
    """Resolve a torch device string ('cuda' -> 'cpu' fallback if unavailable)."""
    import torch

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    return device


def load_shared_vla_policy(ckpt: Optional[str] = None,
                           device: Optional[str] = None):
    """Load the trained GroundedNav policy once per process; share the model.

    The same "template" :class:`~code.apps.warehouse_demo.vla_backend.VlaBackend`
    is returned on every call with the same resolved (checkpoint, device) — the
    ~90 MB weights load onto the GPU exactly once. Every per-robot clone
    (:func:`make_unit_vla_backend`) references this template's ``model``, so the
    whole fleet (and every mission in a batch eval running in one process) shares
    ONE loaded policy. Mirrors
    :func:`code.fleet.perception_bridge.load_shared_detector`.

    Args:
        ckpt: Checkpoint path; None -> :func:`resolve_vla_ckpt`.
        device: Torch device ('cuda'|'cpu'|None -> auto, cuda if available).

    Returns:
        The shared template ``VlaBackend`` (its own proprio window is unused; it
        exists only to own the model + de-norm arrays + zero-input tensors).

    Raises:
        FileNotFoundError: If the resolved checkpoint file does not exist.
        ValueError: If the checkpoint could not be loaded as a GroundedNav.
    """
    from code.apps.warehouse_demo.vla_backend import VlaBackend

    ckpt = resolve_vla_ckpt(ckpt)
    dev = _resolve_device(device)
    key = (os.path.abspath(ckpt), dev)
    cached = _SHARED_POLICIES.get(key)
    if cached is None:
        cached = VlaBackend(ckpt, device=dev)
        _SHARED_POLICIES[key] = cached
        print(f"[locomotion] VLA shared policy loaded {ckpt!r} on device={dev!r} "
              f"(use_phase={cached.use_phase}, residual={cached._use_residual})",
              flush=True)
    return cached


def _clone_backend_sharing_model(shared):
    """Clone a ``VlaBackend`` sharing its model but with a fresh proprio window.

    A shallow copy shares every read-only inference input by reference — most
    importantly ``model`` (the CUDA module), plus the residual de-norm arrays and
    the constant zero image/language tensors — while the per-episode/per-unit
    state (proprio window, gait-phase tracker, prev action, timing logs) is reset
    to empty so each robot owns its own gait history. Robust to the backend
    gaining new *shared* attributes (they are shared automatically); only the
    known per-unit fields — the ones :meth:`VlaBackend.reset` repopulates — are
    cleared.
    """
    clone = copy.copy(shared)          # shares model + all read-only tensors
    clone.phase_tracker = None
    clone.prev_action = None
    clone.proprio_hist = None
    clone._infer_ms = []
    clone._step_ms = []
    return clone


def make_unit_vla_backend(ckpt: Optional[str] = None,
                          device: Optional[str] = None,
                          shared=None):
    """Build a per-robot VLA backend that SHARES the process-wide policy model.

    Args:
        ckpt: Checkpoint path (None -> resolved default); ignored if ``shared``
            is given.
        device: Torch device (None -> auto); ignored if ``shared`` is given.
        shared: A pre-loaded template ``VlaBackend`` to clone from; when None the
            process-wide shared policy is loaded/reused via
            :func:`load_shared_vla_policy`.

    Returns:
        A ``VlaBackend`` clone whose ``model`` is the shared one (``clone.model
        is shared.model``) and whose proprio window is its own.
    """
    if shared is None:
        shared = load_shared_vla_policy(ckpt, device)
    return _clone_backend_sharing_model(shared)


def attach_vla_to_nav(nav, backend) -> None:
    """Switch a settled teacher-mode ``StepwiseNav`` onto a VLA backend (F5).

    Reproduces ``StepwiseNav``'s own ``backend="vla"`` initialisation exactly:
    the WBC teacher already ran the shared settle phase during construction, and
    the trained policy now takes over. The proprio window is primed from the
    settled standing pose — skipped if the robot fell during settle (mirroring
    ``nav_core``'s ``not self.fell`` guard).

    Args:
        nav: A :class:`~code.apps.warehouse_demo.nav_core.StepwiseNav` built with
            ``backend="teacher"`` (its WBC settle has run).
        backend: The per-unit ``VlaBackend`` (see :func:`make_unit_vla_backend`).
    """
    nav.vla = backend
    nav.backend = "vla"
    if not nav.fell:
        backend.reset(nav.teacher.data, nav.teacher._target_dof)


def clear_shared_policies() -> None:
    """Drop the process-wide shared-policy cache (test isolation helper)."""
    _SHARED_POLICIES.clear()
