"""fleet.py — Multi-robot co-simulation with a mutual-proximity pause.

:class:`Fleet` constructs one :class:`~code.fleet.robot_unit.RobotUnit` per
callsign (each with its own federated physics), the shared
:class:`~code.fleet.viz.FleetViz` model, and steps them together. Because robots
never share a physics model they cannot collide with each other by construction;
to keep them from *walking through* one another in the shared view, the fleet
applies a purely kinematic **mutual-proximity pause**: when two still-walking
robots come within ``engage`` metres, the lower-priority one (fixed priority =
callsign order, Alpha highest) is commanded zero velocity until separation
exceeds ``release`` metres (hysteresis). No inter-robot contact physics is ever
simulated (docs/multi_plan.md sec 3/6).
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Set

import numpy as np

from code.fleet.robot_unit import NavParams, RobotUnit
from code.fleet.viz import FleetViz
from code.warehouse.arena import warehouse_scene_cfg
from code.warehouse.layout import CALLSIGNS, WarehouseLayout

Point = tuple

# Proximity-pause thresholds (metres). engage < release gives the hysteresis
# band that prevents chatter around the trigger distance.
ENGAGE_M: float = 1.0
RELEASE_M: float = 1.2


def compute_pauses(
    positions: Dict[str, Sequence[float]],
    active: Dict[str, bool],
    priorities: Dict[str, int],
    currently_paused: Set[str],
    *,
    engage: float = ENGAGE_M,
    release: float = RELEASE_M,
) -> Set[str]:
    """Return the set of robots that must hold still this step.

    A still-walking robot ``j`` yields (is paused) if any *higher-priority*
    still-walking robot ``k`` (``priorities[k] < priorities[j]``) is within the
    active threshold: ``release`` when ``j`` is already paused (harder to leave
    the paused state), else ``engage``. Only the lower-priority robot yields, so
    the higher-priority robot keeps moving and the pair always separates — the
    total priority order rules out mutual deadlock.

    Args:
        positions: callsign -> (x, y) world position.
        active: callsign -> True if still walking/paused (eligible to pause or
            to block others). Arrived/fallen/idle robots are inactive.
        priorities: callsign -> priority rank (lower = higher priority).
        currently_paused: Robots paused on the previous step (for hysteresis).
        engage: Distance at which a pause starts (m).
        release: Distance a paused robot must exceed to resume (m); >= engage.

    Returns:
        The set of robots to hold still this step.
    """
    new_paused: Set[str] = set()
    names = list(positions)
    for j in names:
        if not active.get(j, False):
            continue
        thr = release if j in currently_paused else engage
        for k in names:
            if k == j or not active.get(k, False):
                continue
            if priorities[k] < priorities[j]:
                dx = positions[k][0] - positions[j][0]
                dy = positions[k][1] - positions[j][1]
                if math.hypot(dx, dy) < thr:
                    new_paused.add(j)
                    break
    return new_paused


class Fleet:
    """A fleet of named G1 robots co-simulated in one warehouse."""

    def __init__(
        self,
        layout: WarehouseLayout,
        goals: Dict[str, Point],
        *,
        callsigns: Sequence[str] = CALLSIGNS,
        params: Optional[NavParams] = None,
        use_gpu: bool = True,
        engage: float = ENGAGE_M,
        release: float = RELEASE_M,
        build_viz: bool = True,
        seed: int = 0,
        teachers: Optional[Dict[str, "object"]] = None,
        objects: Optional[List[dict]] = None,
        locomotion: str = "teacher",
        vla_ckpt: Optional[str] = None,
        vla_device: Optional[str] = None,
    ) -> None:
        """Build every robot, assign goals and (optionally) the shared viz model.

        Args:
            layout: The warehouse layout (spawns + geometry).
            goals: callsign -> (x, y) goal. Robots without a goal stay idle.
            callsigns: Robots to instantiate (order fixes pause priority).
            params: Navigation tunables shared by every robot.
            use_gpu: Prefer CUDA for each robot's ONNX walk policy.
            engage: Proximity-pause engage distance (m).
            release: Proximity-pause release distance (m).
            build_viz: If True, build the shared viz model (BEV/cross-visibility).
            seed: RNG seed for the (shared) object placement.
            teachers: Optional callsign -> :class:`WBCTeacher` to reuse across
                fleets/trials (rebound onto each robot's fresh warehouse model),
                sparing a per-trial ONNX reload. A fresh teacher is built for any
                callsign missing here.
            objects: Explicit scene object list; when None one object per layout
                spot is sampled from the seeded palettes (Phase-4 missions pass a
                scenario-specific placement here).
            locomotion: ``"teacher"`` (default; every robot's WBC walk policy
                drives locomotion — unchanged behaviour) or ``"vla"`` (F5: the
                trained GroundedNav policy). All four robots SHARE one loaded
                policy model (loaded once via
                :func:`code.fleet.locomotion.load_shared_vla_policy`), each with
                its own per-robot proprio window.
            vla_ckpt: GroundedNav checkpoint (``locomotion="vla"``); None resolves
                the F5 default.
            vla_device: Torch device for the shared VLA policy (None -> auto).
        """
        self.callsigns: List[str] = list(callsigns)
        self.priorities: Dict[str, int] = {c: i for i, c in enumerate(self.callsigns)}
        self.engage = float(engage)
        self.release = float(release)
        self.locomotion = locomotion

        scene_cfg = warehouse_scene_cfg(
            layout, robot=self.callsigns[0], objects=objects,
            rng=np.random.default_rng(seed))
        self.scene_cfg = scene_cfg

        # F5: load the ONE shared policy up front (fail-fast on a bad checkpoint,
        # loaded exactly once for the whole fleet) and hand each robot a clone
        # that references it — one model on the GPU, per-robot proprio windows.
        shared_vla = None
        if locomotion == "vla":
            from code.fleet.locomotion import (load_shared_vla_policy,
                                               make_unit_vla_backend,
                                               resolve_vla_ckpt)
            vla_ckpt = resolve_vla_ckpt(vla_ckpt)
            shared_vla = load_shared_vla_policy(vla_ckpt, vla_device)
        self.vla_ckpt = vla_ckpt

        teachers = teachers or {}
        self.units: Dict[str, RobotUnit] = {}
        for name in self.callsigns:
            sx, sy, syaw = layout.spawn_poses[name]
            unit_backend = (make_unit_vla_backend(shared=shared_vla)
                            if shared_vla is not None else None)
            unit = RobotUnit(name, scene_cfg, (sx, sy), syaw,
                             params=params, teacher=teachers.get(name),
                             use_gpu=use_gpu, locomotion=locomotion,
                             vla_ckpt=vla_ckpt, vla_device=vla_device,
                             vla_backend=unit_backend)
            if name in goals:
                unit.assign_goal(goals[name])
            self.units[name] = unit

        self.viz: Optional[FleetViz] = (
            FleetViz(scene_cfg, self.callsigns) if build_viz else None)

        self.step_count = 0
        self.pause_events = 0
        self._paused: Set[str] = set()
        self.arrive_step: Dict[str, int] = {}
        if self.viz is not None:
            self.viz.sync(self._poses())

    # ---- Internal ----
    def _poses(self) -> Dict[str, np.ndarray]:
        """Current physics qpos of every robot (for viz sync)."""
        return {name: u.qpos for name, u in self.units.items()}

    # ---- Stepping ----
    def step_all(self) -> Set[str]:
        """Advance every robot one control step, applying the proximity pause.

        Returns:
            The set of robots paused this step.
        """
        positions = {name: u.xy for name, u in self.units.items()}
        active = {name: u.active for name, u in self.units.items()}
        paused = compute_pauses(positions, active, self.priorities, self._paused,
                                engage=self.engage, release=self.release)
        self.pause_events += len(paused - self._paused)
        self._paused = paused

        for name, unit in self.units.items():
            unit.step(paused=(name in paused))

        self.step_count += 1
        for name, unit in self.units.items():
            if unit.done and name not in self.arrive_step:
                self.arrive_step[name] = self.step_count

        if self.viz is not None:
            self.viz.sync(self._poses())
        return paused

    def run(self, max_steps: int, *, on_step=None) -> int:
        """Step the fleet until every robot is terminal or ``max_steps`` hit.

        Args:
            max_steps: Hard cap on control steps.
            on_step: Optional callable ``on_step(fleet, step_index)`` invoked
                after each step (e.g. to capture a video frame).

        Returns:
            The number of control steps executed.
        """
        for i in range(max_steps):
            self.step_all()
            if on_step is not None:
                on_step(self, i)
            if self.all_terminal:
                break
        return self.step_count

    # ---- Status ----
    @property
    def all_terminal(self) -> bool:
        """True once no robot is still walking/paused."""
        return all(u.terminal for u in self.units.values())

    @property
    def all_arrived(self) -> bool:
        """True if every robot reached its goal upright."""
        return all(u.done for u in self.units.values())

    @property
    def any_fell(self) -> bool:
        """True if any robot has fallen."""
        return any(u.fell for u in self.units.values())

    @property
    def makespan(self) -> Optional[int]:
        """Step index of the last arrival, or None if not all arrived."""
        if not self.all_arrived:
            return None
        return max(self.arrive_step.values()) if self.arrive_step else 0

    def statuses(self) -> List[str]:
        """Per-robot status lines (fixed callsign order)."""
        return [self.units[n].status_line() for n in self.callsigns]

    def mean_vla_infer_ms(self) -> float:
        """Mean VLA policy forward-pass time (ms) across robots (0.0 in teacher mode).

        The step cost of the trained locomotion backend: averaged over every
        robot that actually ran a forward pass this run. 0.0 when
        ``locomotion="teacher"`` (no VLA policy) or before any stepping.
        """
        vals = [u.vla_infer_ms for u in self.units.values() if u.vla_infer_ms > 0.0]
        return float(sum(vals) / len(vals)) if vals else 0.0

    def close(self) -> None:
        """Release the viz renderers."""
        if self.viz is not None:
            self.viz.close()
