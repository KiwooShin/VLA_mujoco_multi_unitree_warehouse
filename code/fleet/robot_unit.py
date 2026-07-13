"""robot_unit.py — One named G1 in the fleet (federated physics).

A :class:`RobotUnit` bundles a callsign with its OWN warehouse physics
(``MjModel`` / ``MjData`` / :class:`~code.sim.teacher.WBCTeacher`, driven through
:class:`~code.apps.warehouse_demo.nav_core.StepwiseNav`). Because the robot is
alone in its model, every baseline single-robot assumption (pelvis at
``qpos[0:3]``, the walk-policy hygiene, fall detection) holds verbatim, and no
two robots ever collide by construction (docs/multi_plan.md sec 1/3).

The unit tracks a small state machine — ``idle`` (no goal) -> ``walking`` ->
``arrived`` (or ``fallen``), with ``paused`` interleaved while the fleet's
mutual-proximity rule holds it still. :meth:`step` advances exactly one 50 Hz
control step; the fleet copies :attr:`qpos` into the shared viz model afterwards.
"""

from __future__ import annotations

import enum
import math
from typing import Callable, Optional, Tuple

import numpy as np

from code.apps.warehouse_demo.nav_core import NavParams, StepInfo, StepwiseNav
from code.planner.reserve import (DEFAULT_SPEED_MPS, ReservationContext,
                                  ReservationTable)
from code.sim.teacher import WBCTeacher

Point = Tuple[float, float]


class RobotState(enum.Enum):
    """Lifecycle state of a fleet robot.

    Values:
        IDLE: Spawned/settled, no goal assigned (or a plan failed).
        WALKING: Actively following its A* path.
        PAUSED: Held in place this step by the fleet proximity rule.
        ARRIVED: Reached its goal upright (terminal).
        FALLEN: Pelvis dropped below the fall height (terminal).
    """

    IDLE = "idle"
    WALKING = "walking"
    PAUSED = "paused"
    ARRIVED = "arrived"
    FALLEN = "fallen"


def advance_state(
    state: RobotState, *, fell: bool, done: bool, paused: bool,
) -> RobotState:
    """Pure state-machine transition for one control step.

    Precedence: terminal states stick (``arrived``/``fallen`` never change), then
    a fall dominates, then an idle robot stays idle, then arrival, else the robot
    is walking (or paused this step if the fleet held it).

    Args:
        state: Current state.
        fell: True if the pelvis is below the fall height this step.
        done: True if the follower has arrived at the goal.
        paused: True if the fleet held this robot still this step.

    Returns:
        The next :class:`RobotState`.
    """
    if state == RobotState.ARRIVED:
        return RobotState.ARRIVED
    if state == RobotState.FALLEN:
        return RobotState.FALLEN
    if fell:
        return RobotState.FALLEN
    if state == RobotState.IDLE:
        return RobotState.IDLE
    if done:
        return RobotState.ARRIVED
    return RobotState.PAUSED if paused else RobotState.WALKING


class RobotUnit:
    """A single named G1 robot with its own federated physics + navigator."""

    def __init__(
        self,
        name: str,
        scene_cfg: dict,
        spawn_xy: Point,
        spawn_yaw: float,
        *,
        params: Optional[NavParams] = None,
        teacher: Optional[WBCTeacher] = None,
        use_gpu: bool = True,
        locomotion: str = "teacher",
        vla_ckpt: Optional[str] = None,
        vla_device: Optional[str] = None,
        vla_backend: Optional[object] = None,
        reservations: bool = False,
        reservation_table: Optional[ReservationTable] = None,
        reservation_speed: Optional[float] = None,
        reservation_now: Optional[Callable[[], int]] = None,
    ) -> None:
        """Build the robot's own physics, spawn it and run the settle phase.

        Args:
            name: Callsign (e.g. "Alpha").
            scene_cfg: Shared warehouse scene_cfg (walls/objects/zones).
            spawn_xy: World (x, y) spawn position (this robot's home bay).
            spawn_yaw: Spawn yaw (rad).
            params: Navigation tunables (defaults used if None).
            teacher: Optional pre-loaded :class:`WBCTeacher` to own; a fresh one
                is created if None. The WBC teacher runs the shared settle phase
                in BOTH locomotion modes (F5).
            use_gpu: Prefer CUDA for the fresh teacher's ONNX session.
            locomotion: ``"teacher"`` (default; the WBC walk policy drives every
                step — unchanged behaviour for all existing evals) or ``"vla"``
                (F5: the trained GroundedNav policy drives locomotion once the
                robot has settled).
            vla_ckpt: GroundedNav checkpoint path (``locomotion="vla"``); None
                resolves the F5 default via
                :func:`code.fleet.locomotion.resolve_vla_ckpt`.
            vla_device: Torch device for the VLA policy ('cuda'|'cpu'|None auto).
            vla_backend: A pre-built per-unit ``VlaBackend`` sharing the fleet's
                one loaded policy model (injected by :class:`~code.fleet.fleet.
                Fleet`); when None one is created from the process-wide shared
                policy, so several units still share ONE model.
            reservations: Enable proactive space-time reservation planning (F7).
                Default False -> every plan is the plain A* path (byte-identical
                to the baseline). When True (and ``reservation_table`` is given),
                each :meth:`assign_goal`/:meth:`halt` books/releases this robot's
                route in the shared table and plans around other robots' bookings.
                The proximity pause stays armed underneath in BOTH modes.
            reservation_table: Shared :class:`~code.planner.reserve.ReservationTable`
                owned by the :class:`~code.fleet.fleet.Fleet` (required for
                ``reservations=True`` to have any effect).
            reservation_speed: Conservative model walking speed (m/s) for the
                space-time cost/window model (None -> ``DEFAULT_SPEED_MPS``).
            reservation_now: Callable returning the current control-step index
                (the fleet's clock); reservations are booked from this ``t0``
                (None -> a constant 0 clock, used by unit tests).

        Raises:
            ValueError: If ``locomotion`` is not ``"teacher"``/``"vla"``.
            FileNotFoundError: If ``locomotion="vla"`` and the resolved
                checkpoint file is missing.
        """
        if locomotion not in ("teacher", "vla"):
            raise ValueError(
                f"locomotion must be 'teacher' or 'vla'; got {locomotion!r}")
        self.name = name
        self.spawn_xy: Point = (float(spawn_xy[0]), float(spawn_xy[1]))
        self.spawn_yaw = float(spawn_yaw)
        self.locomotion = locomotion
        self.goal_xy: Optional[Point] = None
        self.plan_ok: Optional[bool] = None

        # F7 space-time reservations (additive; off unless a table is supplied).
        self._resv_on: bool = bool(reservations and reservation_table is not None)
        self._resv_table: Optional[ReservationTable] = reservation_table
        self._resv_speed: float = (float(reservation_speed)
                                   if reservation_speed is not None
                                   else DEFAULT_SPEED_MPS)
        self._resv_now: Callable[[], int] = reservation_now or (lambda: 0)
        self.st_fallbacks: int = 0  # times the ST search fell back to plain A*
        self.st_replans: int = 0    # times a reserved route was (re)planned

        teacher = teacher or WBCTeacher(use_gpu=use_gpu)
        # In VLA mode the nav is built teacher-mode (its WBC settle runs), then
        # switched onto the shared trained policy — so the ~90 MB weights are
        # never reloaded per robot (the fleet shares ONE model).
        self._nav = StepwiseNav(scene_cfg, self.spawn_xy, self.spawn_yaw,
                                teacher, params)
        if locomotion == "vla":
            from code.fleet.locomotion import (attach_vla_to_nav,
                                               make_unit_vla_backend)
            backend = vla_backend or make_unit_vla_backend(vla_ckpt, vla_device)
            attach_vla_to_nav(self._nav, backend)
        self.state = RobotState.FALLEN if self._nav.fell else RobotState.IDLE

    # ---- Goal assignment ----
    def assign_goal(self, goal_xy: Point) -> bool:
        """Plan an A* path to ``goal_xy`` and start walking if reachable.

        Args:
            goal_xy: World (x, y) goal (typically an occluded object spot).

        Returns:
            True if a collision-free path was found (state -> ``walking``);
            False if unreachable or the robot has already fallen (state left
            ``idle``/``fallen``).
        """
        if self.state == RobotState.FALLEN:
            self.plan_ok = False
            return False
        if not self._resv_on:
            ok = self._nav.plan(goal_xy)
        else:
            ok = self._plan_reserved(goal_xy)
        self.plan_ok = ok
        if ok:
            self.goal_xy = (float(goal_xy[0]), float(goal_xy[1]))
            self.state = RobotState.WALKING
        else:
            self.goal_xy = None
            self.state = RobotState.IDLE
        return ok

    def _plan_reserved(self, goal_xy: Point) -> bool:
        """Plan a space-time route and (re)book it in the shared table (F7).

        Releases this robot's prior booking first (a replan supersedes it), plans
        around every other robot's booking via the ST A*, then books the chosen
        route from the current fleet clock. Returns like the plain planner.
        """
        table = self._resv_table
        assert table is not None  # guarded by self._resv_on
        table.release(self.name)
        t0 = int(self._resv_now())
        ctx = ReservationContext(table=table, t0=t0, speed=self._resv_speed,
                                 robot_id=self.name)
        ok = self._nav.plan(goal_xy, reserve=ctx)
        if ok:
            self.st_replans += 1
            if self._nav.st_fell_back:
                self.st_fallbacks += 1
            booking = self._nav.last_booking
            if booking is not None:
                cells, cell_times = booking
                table.reserve(cells, t0, cell_times, self.name)
        return ok

    def release_reservation(self) -> None:
        """Drop this robot's space-time booking (arrival/terminal cleanup, F7).

        No-op unless reservations are enabled. Idempotent — safe to call every
        step once a robot is terminal.
        """
        if self._resv_on and self._resv_table is not None:
            self._resv_table.release(self.name)

    def halt(self) -> None:
        """Abort the current goal and hold in place (search-cancel / stop).

        Clears the underlying plan and returns a walking/paused robot to
        ``idle``; terminal (arrived/fallen) robots are left untouched. Additive
        control hook for the Phase-4 search controller — the baseline nav loops
        never call it.
        """
        if self.state in (RobotState.WALKING, RobotState.PAUSED):
            self._nav.clear_goal()
            self.goal_xy = None
            self.state = RobotState.IDLE
            self.release_reservation()

    # ---- One control step ----
    def step(self, *, paused: bool = False) -> StepInfo:
        """Advance one 50 Hz control step and update the state machine.

        Args:
            paused: If True, the fleet is holding this robot still (proximity
                rule); it commands zero velocity and enters ``paused`` state.

        Returns:
            The :class:`StepInfo` from the underlying navigator.
        """
        hold = paused or self.state in (
            RobotState.ARRIVED, RobotState.FALLEN, RobotState.IDLE)
        if self.locomotion == "vla":
            self._select_hold_backend(hold)
        info = self._nav.step(hold=hold)
        self.state = advance_state(self.state, fell=info.fell, done=info.done,
                                   paused=paused)
        return info

    def _select_hold_backend(self, hold: bool) -> None:
        """Pick the backend for this step under VLA locomotion (F5).

        The trained VLA policy drives every step the robot actually WALKS, but a
        zero-velocity hold (idle after a search, arrived at goal, or a
        proximity-pause) is a *balance* task, not locomotion: the distilled walk
        policy — fed a zero command with a still-walking proprio window — slowly
        topples a robot that has just been walking (measured: pelvis 0.73 -> fall
        over ~40 steps), whereas the WBC balance controller (which already runs
        the settle phase in both modes) holds a stand indefinitely. So held steps
        run on the WBC and active walking runs on the VLA. On each hold -> walk
        resume the VLA proprio window is re-primed from the current standing pose
        so the GRU restarts from a clean, in-distribution stand rather than a
        stale window frozen across the hold. All visible room-to-room locomotion
        is therefore VLA-driven; only standing-in-place is balanced by the WBC.
        """
        nav = self._nav
        if hold:
            nav.backend = "teacher"
        elif nav.backend != "vla":
            nav.backend = "vla"
            if not nav.fell and nav.vla is not None:
                nav.vla.reset(nav.teacher.data, nav.teacher._target_dof)

    # ---- Read-only state ----
    @property
    def teacher(self) -> WBCTeacher:
        """The robot's own walk-policy teacher (owns its physics MjData)."""
        return self._nav.teacher

    @property
    def xy(self) -> Point:
        """Pelvis (x, y) world position (m)."""
        return self._nav.xy

    @property
    def yaw(self) -> float:
        """Pelvis yaw (rad)."""
        return self._nav.yaw

    @property
    def base_height(self) -> float:
        """Pelvis height (m)."""
        return self._nav.base_height

    @property
    def qpos(self) -> np.ndarray:
        """Full physics qpos (free joint + joints) for viz-model sync."""
        return self._nav.qpos

    @property
    def done(self) -> bool:
        """True once the robot has arrived at its goal."""
        return self.state == RobotState.ARRIVED

    @property
    def fell(self) -> bool:
        """True once the robot has fallen."""
        return self.state == RobotState.FALLEN

    @property
    def active(self) -> bool:
        """True while the robot is still moving toward its goal (walking/paused)."""
        return self.state in (RobotState.WALKING, RobotState.PAUSED)

    @property
    def terminal(self) -> bool:
        """True once the robot can no longer make progress (arrived/fallen)."""
        return self.state in (RobotState.ARRIVED, RobotState.FALLEN)

    @property
    def walked_length(self) -> float:
        """Arc length walked by the pelvis since settle (m)."""
        return self._nav.walked

    @property
    def planned_length(self) -> float:
        """Arc length of the smoothed A* path (m)."""
        return self._nav.planned_len

    @property
    def planned_path(self):
        """The smoothed A* waypoints (world xy)."""
        return self._nav.planned_path

    @property
    def min_wall_clearance(self) -> float:
        """Min pelvis-to-wall clearance observed since settle (m)."""
        return self._nav.min_clear

    @property
    def wall_collision(self) -> bool:
        """True if any robot geom has contacted a wall geom."""
        return self._nav.wall_collision

    @property
    def vla_infer_ms(self) -> float:
        """Mean VLA policy forward-pass time (ms); 0.0 in teacher mode / unused."""
        vla = getattr(self._nav, "vla", None)
        return float(vla.mean_infer_ms) if vla is not None else 0.0

    def distance_to_goal(self) -> float:
        """Straight-line distance from the pelvis to the goal (m); inf if none."""
        if self.goal_xy is None:
            return float("inf")
        x, y = self.xy
        return math.hypot(self.goal_xy[0] - x, self.goal_xy[1] - y)

    def status_line(self) -> str:
        """One-line human-readable status for HUD overlays."""
        d = self.distance_to_goal()
        dtxt = f"{d:4.1f}m" if math.isfinite(d) else "  -- "
        return f"{self.name:<7} {self.state.value:<7} d={dtxt}"
