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
from typing import Optional, Tuple

import numpy as np

from code.apps.warehouse_demo.nav_core import NavParams, StepInfo, StepwiseNav
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
    ) -> None:
        """Build the robot's own physics, spawn it and run the settle phase.

        Args:
            name: Callsign (e.g. "Alpha").
            scene_cfg: Shared warehouse scene_cfg (walls/objects/zones).
            spawn_xy: World (x, y) spawn position (this robot's home bay).
            spawn_yaw: Spawn yaw (rad).
            params: Navigation tunables (defaults used if None).
            teacher: Optional pre-loaded :class:`WBCTeacher` to own; a fresh one
                is created if None.
            use_gpu: Prefer CUDA for the fresh teacher's ONNX session.
        """
        self.name = name
        self.spawn_xy: Point = (float(spawn_xy[0]), float(spawn_xy[1]))
        self.spawn_yaw = float(spawn_yaw)
        self.goal_xy: Optional[Point] = None
        self.plan_ok: Optional[bool] = None

        teacher = teacher or WBCTeacher(use_gpu=use_gpu)
        self._nav = StepwiseNav(scene_cfg, self.spawn_xy, self.spawn_yaw,
                                teacher, params)
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
        ok = self._nav.plan(goal_xy)
        self.plan_ok = ok
        if ok:
            self.goal_xy = (float(goal_xy[0]), float(goal_xy[1]))
            self.state = RobotState.WALKING
        else:
            self.goal_xy = None
            self.state = RobotState.IDLE
        return ok

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
        info = self._nav.step(hold=hold)
        self.state = advance_state(self.state, fell=info.fell, done=info.done,
                                   paused=paused)
        return info

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
