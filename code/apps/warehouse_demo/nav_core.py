"""nav_core.py — Reusable stepwise A* walk engine shared by Phase-1c and Phase-2.

This module is the extracted, simulator-coupled core of ``nav_rollout.py``: the
navigation tunables (:class:`NavParams`), the pure control-loop helpers
(:func:`classify_termination`, :func:`shape_command`), the teacher/model
plumbing (:func:`inject_arena`, :func:`wall_geom_ids`, :func:`touches_wall`) and
:class:`StepwiseNav` — a single-robot navigator that exposes ONE 50 Hz control
step at a time.

``nav_rollout.run_nav_rollout`` (the single-robot demo/eval loop with video +
:class:`NavResult`) and ``code.fleet`` (the multi-robot co-simulation) both drive
navigation through the identical primitives here, so the walk-policy hygiene
(settle phase, turn-in-place guard, fall detection) can never skew between the
single- and multi-robot code paths (docs/multi_plan.md sec 3).

The federated-physics contract (docs/multi_plan.md sec 1): each robot owns its
OWN warehouse ``MjModel`` / ``MjData`` / :class:`~code.sim.teacher.WBCTeacher`
(the robot is alone in its model, so every baseline qpos-slicing assumption stays
valid verbatim). :class:`StepwiseNav` never touches any other robot.
"""

from __future__ import annotations

import dataclasses
import math
import os
from typing import List, Optional, Set, Tuple

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import mujoco
import numpy as np

from code.apps.warehouse_demo.planning import build_inflated_grid, min_wall_clearance
from code.control.steer import goal_vec as steer_goal_vec, steer
from code.planner.astar import PathNotFoundError, path_length, plan_path, shortcut_path
from code.planner.follower import WaypointFollower
from code.sim.teacher import SIM_DT, WBCTeacher

Point = Tuple[float, float]

FALL_HEIGHT: float = 0.50  # pelvis z below this == fallen (baseline convention)
_WALL_PREFIXES: Tuple[str, ...] = ("wall_", "shelf_", "part_")


# ---------------------------------------------------------------------------
# Tunables (all steering/follower knobs live HERE — planner/steer defaults are
# never mutated; docs require tuning to stay in this package).
# ---------------------------------------------------------------------------
@dataclasses.dataclass(frozen=True)
class NavParams:
    """Navigation tunables.

    Rationale for ``inflate_radius`` (0.40 m): the G1 stance/shoulder footprint
    is ~0.30-0.35 m half-width; +0.05-0.10 m absorbs lateral gait sway and the
    commanded-vs-realized yaw gap. The occupancy grid rasterizes wall *centers*,
    so a 0.40 m dilation keeps the pelvis center >= ~0.40 m off every wall
    footprint while still leaving the >= 2.2 m aisles and the 1.4 m mid-row gap
    comfortably traversable (verified: 0/32 bay->spot pairs unreachable).
    """

    inflate_radius: float = 0.40
    grid_resolution: float = 0.10
    snap_radius: float = 0.60
    arrive_radius: float = 0.45
    lookahead: float = 0.85
    stop_r: float = 0.30
    max_vx: float = 0.60
    max_wz: float = 0.80
    settle_steps: int = 80
    max_turn_run: int = 440          # guard below the ~470-step OOD spin limit
    turn_break_vx: float = 0.15      # small creep injected if the guard trips
    bev_size: Tuple[int, int] = (640, 480)  # <= model offscreen buffer
    render_decimation: int = 2
    video_fps: int = 30
    ego_pip: bool = True


# ---------------------------------------------------------------------------
# Per-step info returned by StepwiseNav.step (multi-robot orchestration reads it)
# ---------------------------------------------------------------------------
@dataclasses.dataclass
class StepInfo:
    """Outcome of one :meth:`StepwiseNav.step`.

    Attributes:
        done: True once the follower has arrived within its arrive radius.
        fell: True if the pelvis is below :data:`FALL_HEIGHT` this step.
        dist: Distance to the current steering target (m); 0.0 when held.
        yaw_err: Signed bearing error to the target (rad); 0.0 when held.
        is_turn: True if this step was a turn-in-place (no forward speed).
        target_xy: The pure-pursuit steering target this step, or None.
    """

    done: bool
    fell: bool
    dist: float
    yaw_err: float
    is_turn: bool
    target_xy: Optional[Point]


# ---------------------------------------------------------------------------
# Pure control-loop helpers (unit-tested without a simulator).
# ---------------------------------------------------------------------------
def classify_termination(
    done: bool, pelvis_z: float, step: int, max_steps: int,
    *, fall_height: float = FALL_HEIGHT,
) -> Tuple[bool, str]:
    """Decide whether/why a rollout loop should stop this step.

    Precedence: fall (unrecoverable) > success (follower arrived) > timeout.

    Args:
        done: ``WaypointFollower.done`` this step.
        pelvis_z: Pelvis height (m).
        step: 0-based control step index just executed.
        max_steps: Hard cap on control steps.
        fall_height: Pelvis-z fall threshold (m).

    Returns:
        (terminated, outcome) where outcome is ''|'fall'|'success'|'timeout'.
    """
    if pelvis_z < fall_height:
        return True, "fall"
    if done:
        return True, "success"
    if step + 1 >= max_steps:
        return True, "timeout"
    return False, ""


def shape_command(
    robot_xy: Point, robot_yaw: float, target_xy: Point, params: NavParams,
    *, turn_run: int,
) -> Tuple[np.ndarray, float, float, bool]:
    """Turn a follower target into a bounded velocity command.

    Wraps ``code.control.steer.steer`` (planner/steer defaults untouched) with
    the OOD turn-in-place guard: if the robot has been turning in place for more
    than ``params.max_turn_run`` steps, inject a small forward creep so the walk
    policy does not enter the >~470-step continuous-rotation OOD regime.

    Args:
        robot_xy: Pelvis (x, y) world position (m).
        robot_yaw: Pelvis yaw (rad).
        target_xy: Steering target (m).
        params: Tunables.
        turn_run: Consecutive turn-in-place steps taken so far.

    Returns:
        (vel_cmd, dist, yaw_err, is_turn_in_place).
    """
    vel_cmd, dist, yaw_err = steer(
        robot_xy, robot_yaw, target_xy, params.stop_r,
        max_vx=params.max_vx, max_wz=params.max_wz,
    )
    is_turn = bool(vel_cmd[0] <= 1e-6 and abs(vel_cmd[2]) > 1e-3)
    if is_turn and turn_run >= params.max_turn_run:
        vel_cmd = np.array([params.turn_break_vx, 0.0, vel_cmd[2]], dtype=np.float32)
    return vel_cmd, dist, yaw_err, is_turn


def wall_geom_ids(model: mujoco.MjModel) -> Set[int]:
    """Return the set of geom ids whose names are warehouse walls."""
    ids: Set[int] = set()
    for gid in range(model.ngeom):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, gid)
        if name and name.startswith(_WALL_PREFIXES):
            ids.add(gid)
    return ids


def inject_arena(teacher: WBCTeacher, model: mujoco.MjModel) -> None:
    """Overwrite the teacher's model/data with the warehouse model in place.

    Mirrors ``code/runtime/rollout_state.py``'s model-swap pattern: the teacher
    keeps its loaded ONNX policy but drives the warehouse-built model so all
    baseline qpos slicing (pelvis at ``qpos[0:3]``, joints at ``qpos[7:22]``)
    stays valid — the robot is alone in ``model``.

    Args:
        teacher: The :class:`WBCTeacher` to rebind (mutated in place).
        model: The warehouse ``MjModel`` (exactly one G1 robot).
    """
    teacher.model = model
    teacher.data = mujoco.MjData(model)
    teacher.model.opt.timestep = SIM_DT
    teacher._nj = model.nq - 7
    teacher._pelvis_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")


def touches_wall(data: mujoco.MjData, wall_ids: Set[int]) -> bool:
    """True if any active contact this step involves a wall geom."""
    for i in range(data.ncon):
        con = data.contact[i]
        if con.geom1 in wall_ids or con.geom2 in wall_ids:
            return True
    return False


# ---------------------------------------------------------------------------
# StepwiseNav — the extracted single-robot stepwise navigator.
# ---------------------------------------------------------------------------
class StepwiseNav:
    """Drives one G1 along an A* path, one 50 Hz control step per call.

    Construction builds the robot's own warehouse ``MjModel``, rebinds the
    supplied teacher onto it, resets the robot into its spawn bay and runs the
    zero-velocity settle phase (so the pelvis drops onto the floor before
    planning). :meth:`plan` fits an inflated-grid A* path to a goal; :meth:`step`
    then advances the closed loop one control step and reports a :class:`StepInfo`.

    All walk-policy hygiene lives here: turn-in-place first (``steer`` zeroes
    forward speed while ``|yaw_err|`` > 25 deg), a bounded turn-run guard, and
    per-step fall detection. The navigator NEVER references any other robot.
    """

    def __init__(
        self,
        scene_cfg: dict,
        spawn_xy: Point,
        spawn_yaw: float,
        teacher: WBCTeacher,
        params: Optional[NavParams] = None,
        *,
        backend: str = "teacher",
        vla_ckpt: Optional[str] = None,
        vla_device: Optional[str] = None,
    ) -> None:
        """Build the robot's model, rebind the teacher, spawn and settle.

        Args:
            scene_cfg: Warehouse scene_cfg (walls/objects) shared by the fleet.
            spawn_xy: World (x, y) spawn position (this robot's home bay).
            spawn_yaw: Spawn yaw (rad).
            teacher: A :class:`WBCTeacher` owned by this navigator; rebound onto
                the warehouse model here.
            params: Navigation tunables (defaults used if None).
            backend: Locomotion backend — ``"teacher"`` (default; drives the
                WBC walk policy, unchanged behaviour) or ``"vla"`` (F5: drives
                the trained GroundedNav policy, loaded from ``vla_ckpt``). The
                WBC teacher still runs the settle phase in BOTH modes (the walk
                policy takes over only after the robot is standing) — matching
                the warehouse DART datagen recipe.
            vla_ckpt: Path to the GroundedNav checkpoint (required when
                ``backend="vla"``).
            vla_device: Torch device for the VLA policy ('cuda'|'cpu'|None →
                auto). Ignored for the teacher backend.

        Raises:
            ValueError: If ``backend`` is not ``"teacher"``/``"vla"``, or if
                ``backend="vla"`` without a ``vla_ckpt``.
            FileNotFoundError: If ``backend="vla"`` and ``vla_ckpt`` is missing.
        """
        from code.warehouse.arena import build_warehouse_arena

        if backend not in ("teacher", "vla"):
            raise ValueError(f"backend must be 'teacher' or 'vla'; got {backend!r}")
        self.backend = backend

        # ---- VLA backend: load the trained policy FIRST, so a bad backend
        # string / missing-ckpt path fails fast, before any arena is compiled or
        # physics is stepped (also makes the validation cheaply unit-testable) ----
        self.vla = None
        if backend == "vla":
            if not vla_ckpt:
                raise ValueError("backend='vla' requires vla_ckpt (checkpoint path)")
            from code.apps.warehouse_demo.vla_backend import VlaBackend
            self.vla = VlaBackend(vla_ckpt, device=vla_device)

        self.params = params or NavParams()
        self._scene_cfg = scene_cfg
        self.teacher = teacher
        self.model = build_warehouse_arena(scene_cfg)
        inject_arena(teacher, self.model)
        self._wall_ids = wall_geom_ids(self.model)
        self._walls = scene_cfg.get("walls", [])

        teacher.reset(pos_xy=(float(spawn_xy[0]), float(spawn_xy[1])),
                      yaw=float(spawn_yaw))

        self.fell: bool = False
        for _ in range(self.params.settle_steps):
            teacher.step(vel_cmd=(0.0, 0.0, 0.0))
            if teacher.base_height < FALL_HEIGHT:
                self.fell = True
                break

        # ---- Prime the VLA proprio window from the settled standing state ----
        if self.vla is not None and not self.fell:
            self.vla.reset(teacher.data, teacher._target_dof)

        self.follower: Optional[WaypointFollower] = None
        self.goal_xy: Optional[Point] = None
        self.planned_path: List[Point] = []
        self.planned_len: float = 0.0
        # Space-time reservation bookkeeping (only touched when plan() is given a
        # ReservationContext; None/False keeps the default nav path byte-identical).
        self.last_booking: Optional[Tuple[List[Tuple[int, int]], List[int]]] = None
        self.st_fell_back: bool = False
        self.walked: float = 0.0
        self.min_clear: float = float("inf")
        self.wall_collision: bool = False
        self.turn_run: int = 0
        self.max_turn_run: int = 0
        self._prev_xy: Point = self.xy

    # ---- Read-only pose accessors ----
    @property
    def xy(self) -> Point:
        """Pelvis (x, y) world position (m)."""
        return (float(self.teacher.data.qpos[0]), float(self.teacher.data.qpos[1]))

    @property
    def yaw(self) -> float:
        """Pelvis yaw (rad)."""
        return self.teacher.base_yaw

    @property
    def base_height(self) -> float:
        """Pelvis height (m)."""
        return self.teacher.base_height

    @property
    def qpos(self) -> np.ndarray:
        """The robot's full physics qpos (free joint + all hinge joints)."""
        return self.teacher.data.qpos

    @property
    def done(self) -> bool:
        """True once the follower has reached the goal."""
        return bool(self.follower is not None and self.follower.done)

    # ---- Planning ----
    def plan(self, goal_xy: Point, *, reserve: Optional["object"] = None) -> bool:
        """Fit an inflated-grid A* path from the current pose to ``goal_xy``.

        Args:
            goal_xy: World (x, y) goal (typically an occluded object spot).
            reserve: Optional
                :class:`~code.planner.reserve.ReservationContext`. When None
                (default) the plain A* path is planned exactly as before — this
                keeps the whole nav path byte-identical for every existing caller.
                When provided, a space-time A* route that dodges other robots'
                bookings is planned instead, and :attr:`last_booking` /
                :attr:`st_fell_back` are set so the fleet can book the route and
                log ST-search fallbacks. The proximity pause stays the safety net.

        Returns:
            True if a collision-free path was found (a follower is now armed);
            False if the goal is unreachable at the deployed inflation radius.
        """
        grid = build_inflated_grid(self._scene_cfg, self.params.grid_resolution,
                                   self.params.inflate_radius, goal_xy=goal_xy)
        start_xy = self.xy
        self.last_booking = None
        self.st_fell_back = False
        try:
            if reserve is None:
                raw = plan_path(grid, start_xy, goal_xy, snap_radius_m=self.params.snap_radius)
            else:
                from code.planner.reserve import plan_path_st
                st = plan_path_st(
                    grid, reserve.table, start_xy, goal_xy, reserve.t0, reserve.speed,
                    ignore_id=reserve.robot_id, snap_radius_m=self.params.snap_radius)
                raw = st.path
                self.last_booking = (st.cells, st.cell_times)
                self.st_fell_back = st.fell_back
            path = shortcut_path(grid, raw)
        except PathNotFoundError:
            return False
        self.planned_path = [tuple(p) for p in path]
        self.planned_len = path_length(path)
        self.follower = WaypointFollower(path, arrive_radius=self.params.arrive_radius,
                                         lookahead=self.params.lookahead)
        self.goal_xy = (float(goal_xy[0]), float(goal_xy[1]))
        return True

    def clear_goal(self) -> None:
        """Drop the current plan so the robot holds in place.

        Used to abort a delegated search mid-patrol (:meth:`RobotUnit.halt`): the
        follower is discarded so subsequent :meth:`step` calls command zero
        velocity until a new goal is planned.
        """
        self.follower = None
        self.goal_xy = None

    # ---- One control step ----
    def step(self, *, hold: bool = False) -> StepInfo:
        """Advance one 50 Hz control step.

        Args:
            hold: If True (or no plan / already arrived), command zero velocity
                so the robot balances in place. The fleet sets ``hold=True`` to
                realize the mutual-proximity pause (zero-velocity command, no
                inter-robot contact physics).

        Returns:
            A :class:`StepInfo` describing this step.
        """
        pxy = self.xy
        pyaw = self.yaw

        target: Optional[Point] = None
        dist = 0.0
        yaw_err = 0.0
        is_turn = False

        if self.follower is None or hold:
            self._execute((0.0, 0.0, 0.0), 0.0, 0.0)
        else:
            target = self.follower.target(pxy)
            if self.follower.done or target is None:
                self._execute((0.0, 0.0, 0.0), 0.0, 0.0)
            else:
                vel_cmd, dist, yaw_err, is_turn = shape_command(
                    pxy, pyaw, target, self.params, turn_run=self.turn_run)
                self.turn_run = self.turn_run + 1 if is_turn else 0
                self.max_turn_run = max(self.max_turn_run, self.turn_run)
                self._execute(vel_cmd, dist, yaw_err)

        if not is_turn:
            self.turn_run = 0
        self._accumulate_metrics()
        fell = self.base_height < FALL_HEIGHT
        self.fell = self.fell or fell
        return StepInfo(done=self.done, fell=fell, dist=dist, yaw_err=yaw_err,
                        is_turn=is_turn, target_xy=target)

    def _execute(self, vel_cmd, dist: float, yaw_err: float) -> None:
        """Advance one control step under the active locomotion backend.

        Both backends consume the SAME steer command produced by
        :func:`shape_command`. The teacher backend hands ``vel_cmd`` to the WBC
        walk policy; the VLA backend injects the egocentric goal
        (``goal_vec(dist, yaw_err)``, exactly as the warehouse DART datagen
        logged it) and ``vel_cmd`` into the trained policy and drives the same
        student PD path. A zero command ``(0, 0, 0)`` is the in-distribution
        standing command in both modes (settle/arrival hold).

        Args:
            vel_cmd: (3,) steer velocity command ``[vx, vy, ωz]``.
            dist: Distance to the steering target (m); 0.0 when holding.
            yaw_err: Signed bearing error to the target (rad); 0.0 when holding.
        """
        if self.backend == "teacher":
            self.teacher.step(vel_cmd=tuple(float(v) for v in vel_cmd))
        else:
            gv = steer_goal_vec(dist, yaw_err)
            self.vla.step(self.model, self.teacher.data, self.teacher._nj,
                          gv, np.asarray(vel_cmd, dtype=np.float32))

    def _accumulate_metrics(self) -> None:
        """Update walked length, min wall clearance and wall-contact flag."""
        nxy = self.xy
        self.walked += math.hypot(nxy[0] - self._prev_xy[0], nxy[1] - self._prev_xy[1])
        self._prev_xy = nxy
        self.min_clear = min(self.min_clear, min_wall_clearance(nxy[0], nxy[1], self._walls))
        if touches_wall(self.teacher.data, self._wall_ids):
            self.wall_collision = True
