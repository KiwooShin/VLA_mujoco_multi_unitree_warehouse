"""nav_rollout.py — Single-robot A*-planned WALK across the warehouse.

``run_nav_rollout`` is the load-bearing de-risk for the multi-robot project: it
drives ONE Unitree G1 (the baseline :class:`code.sim.teacher.WBCTeacher` walk
policy) along a collision-free, inflated-grid A* path to an occluded goal, using
only baseline building blocks:

* geometry / MJCF from ``code.warehouse`` (build the model, occupancy grid),
* the planner from ``code.planner`` (``inflate`` -> ``plan_path`` ->
  ``shortcut_path`` -> :class:`WaypointFollower`),
* the steering law ``code.control.steer.steer`` fed the follower's lookahead
  point each 50 Hz control step, whose ``(vx, vy, wz)`` velocity command is
  handed to the teacher EXACTLY the way ``code/datagen`` drives it.

The teacher model/data are overwritten with the warehouse-built model (the
``code/runtime/rollout_state.py`` L181-198 pattern), then a settle phase, then
the per-control-step closed loop: read pelvis xy/yaw from qpos, ask the follower
for a target, steer to a command, step the teacher. Terminates on
``follower.done`` (success), a fall (pelvis z < 0.5), or ``max_steps``.

Walk-policy hygiene (project docs): the robot turns in place first (``steer``
zeroes forward speed while ``|yaw_err|`` > 25 deg) so it never makes a large
heading change while translating out of spawn; continuous turn-in-place runs are
bounded well under the ~470-step OOD limit and guarded.
"""

from __future__ import annotations

import dataclasses
import math
import os
import time
from typing import Dict, List, Optional, Tuple

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import mujoco
import numpy as np

from code.apps.warehouse_demo.planning import (
    build_inflated_grid,
    layout_from_scene_cfg,
    min_wall_clearance,
)
from code.control.steer import steer
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
# Result
# ---------------------------------------------------------------------------
@dataclasses.dataclass
class NavResult:
    """Outcome of one ``run_nav_rollout`` episode.

    Attributes:
        success: True if the follower reached the goal (arrive radius) upright.
        outcome: 'success' | 'fall' | 'timeout' | 'plan_failed'.
        steps: Control steps executed after settle.
        path_length_planned: Arc length of the smoothed A* path (m).
        path_length_walked: Arc length actually walked by the pelvis (m).
        path_efficiency: planned / walked in (0, 1]; 0.0 if nothing walked.
        min_wall_clearance: Min pelvis-center distance to any wall footprint (m).
        fell: True if the robot fell.
        wall_collision: True if any robot geom contacted a wall geom.
        time_s: Wall-clock seconds spent in the rollout.
        goal_xy: The goal the robot navigated to.
        max_turn_run: Longest continuous turn-in-place run observed (steps).
        planned_path: The smoothed A* waypoints (world xy).
        video_path: Path to the recorded MP4, if ``record_video``.
    """

    success: bool
    outcome: str
    steps: int
    path_length_planned: float
    path_length_walked: float
    path_efficiency: float
    min_wall_clearance: float
    fell: bool
    wall_collision: bool
    time_s: float
    goal_xy: Point
    max_turn_run: int = 0
    planned_path: List[Point] = dataclasses.field(default_factory=list)
    video_path: Optional[str] = None

    def to_dict(self) -> Dict[str, object]:
        """JSON-serializable view (tuples/lists flattened to plain floats)."""
        d = dataclasses.asdict(self)
        d["goal_xy"] = [float(self.goal_xy[0]), float(self.goal_xy[1])]
        d["planned_path"] = [[float(x), float(y)] for (x, y) in self.planned_path]
        return d


# ---------------------------------------------------------------------------
# Pure control-loop helpers (unit-tested without a simulator). Grid/clearance
# geometry lives in ``planning.py``.
# ---------------------------------------------------------------------------
def classify_termination(
    done: bool, pelvis_z: float, step: int, max_steps: int,
    *, fall_height: float = FALL_HEIGHT,
) -> Tuple[bool, str]:
    """Decide whether/why the rollout loop should stop this step.

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
    """Turn the follower target into a bounded velocity command.

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


def _wall_geom_ids(model: mujoco.MjModel) -> set:
    """Return the set of geom ids whose names are warehouse walls."""
    ids = set()
    for gid in range(model.ngeom):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, gid)
        if name and name.startswith(_WALL_PREFIXES):
            ids.add(gid)
    return ids


def _inject_arena(teacher: WBCTeacher, model: mujoco.MjModel) -> None:
    """Overwrite the teacher's model/data with the warehouse model in place."""
    teacher.model = model
    teacher.data = mujoco.MjData(model)
    teacher.model.opt.timestep = SIM_DT
    teacher._nj = model.nq - 7
    teacher._pelvis_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")


def _touches_wall(data: mujoco.MjData, wall_ids: set) -> bool:
    """True if any active contact this step involves a wall geom."""
    for i in range(data.ncon):
        con = data.contact[i]
        if con.geom1 in wall_ids or con.geom2 in wall_ids:
            return True
    return False


def run_nav_rollout(
    scene_cfg: dict,
    goal_xy: Point,
    *,
    seed: int = 0,
    record_video: bool = False,
    out_dir: Optional[str] = None,
    max_steps: int = 1500,
    params: Optional[NavParams] = None,
    teacher: Optional[WBCTeacher] = None,
    video_name: Optional[str] = None,
) -> NavResult:
    """Walk one G1 along an A* path to ``goal_xy`` through the warehouse.

    Args:
        scene_cfg: Warehouse scene_cfg (from ``warehouse_scene_cfg``); supplies
            ``walls``, ``robot_xy``, ``robot_yaw``, ``objects``.
        goal_xy: World (x, y) goal (typically an occluded object spot).
        seed: Unused by the deterministic controller; recorded for provenance.
        record_video: If True, render a fixed wide-BEV MP4 with the planned-path
            overlay to ``out_dir``.
        out_dir: Directory for the MP4 (required when ``record_video``).
        max_steps: Hard cap on post-settle control steps.
        params: Optional :class:`NavParams`; defaults used otherwise.
        teacher: Optional pre-loaded :class:`WBCTeacher` to reuse across
            episodes (its model/data are overwritten here). A fresh one is
            created if None.
        video_name: Basename (no extension) for the MP4; auto-derived otherwise.

    Returns:
        A :class:`NavResult`.
    """
    params = params or NavParams()
    del seed  # deterministic controller; kept in the signature for provenance
    t0 = time.time()

    from code.warehouse.arena import build_warehouse_arena

    model = build_warehouse_arena(scene_cfg)
    wall_ids = _wall_geom_ids(model)
    walls = scene_cfg.get("walls", [])

    if teacher is None:
        teacher = WBCTeacher(use_gpu=True)
    _inject_arena(teacher, model)

    rx, ry = scene_cfg["robot_xy"]
    ryaw = float(scene_cfg.get("robot_yaw", 0.0))
    teacher.reset(pos_xy=(rx, ry), yaw=ryaw)

    # ---- Settle (zero velocity) ----
    for _ in range(params.settle_steps):
        teacher.step(vel_cmd=(0.0, 0.0, 0.0))
        if teacher.base_height < FALL_HEIGHT:
            return NavResult(
                success=False, outcome="fall", steps=0,
                path_length_planned=0.0, path_length_walked=0.0,
                path_efficiency=0.0, min_wall_clearance=float("nan"),
                fell=True, wall_collision=False, time_s=time.time() - t0,
                goal_xy=(float(goal_xy[0]), float(goal_xy[1])),
            )

    # ---- Plan (inflated grid A* -> shortcut smoothing) ----
    grid = build_inflated_grid(scene_cfg, params.grid_resolution,
                               params.inflate_radius, goal_xy=goal_xy)
    start_xy = (float(teacher.data.qpos[0]), float(teacher.data.qpos[1]))
    try:
        raw = plan_path(grid, start_xy, goal_xy, snap_radius_m=params.snap_radius)
        path = shortcut_path(grid, raw)
    except PathNotFoundError:
        return NavResult(
            success=False, outcome="plan_failed", steps=0,
            path_length_planned=0.0, path_length_walked=0.0,
            path_efficiency=0.0, min_wall_clearance=float("nan"),
            fell=False, wall_collision=False, time_s=time.time() - t0,
            goal_xy=(float(goal_xy[0]), float(goal_xy[1])),
        )
    planned_len = path_length(path)
    follower = WaypointFollower(path, arrive_radius=params.arrive_radius,
                                lookahead=params.lookahead)

    # ---- Optional BEV video setup ----
    renderer = None
    bev_cam = None
    frames: List[np.ndarray] = []
    if record_video:
        from code.apps.warehouse_demo import bev as bevmod
        from code.sim.arena_render import ArenaRenderer

        bw, bh = params.bev_size
        renderer = ArenaRenderer(model, tp_w=bw, tp_h=bh)
        hall = layout_from_scene_cfg(scene_cfg)
        bev_cam = bevmod.fit_bev_camera(
            hall.hall_x, hall.hall_y,
            width=bw, height=bh, fovy_deg=float(model.vis.global_.fovy),
        )

    # ---- Follow loop ----
    walked = 0.0
    clear_min = float("inf")
    wall_collision = False
    turn_run = 0
    max_turn_run = 0
    prev_xy = start_xy
    traj: List[Point] = [start_xy]
    outcome = "timeout"
    step = 0

    for step in range(max_steps):
        pxy = (float(teacher.data.qpos[0]), float(teacher.data.qpos[1]))
        pyaw = teacher.base_yaw
        target = follower.target(pxy)

        if follower.done or target is None:
            outcome = "success"
            break

        vel_cmd, dist, yaw_err, is_turn = shape_command(
            pxy, pyaw, target, params, turn_run=turn_run)
        turn_run = turn_run + 1 if is_turn else 0
        max_turn_run = max(max_turn_run, turn_run)

        teacher.step(vel_cmd=tuple(float(v) for v in vel_cmd))

        # Metrics from the post-step state.
        nxy = (float(teacher.data.qpos[0]), float(teacher.data.qpos[1]))
        walked += math.hypot(nxy[0] - prev_xy[0], nxy[1] - prev_xy[1])
        prev_xy = nxy
        traj.append(nxy)
        clear_min = min(clear_min, min_wall_clearance(nxy[0], nxy[1], walls))
        if _touches_wall(teacher.data, wall_ids):
            wall_collision = True

        if record_video and step % params.render_decimation == 0:
            frames.append(_bev_frame(renderer, bev_cam, teacher, path, goal_xy,
                                     traj, step, dist, params))

        terminated, term = classify_termination(
            follower.done, teacher.base_height, step, max_steps)
        if terminated:
            outcome = term
            break

    steps = step + 1
    fell = outcome == "fall"
    success = outcome == "success"
    efficiency = (planned_len / walked) if walked > 1e-6 else 0.0
    efficiency = min(efficiency, 1.0)

    video_path = None
    if record_video:
        video_path = _finalize_video(frames, out_dir, video_name, params, success)
        if renderer is not None:
            renderer.close()

    return NavResult(
        success=success, outcome=outcome, steps=steps,
        path_length_planned=planned_len, path_length_walked=walked,
        path_efficiency=efficiency,
        min_wall_clearance=(clear_min if clear_min != float("inf") else float("nan")),
        fell=fell, wall_collision=wall_collision, time_s=time.time() - t0,
        goal_xy=(float(goal_xy[0]), float(goal_xy[1])),
        max_turn_run=max_turn_run, planned_path=[tuple(p) for p in path],
        video_path=video_path,
    )


def _bev_frame(renderer, bev_cam, teacher, path, goal_xy, traj, step, dist,
               params: NavParams) -> np.ndarray:
    """Render one annotated BEV frame (planned path + trail + robot + HUD)."""
    from code.apps.warehouse_demo import bev as bevmod

    frame = bevmod.render_bev(renderer, teacher.data, bev_cam)
    bevmod.draw_path(frame, bev_cam, path, color=(0, 200, 255))
    bevmod.draw_marker(frame, bev_cam, goal_xy, color=(60, 220, 60),
                       radius=9, filled=False)
    bevmod.draw_polyline(frame, bev_cam, traj, (255, 170, 60), thickness=2,
                         z=0.02)
    pxy = (float(teacher.data.qpos[0]), float(teacher.data.qpos[1]))
    bevmod.draw_robot(frame, bev_cam, pxy, teacher.base_yaw)
    bevmod.put_hud(frame, [f"step {step}  dist {dist:4.2f}m  h {teacher.base_height:4.2f}m"])
    if params.ego_pip:
        try:
            rgb, _, _ = renderer.render_ego(teacher.data, teacher.base_yaw,
                                            render_depth=False)
            bevmod.paste_pip(frame, rgb, scale=0.6)
        except Exception:
            pass
    return frame


def _finalize_video(frames: List[np.ndarray], out_dir: Optional[str],
                    video_name: Optional[str], params: NavParams,
                    success: bool) -> Optional[str]:
    """Write accumulated BGR frames to an MP4 and return its path."""
    if not frames or out_dir is None:
        return None
    import cv2

    os.makedirs(out_dir, exist_ok=True)
    name = video_name or f"nav_{'ok' if success else 'x'}"
    path = os.path.join(out_dir, f"{name}.mp4")
    h, w = frames[0].shape[:2]
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"),
                             params.video_fps, (w, h))
    for f in frames:
        writer.write(f)
    writer.release()
    return path
