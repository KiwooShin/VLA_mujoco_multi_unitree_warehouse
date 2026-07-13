"""nav_rollout.py — Single-robot A*-planned WALK across the warehouse.

``run_nav_rollout`` is the load-bearing de-risk for the multi-robot project: it
drives ONE Unitree G1 (the baseline :class:`code.sim.teacher.WBCTeacher` walk
policy) along a collision-free, inflated-grid A* path to an occluded goal.

The stepwise navigation engine itself now lives in
:mod:`code.apps.warehouse_demo.nav_core` (:class:`~code.apps.warehouse_demo.nav_core.StepwiseNav`
plus the shared tunables/helpers), so the single-robot loop below and the
multi-robot fleet (``code.fleet``) drive the *same* settle phase, turn-in-place
guard and fall detection — never a divergent copy (docs/multi_plan.md sec 3).
This module keeps the demo/eval concerns: the :class:`NavResult` record, the
BEV video recording, and the fixed max-steps run loop. Its public API
(``NavParams``, ``NavResult``, ``classify_termination``, ``shape_command``,
``run_nav_rollout``) is unchanged and re-exported for existing callers/tests.

Walk-policy hygiene (project docs): the robot turns in place first (``steer``
zeroes forward speed while ``|yaw_err|`` > 25 deg) so it never makes a large
heading change while translating out of spawn; continuous turn-in-place runs are
bounded well under the ~470-step OOD limit and guarded.
"""

from __future__ import annotations

import dataclasses
import os
import time
from typing import Dict, List, Optional, Tuple

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import numpy as np

from code.apps.warehouse_demo.nav_core import (  # noqa: F401  (re-exported API)
    FALL_HEIGHT,
    NavParams,
    StepInfo,
    StepwiseNav,
    classify_termination,
    inject_arena,
    shape_command,
    touches_wall,
    wall_geom_ids,
)
from code.apps.warehouse_demo.planning import layout_from_scene_cfg
from code.sim.teacher import WBCTeacher

Point = Tuple[float, float]


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


def _fail_result(outcome: str, goal_xy: Point, t0: float, *, fell: bool) -> NavResult:
    """Build a zero-progress NavResult for a settle-fall or plan failure."""
    return NavResult(
        success=False, outcome=outcome, steps=0,
        path_length_planned=0.0, path_length_walked=0.0,
        path_efficiency=0.0, min_wall_clearance=float("nan"),
        fell=fell, wall_collision=False, time_s=time.time() - t0,
        goal_xy=(float(goal_xy[0]), float(goal_xy[1])),
    )


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

    if teacher is None:
        teacher = WBCTeacher(use_gpu=True)

    rx, ry = scene_cfg["robot_xy"]
    ryaw = float(scene_cfg.get("robot_yaw", 0.0))
    nav = StepwiseNav(scene_cfg, (rx, ry), ryaw, teacher, params)
    if nav.fell:
        return _fail_result("fall", goal_xy, t0, fell=True)

    if not nav.plan(goal_xy):
        return _fail_result("plan_failed", goal_xy, t0, fell=False)
    path = nav.planned_path

    # ---- Optional BEV video setup ----
    renderer = None
    bev_cam = None
    frames: List[np.ndarray] = []
    if record_video:
        from code.apps.warehouse_demo import bev as bevmod
        from code.sim.arena_render import ArenaRenderer

        bw, bh = params.bev_size
        renderer = ArenaRenderer(nav.model, tp_w=bw, tp_h=bh)
        hall = layout_from_scene_cfg(scene_cfg)
        bev_cam = bevmod.fit_bev_camera(
            hall.hall_x, hall.hall_y,
            width=bw, height=bh, fovy_deg=float(nav.model.vis.global_.fovy),
        )

    # ---- Follow loop (StepwiseNav owns the per-step control law) ----
    traj: List[Point] = [nav.xy]
    outcome = "timeout"
    step = 0
    for step in range(max_steps):
        info = nav.step()
        traj.append(nav.xy)

        if record_video and step % params.render_decimation == 0:
            frames.append(_bev_frame(renderer, bev_cam, teacher, path, goal_xy,
                                     traj, step, info.dist, params))

        terminated, term = classify_termination(
            info.done, nav.base_height, step, max_steps)
        if terminated:
            outcome = term
            break

    steps = step + 1
    fell = outcome == "fall"
    success = outcome == "success"
    efficiency = (nav.planned_len / nav.walked) if nav.walked > 1e-6 else 0.0
    efficiency = min(efficiency, 1.0)

    video_path = None
    if record_video:
        video_path = _finalize_video(frames, out_dir, video_name, params, success)
        if renderer is not None:
            renderer.close()

    return NavResult(
        success=success, outcome=outcome, steps=steps,
        path_length_planned=nav.planned_len, path_length_walked=nav.walked,
        path_efficiency=efficiency,
        min_wall_clearance=(nav.min_clear if nav.min_clear != float("inf")
                            else float("nan")),
        fell=fell, wall_collision=nav.wall_collision, time_s=time.time() - t0,
        goal_xy=(float(goal_xy[0]), float(goal_xy[1])),
        max_turn_run=nav.max_turn_run, planned_path=[tuple(p) for p in path],
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
