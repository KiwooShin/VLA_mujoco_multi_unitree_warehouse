"""rollout.py — Single warehouse DART episode (teacher-driven + ego rendering).

Faithfully replicates the baseline DART recipe
(``code/datagen/gen_dart_rollout.py::run_dart_episode``): on every 50 Hz control
step the teacher computes a CLEAN joint-target action which is stored as the
supervision label, while a NOISY action (clean + N(0, sigma) on the 15 joint
targets) is actually executed via PD for ``CONTROL_DECIMATION`` substeps to
diversify the visited state distribution (Laskey'17 DART). Proprio is built from
the noisy-executed physics state; the gait phase [sin, cos] is tracked and stored.

Differences from the baseline recipe (all additive, schema-preserving):
  * The arena is the WAREHOUSE model (``build_warehouse_arena``): light-gray tiled
    floor, brown shelves, colored pads — the F5 visual domain.
  * The steering target comes from an A*-planned route (``WaypointFollower`` +
    ``shape_command``) or a single line-of-sight target (primitive mode), instead
    of the baseline's fixed object target — yielding the realistic warehouse
    command distribution while keeping the exact same per-frame schema.
  * Ego RGB is rendered on the GPU each step and returned (so the dataset can
    also drive a vision-enabled fine-tune / the F5 visual-domain closing). The
    parquet row schema is byte-identical to the baseline DART schema.

The recorded parquet row is exactly the baseline DART row:
  frame_index, episode_index, index, task_index, timestamp, proprio(55),
  action(15, CLEAN label), goal(3), vel_cmd(3), done, task_description, phase(2).
"""

from __future__ import annotations

import time
from typing import Optional

import mujoco
import numpy as np

from code.warehouse.arena import build_warehouse_arena
from code.arena import ArenaRenderer
from code.control.steer import goal_vec
from code.datagen.gen_dart_phase import GaitPhaseTracker, build_proprio
from code.datagen.gen_dart_rollout import (
    FALL_HEIGHT, FPS, HOLD_STEPS, PROPRIO_DIM, SETTLE_STEPS,
)
from code.teacher import (
    CONTROL_DECIMATION, DEFAULT_ANGLES, KDS, KPS, NUM_ACTIONS, SIM_DT, WBCTeacher,
)
from code.apps.warehouse_demo.nav_core import NavParams, shape_command
from code.planner.follower import WaypointFollower
from code.apps.warehouse_datagen.scene import EpisodePlan

__all__ = ["run_warehouse_dart_episode", "FPS", "PROPRIO_DIM"]


def run_warehouse_dart_episode(
    teacher: WBCTeacher,
    plan: EpisodePlan,
    episode_idx: int,
    global_frame_offset: int,
    *,
    noise_sigma: float = 0.07,
    hard_maxsteps: int = 1400,
    rng_noise: Optional[np.random.Generator] = None,
    render: bool = True,
    nav_params: Optional[NavParams] = None,
    verbose: bool = False,
) -> Optional[dict]:
    """Run one warehouse DART episode.

    Args:
        teacher: Shared :class:`WBCTeacher`; its model/data are rebound to the
            warehouse arena compiled from ``plan.scene_cfg``.
        plan: The :class:`EpisodePlan` describing scene, spawn, and route/target.
        episode_idx: Episode index written into the ``episode_index`` column.
        global_frame_offset: Running frame count added to the ``index`` column.
        noise_sigma: DART joint-target noise std (rad) applied to the executed
            targets (clean target is stored as the label).
        hard_maxsteps: Hard per-episode control-step cap.
        rng_noise: RNG for the DART noise (fresh default if None).
        render: If True, render ego RGB each step (GPU) and return the sequence.
        nav_params: Steering/follower tunables (defaults to warehouse-nav values).
        verbose: Print periodic progress lines.

    Returns:
        dict {rows, ego_rgb_seq, reached, n_steps, mode} or None if the robot
        fell (episode discarded, matching the baseline recipe).
    """
    if rng_noise is None:
        rng_noise = np.random.default_rng()
    if nav_params is None:
        nav_params = NavParams()

    stop_r = nav_params.stop_r

    # ---- Compile the warehouse arena and rebind the teacher onto it ----
    model = build_warehouse_arena(plan.scene_cfg)
    data = mujoco.MjData(model)
    model.opt.timestep = SIM_DT
    teacher.model = model
    teacher.data = data
    teacher._nj = model.nq - 7
    teacher._pelvis_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")

    teacher.reset(pos_xy=plan.spawn_xy, yaw=plan.spawn_yaw)
    nj = teacher._nj

    renderer = ArenaRenderer(model) if render else None

    # ---- Target source: route follower or fixed primitive target ----
    follower = WaypointFollower(
        plan.path, arrive_radius=nav_params.arrive_radius, lookahead=nav_params.lookahead,
    ) if plan.mode == "route" else None

    phase_tracker = GaitPhaseTracker()
    rows: list = []
    ego_rgb_seq: list = []
    reached = False
    hold_count = 0
    turn_run = 0
    fallen = False

    # ---- Settle (zero vel, not logged) ----
    for _ in range(SETTLE_STEPS):
        teacher.step(vel_cmd=(0.0, 0.0, 0.0))
        if teacher.base_height < FALL_HEIGHT:
            if renderer is not None:
                renderer.close()
            return None

    step_t0 = time.time()
    for t in range(hard_maxsteps):
        bxy = teacher.base_pos[:2]
        byaw = teacher.base_yaw

        # ---- Determine steering target for this step ----
        if plan.mode == "route":
            target = follower.target((float(bxy[0]), float(bxy[1])))
            if follower.done or target is None:
                reached = True
                break
        else:
            target = plan.fixed_target

        vel, dist, yaw_err, is_turn = shape_command(
            (float(bxy[0]), float(bxy[1])), float(byaw), target, nav_params,
            turn_run=turn_run,
        )
        turn_run = turn_run + 1 if is_turn else 0

        # ---- DART step: clean label, noisy execution ----
        qpos_save = data.qpos.copy()
        qvel_save = data.qvel.copy()
        ctrl_save = data.ctrl.copy()

        clean_targets = teacher.step(vel_cmd=tuple(float(v) for v in vel))

        data.qpos[:] = qpos_save
        data.qvel[:] = qvel_save
        data.ctrl[:] = ctrl_save
        mujoco.mj_forward(model, data)

        noise = rng_noise.normal(0.0, noise_sigma, size=NUM_ACTIONS).astype(np.float32)
        noisy_targets = clean_targets + noise
        for _ in range(CONTROL_DECIMATION):
            leg_tau = (
                (noisy_targets - data.qpos[7:7 + NUM_ACTIONS]) * KPS
                + (0.0 - data.qvel[6:6 + NUM_ACTIONS]) * KDS
            )
            data.ctrl[:NUM_ACTIONS] = leg_tau
            if nj > NUM_ACTIONS:
                arm_tau = (
                    (0.0 - data.qpos[7 + NUM_ACTIONS:7 + nj]) * 100.0
                    + (0.0 - data.qvel[6 + NUM_ACTIONS:6 + nj]) * 0.5
                )
                data.ctrl[NUM_ACTIONS:nj] = arm_tau
            mujoco.mj_step(model, data)

        if teacher.base_height < FALL_HEIGHT:
            fallen = True
            break

        # ---- Observations from the noisy-executed state ----
        q_lb = data.qpos[7:22].copy()
        sin_phi, cos_phi = phase_tracker.update(q_lb)
        proprio = build_proprio(data, noisy_targets)
        gv = goal_vec(dist, yaw_err)

        if renderer is not None:
            rgb, _, _ = renderer.render_ego(data, teacher.base_yaw, render_depth=False)
            ego_rgb_seq.append(rgb)

        done_flag = int(dist < stop_r)
        rows.append({
            "frame_index":      t,
            "episode_index":    episode_idx,
            "index":            global_frame_offset + len(rows),
            "task_index":       0,
            "timestamp":        float(len(rows)) / FPS,
            "proprio":          proprio.tolist(),
            "action":           clean_targets.tolist(),   # CLEAN supervision label
            "goal":             gv.tolist(),
            "vel_cmd":          vel.tolist(),
            "done":             done_flag,
            "task_description": plan.instruction,
            "phase":            [float(sin_phi), float(cos_phi)],
        })

        if verbose and t % 100 == 0:
            sps = (t + 1) / max(time.time() - step_t0, 1e-6)
            print(f"    [wh-dart ep{episode_idx} {plan.mode}] t={t}/{hard_maxsteps} "
                  f"dist={dist:.2f} h={teacher.base_height:.3f} {sps:.0f}stp/s", flush=True)

        # ---- Primitive-mode arrival (route-mode arrival handled by follower) ----
        if plan.mode == "primitive":
            if dist < stop_r:
                reached = True
                hold_count += 1
            if reached and hold_count >= HOLD_STEPS:
                break

    if renderer is not None:
        renderer.close()

    if fallen or not rows:
        return None

    return {
        "rows":        rows,
        "ego_rgb_seq": ego_rgb_seq,
        "reached":     reached,
        "n_steps":     len(rows),
        "mode":        plan.mode,
    }
