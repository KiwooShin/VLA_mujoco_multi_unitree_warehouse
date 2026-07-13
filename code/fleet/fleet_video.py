"""fleet_video.py — Whole-hall BEV video + cross-visibility proof for the fleet.

Records an MP4 of all four G1 robots walking SIMULTANEOUSLY across the shared
warehouse, drawn from the shared kinematic viz model
(:class:`~code.fleet.viz.FleetViz`): a fixed wide bird's-eye framing of the whole
16x12 m hall with, per robot, its accent-coloured planned A* path, its walked
trail, a heading marker and a name label projected above its head, plus a HUD of
per-robot status lines.

Also emits the mandatory **cross-visibility proof**: robot Alpha's ego camera
rendered against the viz model with Bravo standing ~2.5 m in front of it, then
again with Bravo teleported away — the two PNGs plus their measured pixel diff
demonstrate that one robot's camera really does see the others (the whole reason
the viz model exists). An optional 2x2 ego strip shows every robot's live view.

Outputs land in ``ops/phase2/`` (git-ignored). 30 fps.

Usage
-----
PYTHONPATH=. MUJOCO_GL=egl python code/fleet/fleet_video.py --out ops/phase2
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import numpy as np

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from code.apps.warehouse_demo import bev as bevmod
from code.fleet.fleet import Fleet
from code.fleet.viz import BEV_H, BEV_W, FleetViz
from code.warehouse.arena import warehouse_scene_cfg
from code.warehouse.layout import CALLSIGNS, hero_layout, rooms_layout

_LAYOUTS = {"hero": hero_layout, "rooms": rooms_layout}

Point = Tuple[float, float]
_DEFAULT_OUT = str(_REPO / "ops" / "phase2")
# Seconds of simulated time per control step (50 Hz control loop) — HUD readout.
_SIM_DT: float = 0.02

# Per-callsign overlay colours in BGR (cv2 order), matching the torso accents.
ACCENT_BGR: Dict[str, Tuple[int, int, int]] = {
    "Alpha": (40, 40, 224),     # red
    "Bravo": (224, 90, 40),     # blue
    "Charlie": (40, 205, 238),  # yellow
    "Delta": (210, 60, 158),    # purple
}

# A crossing goal assignment (spot indices) that reliably produces an aisle
# interaction for the demo (Alpha east, Delta west; Bravo/Charlie swap north).
_DEMO_GOALS: Dict[str, int] = {"Alpha": 7, "Bravo": 4, "Charlie": 3, "Delta": 6}

# Per-layout crossing goals. Rooms (F6): every robot leaves its south loading bay
# and drives cross-room through the open doorways — Alpha east into storage B,
# Delta west into storage A (they cross the middle), Bravo/Charlie fan into the
# back room from opposite sides — so paths cross and the proximity pause fires.
_DEMO_GOALS_BY_LAYOUT: Dict[str, Dict[str, int]] = {
    "hero": _DEMO_GOALS,
    "rooms": {"Alpha": 5, "Bravo": 9, "Charlie": 8, "Delta": 2},
}


def _bgr(rgb: np.ndarray) -> np.ndarray:
    """Convert an RGB frame to a contiguous BGR frame for cv2."""
    import cv2

    return np.ascontiguousarray(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))


def draw_fleet_overlay(frame: np.ndarray, cam: bevmod.BevCamera, fleet: Fleet,
                       trails: Dict[str, List[Point]]) -> None:
    """Draw per-robot paths, trails, markers, labels and a HUD (in-place)."""
    import cv2

    for name in fleet.callsigns:
        unit = fleet.units[name]
        color = ACCENT_BGR.get(name, (200, 200, 200))
        if unit.planned_path:
            bevmod.draw_polyline(frame, cam, unit.planned_path, color,
                                 thickness=2, z=0.03)
        if len(trails[name]) >= 2:
            bevmod.draw_polyline(frame, cam, trails[name], color, thickness=2,
                                 z=0.02)
        if unit.goal_xy is not None:
            bevmod.draw_marker(frame, cam, unit.goal_xy, color=color, radius=8)
        bevmod.draw_robot(frame, cam, unit.xy, unit.yaw, color=color)
        # Name label projected above the robot's head.
        u, v = cam.project_xy(unit.xy, z=1.8)
        cv2.putText(frame, name, (u - 22, v), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(frame, name, (u - 22, v), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    color, 1, cv2.LINE_AA)

    hud = ([f"sim time {fleet.step_count * _SIM_DT:5.1f} s   step {fleet.step_count}"
            f"   pauses {fleet.pause_events}"] + fleet.statuses())
    bevmod.put_hud(frame, hud)


def record_fleet_video(
    out_dir: str, seed: int, max_steps: int, fps: int, decimation: int,
    layout_name: str = "hero", locomotion: str = "teacher",
    vla_ckpt: Optional[str] = None, vla_device: Optional[str] = None,
) -> Tuple[Optional[str], Fleet]:
    """Record the whole-hall fleet BEV MP4 and return (path, fleet)."""
    import cv2

    os.makedirs(out_dir, exist_ok=True)
    layout = _LAYOUTS.get(layout_name, hero_layout)()
    spots = layout.object_spots
    goals = {cs: (float(spots[i][0]), float(spots[i][1]))
             for cs, i in _DEMO_GOALS_BY_LAYOUT.get(layout_name, _DEMO_GOALS).items()}
    fleet = Fleet(layout, goals, build_viz=True, seed=seed, locomotion=locomotion,
                  vla_ckpt=vla_ckpt, vla_device=vla_device)
    viz = fleet.viz
    assert viz is not None
    # The shared viz model is display-only (never stepped); robots roam and
    # transiently overlap in it (esp. crossing paths through rooms doorways), so
    # disable contact/constraint solving to keep its render-only ``mj_forward``
    # from tripping FactorizeHessian on a degenerate overlap (same guard the
    # MissionRunner applies).
    import mujoco
    viz.model.opt.disableflags |= (
        int(mujoco.mjtDisableBit.mjDSBL_CONTACT)
        | int(mujoco.mjtDisableBit.mjDSBL_CONSTRAINT))
    cam = bevmod.fit_bev_camera(
        layout.hall_x, layout.hall_y, width=BEV_W, height=BEV_H,
        fovy_deg=float(viz.model.vis.global_.fovy))

    trails: Dict[str, List[Point]] = {c: [] for c in fleet.callsigns}
    frames: List[np.ndarray] = []

    def on_step(fl: Fleet, i: int) -> None:
        for name in fl.callsigns:
            trails[name].append(fl.units[name].xy)
        if i % decimation == 0:
            frame = _bgr(viz.render_bev(cam))
            draw_fleet_overlay(frame, cam, fl, trails)
            frames.append(frame)

    fleet.run(max_steps, on_step=on_step)
    # A short freeze on the final frame so the ending reads clearly.
    if frames:
        frames.extend([frames[-1]] * fps)

    path = None
    if frames:
        path = os.path.join(out_dir, "fleet_bev.mp4")
        h, w = frames[0].shape[:2]
        writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
        for f in frames:
            writer.write(f)
        writer.release()
    print(f"[fleet-video] {path}  frames={len(frames)}  "
          f"arrived={sum(u.done for u in fleet.units.values())}/4 "
          f"pauses={fleet.pause_events} makespan={fleet.makespan}", flush=True)
    return path, fleet


def _robot_qpos(nq: int, x: float, y: float, yaw: float) -> np.ndarray:
    """Build a standing per-robot physics qpos at (x, y, yaw) for viz sync."""
    q = np.zeros(nq)
    q[0:3] = [x, y, 0.72]
    q[3:7] = [math.cos(yaw / 2.0), 0.0, 0.0, math.sin(yaw / 2.0)]
    q[7:22] = [-0.1, 0, 0, 0.3, -0.2, 0, -0.1, 0, 0, 0.3, -0.2, 0, 0, 0, 0]
    return q


def cross_visibility_proof(out_dir: str) -> float:
    """Render Alpha's ego with Bravo present vs. teleported away; save PNGs.

    Returns:
        The fraction of Alpha's ego pixels that change (>20 intensity) when
        Bravo is added ~2.5 m in front — the measured cross-visibility signal.
    """
    import cv2

    os.makedirs(out_dir, exist_ok=True)
    cfg = warehouse_scene_cfg(hero_layout(), rng=np.random.default_rng(0))
    viz = FleetViz(cfg, CALLSIGNS)
    nq = viz.robot_nq
    alpha_yaw = math.pi / 2.0  # Alpha faces +y
    far = {"Charlie": _robot_qpos(nq, 100, 100, 0),
           "Delta": _robot_qpos(nq, 120, 120, 0)}

    seen = {"Alpha": _robot_qpos(nq, 0.0, -1.0, alpha_yaw),
            "Bravo": _robot_qpos(nq, 0.0, 1.5, 0.0), **far}
    viz.sync(seen)
    near = viz.render_ego("Alpha", alpha_yaw)

    clear = {"Alpha": _robot_qpos(nq, 0.0, -1.0, alpha_yaw),
             "Bravo": _robot_qpos(nq, 150, 150, 0.0), **far}
    viz.sync(clear)
    away = viz.render_ego("Alpha", alpha_yaw)

    diff = np.abs(near.astype(np.int16) - away.astype(np.int16))
    frac = float((diff.max(axis=2) > 20).mean())
    cv2.imwrite(os.path.join(out_dir, "xvis_alpha_sees_bravo.png"), _bgr(near))
    cv2.imwrite(os.path.join(out_dir, "xvis_alpha_clear.png"), _bgr(away))
    viz.close()
    print(f"[cross-visibility] Bravo occupies {frac * 100:.1f}% of Alpha's ego "
          f"view -> {out_dir}/xvis_alpha_sees_bravo.png (vs _clear.png)", flush=True)
    return frac


def ego_strip(out_dir: str, seed: int, at_step: int) -> Optional[str]:
    """Save a 2x2 ego strip (all four robots' live views) at a mid-run step."""
    import cv2

    layout = hero_layout()
    spots = layout.object_spots
    goals = {cs: (float(spots[i][0]), float(spots[i][1]))
             for cs, i in _DEMO_GOALS.items()}
    fleet = Fleet(layout, goals, build_viz=True, seed=seed)
    viz = fleet.viz
    assert viz is not None
    fleet.run(at_step)
    tiles: List[np.ndarray] = []
    for name in fleet.callsigns:
        rgb = viz.render_ego(name, fleet.units[name].yaw)
        tile = _bgr(rgb)
        cv2.putText(tile, name, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(tile, name, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    ACCENT_BGR.get(name, (255, 255, 255)), 1, cv2.LINE_AA)
        tiles.append(tile)
    top = np.hstack(tiles[0:2])
    bot = np.hstack(tiles[2:4])
    strip = np.vstack([top, bot])
    path = os.path.join(out_dir, "fleet_ego_strip.png")
    cv2.imwrite(path, strip)
    fleet.close()
    print(f"[ego-strip] {path}  (step {at_step})", flush=True)
    return path


def main(argv: Optional[List[str]] = None) -> None:
    """Parse CLI args, record the video + the cross-visibility artifacts."""
    ap = argparse.ArgumentParser(description="Fleet BEV video + cross-visibility")
    ap.add_argument("--out", type=str, default=_DEFAULT_OUT)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-steps", type=int, default=2000)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--decimation", type=int, default=3)
    ap.add_argument("--ego-step", type=int, default=400)
    ap.add_argument("--no-ego-strip", action="store_true")
    ap.add_argument("--layout", choices=tuple(_LAYOUTS), default="hero",
                    help="hero hall or the multi-room rooms_layout (F6)")
    ap.add_argument("--locomotion", choices=("teacher", "vla"), default="teacher",
                    help="WBC walk policy (default) or the trained VLA policy (F5)")
    ap.add_argument("--ckpt", type=str, default=None,
                    help="GroundedNav checkpoint for --locomotion vla (default: F5 fine-tune)")
    ap.add_argument("--device", type=str, default=None,
                    help="Torch device for the VLA policy (cuda|cpu; default auto)")
    ap.add_argument("--video-only", action="store_true",
                    help="skip the hero-only cross-visibility proof + ego strip")
    args = ap.parse_args(argv)

    if not args.video_only and args.layout == "hero":
        cross_visibility_proof(args.out)
    path, fleet = record_fleet_video(args.out, args.seed, args.max_steps,
                                     args.fps, args.decimation,
                                     layout_name=args.layout,
                                     locomotion=args.locomotion,
                                     vla_ckpt=args.ckpt, vla_device=args.device)
    fleet.close()
    if not args.no_ego_strip and not args.video_only and args.layout == "hero":
        ego_strip(args.out, args.seed, args.ego_step)


if __name__ == "__main__":
    main()
