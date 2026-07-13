"""nav_video.py — Record annotated wide-BEV MP4s of warehouse navigation.

Renders 2-3 episodes with the fixed wide bird's-eye framing that shows the WHOLE
16x12 m hall (``code.apps.warehouse_demo.bev.fit_bev_camera`` — deliberately not
the single-robot follow-cam), the planned A* path drawn as a polyline overlay,
the walked trail, the goal marker, an ego picture-in-picture, and a per-frame
HUD. Individual clips plus a concatenated reel land in ``ops/phase1c/`` (which is
git-ignored via the ``*.mp4`` rule).

Usage
-----
PYTHONPATH=. MUJOCO_GL=egl python code/apps/warehouse_demo/nav_video.py \
    --out ops/phase1c
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import List, Optional, Tuple

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import numpy as np

_REPO = Path(__file__).resolve().parents[3]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from code.apps.warehouse_demo.nav_rollout import NavParams, run_nav_rollout
from code.sim.teacher import WBCTeacher
from code.warehouse.arena import warehouse_scene_cfg
from code.warehouse.layout import hero_layout

# Curated episodes (callsign, object_spot index) chosen to showcase A* routing:
# a long haul into the occluded NE alcove, and a cross-hall diagonal.
_DEFAULT_EPISODES: Tuple[Tuple[str, int], ...] = (
    ("Alpha", 5),    # -> (6.5, 4.7) occluded NE alcove, longest route
    ("Delta", 3),    # -> (-1.5, 3.3) north aisle, east-to-west cross
)


def record_episodes(
    episodes, out_dir: str, seed: int, max_steps: int,
    params: Optional[NavParams] = None,
) -> List[str]:
    """Record one MP4 per episode and return the written paths."""
    params = params or NavParams()
    os.makedirs(out_dir, exist_ok=True)
    layout = hero_layout()
    spots = layout.object_spots
    teacher = WBCTeacher(use_gpu=True)
    paths: List[str] = []
    for i, (callsign, goal_idx) in enumerate(episodes):
        goal_xy = spots[goal_idx]
        cfg = warehouse_scene_cfg(layout, robot=callsign,
                                  rng=np.random.default_rng(seed + i))
        res = run_nav_rollout(
            cfg, goal_xy, seed=seed + i, max_steps=max_steps, teacher=teacher,
            params=params, record_video=True, out_dir=out_dir,
            video_name=f"nav_{i:02d}_{callsign}_{goal_idx}",
        )
        print(f"[video] {callsign} -> {goal_xy}: {res.outcome} "
              f"steps={res.steps} eff={res.path_efficiency:.2f} -> {res.video_path}",
              flush=True)
        if res.video_path:
            paths.append(res.video_path)
    return paths


def concat_reel(clips: List[str], reel_path: str) -> Optional[str]:
    """Concatenate clips into one reel MP4 (returns its path, or None)."""
    import cv2

    valid = [p for p in clips if p and os.path.isfile(p)]
    if not valid:
        return None
    cap0 = cv2.VideoCapture(valid[0])
    w = int(cap0.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap0.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap0.get(cv2.CAP_PROP_FPS)) or 30
    cap0.release()
    os.makedirs(os.path.dirname(os.path.abspath(reel_path)), exist_ok=True)
    out = cv2.VideoWriter(reel_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for vp in valid:
        cap = cv2.VideoCapture(vp)
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame.shape[1] != w or frame.shape[0] != h:
                frame = cv2.resize(frame, (w, h))
            out.write(frame)
        cap.release()
    out.release()
    print(f"[video] reel: {reel_path} ({len(valid)} clips)", flush=True)
    return reel_path


def main(argv: Optional[List[str]] = None) -> None:
    """Parse CLI args, record the curated episodes, and build a reel."""
    ap = argparse.ArgumentParser(description="Warehouse nav BEV video recorder")
    ap.add_argument("--out", type=str, default=str(_REPO / "ops" / "phase1c"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-steps", type=int, default=2600)
    ap.add_argument("--no-reel", action="store_true")
    args = ap.parse_args(argv)

    clips = record_episodes(_DEFAULT_EPISODES, args.out, args.seed, args.max_steps)
    if not args.no_reel and clips:
        concat_reel(clips, os.path.join(args.out, "nav_reel.mp4"))


if __name__ == "__main__":
    main()
