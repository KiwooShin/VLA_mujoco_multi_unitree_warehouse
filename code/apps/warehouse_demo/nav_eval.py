"""nav_eval.py — Batch evaluation of warehouse A* navigation.

Runs N episodes, each a random home-bay spawn to a random occluded
``object_spot`` goal on the hero layout, and reports the fleet-relevant nav
metrics: success rate, mean path efficiency (planned / walked), falls and wall
collisions. Per-episode JSON plus a summary JSON land in ``eval/warehouse_nav/``
and a table is printed.

Usage
-----
PYTHONPATH=. MUJOCO_GL=egl python code/apps/warehouse_demo/nav_eval.py \
    --n 10 --seed 0
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import numpy as np

_REPO = Path(__file__).resolve().parents[3]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from code.apps.warehouse_demo.nav_rollout import NavParams, NavResult, run_nav_rollout
from code.sim.teacher import WBCTeacher
from code.warehouse.layout import CALLSIGNS, hero_layout
from code.warehouse.arena import warehouse_scene_cfg

_DEFAULT_OUT = str(_REPO / "eval" / "warehouse_nav")


def sample_episode(
    rng: np.random.Generator, layout, n_spots: int,
) -> Tuple[str, int]:
    """Pick a random spawn callsign and a random object-spot goal index."""
    callsign = str(rng.choice(list(CALLSIGNS)))
    goal_idx = int(rng.integers(n_spots))
    return callsign, goal_idx


def run_eval(
    n: int, seed: int, out_dir: str, max_steps: int,
    params: Optional[NavParams] = None, record_video: bool = False,
) -> dict:
    """Run ``n`` navigation episodes and write per-episode + summary JSON.

    Args:
        n: Number of episodes.
        seed: Base RNG seed for spawn/goal sampling.
        out_dir: Output directory for JSON artifacts.
        max_steps: Per-episode control-step cap.
        params: Navigation tunables (defaults used if None).
        record_video: If True, also record each episode's BEV MP4 to ``out_dir``.

    Returns:
        The summary dict (also written to ``summary.json``).
    """
    params = params or NavParams()
    os.makedirs(out_dir, exist_ok=True)
    layout = hero_layout()
    spots = layout.object_spots
    rng = np.random.default_rng(seed)
    teacher = WBCTeacher(use_gpu=True)

    rows: List[dict] = []
    results: List[NavResult] = []
    t0 = time.time()
    for ep in range(n):
        callsign, goal_idx = sample_episode(rng, layout, len(spots))
        goal_xy = spots[goal_idx]
        cfg = warehouse_scene_cfg(layout, robot=callsign,
                                  rng=np.random.default_rng(seed + 1000 + ep))
        res = run_nav_rollout(
            cfg, goal_xy, seed=seed + ep, max_steps=max_steps, teacher=teacher,
            params=params, record_video=record_video, out_dir=out_dir,
            video_name=f"ep{ep:02d}_{callsign}",
        )
        results.append(res)
        row = res.to_dict()
        row.update({"episode": ep, "spawn": callsign, "goal_index": goal_idx})
        rows.append(row)
        with open(os.path.join(out_dir, f"episode_{ep:02d}.json"), "w") as f:
            json.dump(row, f, indent=2)
        print(_fmt_row(ep, callsign, goal_xy, res), flush=True)

    summary = _summarize(rows, results, seed, n, time.time() - t0, params)
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    _print_table(rows, summary, out_dir)
    return summary


def _summarize(rows, results, seed, n, elapsed, params) -> dict:
    """Aggregate episode results into a summary dict."""
    succ = [r for r in results if r.success]
    effs = [r.path_efficiency for r in succ]
    clears = [r.min_wall_clearance for r in results
              if r.min_wall_clearance == r.min_wall_clearance]  # drop NaN
    return {
        "n": n,
        "seed": seed,
        "n_success": len(succ),
        "success_rate": round(len(succ) / max(1, n), 3),
        "n_fell": sum(1 for r in results if r.fell),
        "n_wall_collision": sum(1 for r in results if r.wall_collision),
        "n_plan_failed": sum(1 for r in results if r.outcome == "plan_failed"),
        "n_timeout": sum(1 for r in results if r.outcome == "timeout"),
        "mean_path_efficiency": round(float(np.mean(effs)), 3) if effs else 0.0,
        "mean_min_wall_clearance": round(float(np.mean(clears)), 3) if clears else 0.0,
        "min_min_wall_clearance": round(float(np.min(clears)), 3) if clears else 0.0,
        "mean_steps_success": round(float(np.mean([r.steps for r in succ])), 1) if succ else 0.0,
        "total_time_s": round(elapsed, 1),
        "params": {
            "inflate_radius": params.inflate_radius,
            "arrive_radius": params.arrive_radius,
            "lookahead": params.lookahead,
            "max_vx": params.max_vx,
            "max_wz": params.max_wz,
        },
    }


def _fmt_row(ep: int, spawn: str, goal, res: NavResult) -> str:
    """One-line per-episode log."""
    return (f"[ep{ep:02d}] {spawn:7s} -> ({goal[0]:+.1f},{goal[1]:+.1f})  "
            f"{res.outcome:11s} steps={res.steps:4d} eff={res.path_efficiency:.2f} "
            f"clr={res.min_wall_clearance:.2f} fell={int(res.fell)} "
            f"wall={int(res.wall_collision)}")


def _print_table(rows, summary, out_dir) -> None:
    """Print the results table and the summary block."""
    print("\n" + "=" * 78)
    print(f"{'ep':>3} {'spawn':>7} {'goal':>13} {'outcome':>11} {'steps':>6} "
          f"{'eff':>5} {'clr':>5} {'fell':>4} {'wall':>4}")
    print("-" * 78)
    for r in rows:
        gx, gy = r["goal_xy"]
        print(f"{r['episode']:>3} {r['spawn']:>7} "
              f"({gx:+.1f},{gy:+.1f})".rjust(13)
              + f" {r['outcome']:>11} {r['steps']:>6} "
              f"{r['path_efficiency']:>5.2f} {r['min_wall_clearance']:>5.2f} "
              f"{int(r['fell']):>4} {int(r['wall_collision']):>4}")
    print("-" * 78)
    print(f"success {summary['n_success']}/{summary['n']} "
          f"({summary['success_rate']*100:.0f}%)   "
          f"mean_eff={summary['mean_path_efficiency']}   "
          f"falls={summary['n_fell']}   wall_collisions={summary['n_wall_collision']}   "
          f"plan_failed={summary['n_plan_failed']}   timeouts={summary['n_timeout']}")
    print(f"mean_min_clearance={summary['mean_min_wall_clearance']}m  "
          f"min={summary['min_min_wall_clearance']}m   "
          f"time={summary['total_time_s']}s")
    print(f"artifacts: {out_dir}/")
    print("=" * 78, flush=True)


def main(argv: Optional[List[str]] = None) -> None:
    """Parse CLI args and run the evaluation."""
    ap = argparse.ArgumentParser(description="Warehouse A* nav evaluation")
    ap.add_argument("--n", type=int, default=15)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=str, default=_DEFAULT_OUT)
    ap.add_argument("--max-steps", type=int, default=2600)
    ap.add_argument("--record-video", action="store_true")
    args = ap.parse_args(argv)
    run_eval(args.n, args.seed, args.out, args.max_steps,
             record_video=args.record_video)


if __name__ == "__main__":
    main()
