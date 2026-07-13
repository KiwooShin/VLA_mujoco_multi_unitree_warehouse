"""fleet_eval.py — Batch evaluation of 4-robot warehouse co-simulation.

Runs N trials of the full fleet (default 5 seeds). Each trial assigns the four
robots four *distinct* occluded object-spot goals sampled so their A* paths CROSS
the hall (at least one robot drives strongly east while another drives strongly
west), forcing an aisle-sharing interaction that exercises the mutual-proximity
pause. A trial is a fleet success iff EVERY robot arrives upright and none falls.

Reports per-robot arrival/fall/efficiency and the fleet-level success rate, pause
events, and makespan (control steps until the last arrival). Per-trial JSON plus
a summary land in ``eval/fleet_nav/`` and a table is printed.

Usage
-----
PYTHONPATH=. MUJOCO_GL=egl python code/fleet/fleet_eval.py --n 5 --seed 0
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import numpy as np

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from code.fleet.fleet import Fleet
from code.fleet.robot_unit import RobotState
from code.sim.teacher import WBCTeacher
from code.warehouse.layout import CALLSIGNS, WarehouseLayout, hero_layout

_DEFAULT_OUT = str(_REPO / "eval" / "fleet_nav")
Point = Tuple[float, float]

# Occluded object-spot indices used as fleet goals (hero layout; all verified
# reachable from every home bay at the deployed inflation radius).
_GOAL_SPOTS: Tuple[int, ...] = (0, 1, 3, 4, 5, 6, 7)
_CROSS_M: float = 2.5  # a robot must gain/lose this much x to count as "crossing"


def sample_crossing_goals(
    rng: np.random.Generator, layout: WarehouseLayout, callsigns: List[str],
) -> Tuple[Dict[str, Point], Dict[str, int]]:
    """Sample 4 distinct occluded goals whose paths cross the hall.

    Accepts an assignment only when at least one robot must drive strongly east
    (goal_x - spawn_x >= +2.5 m) and another strongly west (<= -2.5 m), so the
    fleet always shares an aisle somewhere near the hall centre.

    Args:
        rng: Caller-owned RNG.
        layout: The (hero) layout supplying spawns + object spots.
        callsigns: The robots to assign, in priority order.

    Returns:
        (goals, goal_index) mapping callsign -> goal (x, y) and -> spot index.
    """
    spots = layout.object_spots
    for _ in range(400):
        chosen = rng.choice(np.array(_GOAL_SPOTS), size=len(callsigns), replace=False)
        goals = {cs: (float(spots[int(k)][0]), float(spots[int(k)][1]))
                 for cs, k in zip(callsigns, chosen)}
        deltas = [goals[cs][0] - layout.spawn_poses[cs][0] for cs in callsigns]
        if max(deltas) >= _CROSS_M and min(deltas) <= -_CROSS_M:
            return goals, {cs: int(k) for cs, k in zip(callsigns, chosen)}
    # Fallback: a hand-picked crossing assignment (Alpha east, Delta west).
    idx = {"Alpha": 7, "Bravo": 4, "Charlie": 3, "Delta": 6}
    goals = {cs: (float(spots[idx[cs]][0]), float(spots[idx[cs]][1]))
             for cs in callsigns}
    return goals, {cs: idx[cs] for cs in callsigns}


def run_trial(
    seed: int, layout: WarehouseLayout, max_steps: int,
    teachers: Dict[str, WBCTeacher], engage: float, release: float,
) -> dict:
    """Run one fleet trial and return its per-robot + fleet metrics dict."""
    callsigns = list(CALLSIGNS)
    rng = np.random.default_rng(seed)
    goals, goal_idx = sample_crossing_goals(rng, layout, callsigns)

    t0 = time.time()
    fleet = Fleet(layout, goals, callsigns=callsigns, build_viz=False,
                  teachers=teachers, engage=engage, release=release, seed=seed)
    fleet.run(max_steps)
    elapsed = time.time() - t0

    robots: Dict[str, dict] = {}
    for name in callsigns:
        u = fleet.units[name]
        eff = (u.planned_length / u.walked_length) if u.walked_length > 1e-6 else 0.0
        robots[name] = {
            "goal_index": goal_idx[name],
            "goal_xy": list(goals[name]),
            "state": u.state.value,
            "arrived": u.state == RobotState.ARRIVED,
            "fell": u.fell,
            "plan_ok": bool(u.plan_ok),
            "walked": round(u.walked_length, 3),
            "planned": round(u.planned_length, 3),
            "efficiency": round(min(eff, 1.0), 3),
            "min_wall_clearance": round(float(u.min_wall_clearance), 3),
            "wall_collision": u.wall_collision,
            "arrive_step": fleet.arrive_step.get(name),
        }
    return {
        "seed": seed,
        "fleet_success": fleet.all_arrived and not fleet.any_fell,
        "all_arrived": fleet.all_arrived,
        "any_fell": fleet.any_fell,
        "n_arrived": sum(1 for r in robots.values() if r["arrived"]),
        "pause_events": fleet.pause_events,
        "makespan": fleet.makespan,
        "steps": fleet.step_count,
        "time_s": round(elapsed, 1),
        "robots": robots,
    }


def run_eval(n: int, base_seed: int, out_dir: str, max_steps: int,
             engage: float, release: float) -> dict:
    """Run ``n`` fleet trials, write per-trial + summary JSON, print a table."""
    os.makedirs(out_dir, exist_ok=True)
    layout = hero_layout()
    teachers = {name: WBCTeacher(use_gpu=True) for name in CALLSIGNS}

    trials: List[dict] = []
    t0 = time.time()
    for t in range(n):
        seed = base_seed + t
        trial = run_trial(seed, layout, max_steps, teachers, engage, release)
        trials.append(trial)
        with open(os.path.join(out_dir, f"trial_{t:02d}.json"), "w") as f:
            json.dump(trial, f, indent=2)
        _print_trial(t, trial)

    summary = _summarize(trials, base_seed, n, time.time() - t0, engage, release)
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    _print_summary(summary, out_dir)
    return summary


def _summarize(trials, base_seed, n, elapsed, engage, release) -> dict:
    """Aggregate trial results."""
    n_success = sum(1 for t in trials if t["fleet_success"])
    makespans = [t["makespan"] for t in trials if t["makespan"] is not None]
    per_robot: Dict[str, int] = {c: 0 for c in CALLSIGNS}
    for t in trials:
        for name, r in t["robots"].items():
            if r["arrived"]:
                per_robot[name] += 1
    return {
        "n": n,
        "base_seed": base_seed,
        "fleet_success": n_success,
        "fleet_success_rate": round(n_success / max(1, n), 3),
        "per_robot_arrivals": per_robot,
        "n_falls_total": sum(sum(1 for r in t["robots"].values() if r["fell"])
                             for t in trials),
        "mean_pause_events": round(float(np.mean([t["pause_events"] for t in trials])), 2),
        "mean_makespan": round(float(np.mean(makespans)), 1) if makespans else None,
        "max_makespan": max(makespans) if makespans else None,
        "engage_m": engage,
        "release_m": release,
        "total_time_s": round(elapsed, 1),
    }


def _print_trial(t: int, trial: dict) -> None:
    """Print a per-trial line + one line per robot."""
    print(f"\n[trial {t:02d}] seed={trial['seed']} "
          f"success={trial['fleet_success']} arrived={trial['n_arrived']}/4 "
          f"pauses={trial['pause_events']} makespan={trial['makespan']} "
          f"({trial['time_s']}s)", flush=True)
    for name in CALLSIGNS:
        r = trial["robots"][name]
        print(f"    {name:<7} -> spot{r['goal_index']} {tuple(r['goal_xy'])}  "
              f"{r['state']:<8} eff={r['efficiency']:.2f} "
              f"clr={r['min_wall_clearance']:.2f} arr={r['arrive_step']}", flush=True)


def _print_summary(summary: dict, out_dir: str) -> None:
    """Print the aggregate summary block."""
    print("\n" + "=" * 74)
    print(f"FLEET EVAL  ({summary['n']} trials, seeds "
          f"{summary['base_seed']}..{summary['base_seed'] + summary['n'] - 1})")
    print("-" * 74)
    print(f"fleet success (all-4-arrive, no falls): "
          f"{summary['fleet_success']}/{summary['n']} "
          f"({summary['fleet_success_rate'] * 100:.0f}%)")
    print(f"per-robot arrivals: {summary['per_robot_arrivals']}")
    print(f"falls total={summary['n_falls_total']}   "
          f"mean pause events={summary['mean_pause_events']}   "
          f"mean makespan={summary['mean_makespan']}  "
          f"(max {summary['max_makespan']})")
    print(f"pause band: engage={summary['engage_m']}m release={summary['release_m']}m"
          f"   time={summary['total_time_s']}s")
    print(f"artifacts: {out_dir}/")
    print("=" * 74, flush=True)


def main(argv: Optional[List[str]] = None) -> None:
    """Parse CLI args and run the fleet evaluation."""
    ap = argparse.ArgumentParser(description="4-robot warehouse fleet nav eval")
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=str, default=_DEFAULT_OUT)
    ap.add_argument("--max-steps", type=int, default=2400)
    ap.add_argument("--engage", type=float, default=1.0)
    ap.add_argument("--release", type=float, default=1.2)
    args = ap.parse_args(argv)
    run_eval(args.n, args.seed, args.out, args.max_steps, args.engage, args.release)


if __name__ == "__main__":
    main()
