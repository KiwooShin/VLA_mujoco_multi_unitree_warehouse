"""gen_eval.py — Randomized-layout generalization eval (post-release rigor).

Demonstrates, with numbers, that the whole collaborative-fetch system — planner,
fleet, comms, delegated search, fine-tuned detector and VLA locomotion — is not
overfit to the two hand-built maps (``hero_layout`` / ``rooms_layout``). It
samples N *randomized* layouts from a family (:func:`sample_layout` hero jitter
or :func:`sample_rooms_layout` four-room family), runs the standard seeded
mission classes on each, and tabulates success/falls/plan-failures/search-steps
per layout, per family and in aggregate — for BOTH the full learned stack
(``--perception groundnet --locomotion vla``) and the oracle+teacher baseline.

Every sampled layout is self-validated AND plan-reachable at 0.40/0.45 m
inflation before any mission runs (the sec-5b gate lives in the samplers), so a
mission that still fails is a genuine generalization result, not a broken map.

Mission classes (same contract as ``mission_eval``):

* **A/B/C** (hero family) — addressed fetch with the target visible to the owner
  (A), only to a peer (B), or to nobody / full delegated search (C);
* **C** (rooms family) — every rooms spot is hidden from the bays, so all fetches
  are full room-to-room delegated searches;
* **D** — fleet-addressed allocation correctness (path-shortest robot).

Usage
-----
PYTHONPATH=. MUJOCO_GL=egl python code/fleet/gen_eval.py \\
    --layout-family rooms --layout-seeds 8 --mission-seeds 3 \\
    --stacks both --out ops/gen_eval/rooms
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from statistics import median
from typing import Dict, List, Optional, Sequence, Tuple

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import numpy as np

from code.fleet.mission_eval import _classify_spots, run_mission
from code.fleet.search import region_name_for_xy, search_regions_for_layout
from code.fleet.visibility import VisibilityConfig
from code.sim.teacher import WBCTeacher
from code.warehouse.layout import (CALLSIGNS, WarehouseLayout, sample_layout,
                                   sample_rooms_layout)

_DEFAULT_OUT = str(_REPO / "ops" / "gen_eval")

# The two stacks under test. "learned" is the full trained pipeline; "baseline"
# is the oracle-perception + WBC-teacher-locomotion reference the system was
# bootstrapped from. Identical mission logic / message flow in both.
STACKS: Dict[str, Dict[str, str]] = {
    "learned": {"perception_mode": "groundnet", "locomotion": "vla"},
    "baseline": {"perception_mode": "oracle", "locomotion": "teacher"},
}


# ---------------------------------------------------------------------------
# Mission plans (layout-parameterized; mirror mission_eval's fixed-map plans).
# ---------------------------------------------------------------------------
def hero_plan(layout: WarehouseLayout, seeds: int) -> List[Tuple[str, int, int]]:
    """Build the ``(class, seed, spot)`` plan for a hero-family layout.

    Classifies each object spot by who can see it from the spawn bays and, like
    ``mission_eval._plan``, emits ``3*seeds`` A/B/C fetches (falling back through
    the visibility groups when a class is empty for this sampled layout) plus 3
    fleet-addressed (D) allocation checks.
    """
    groups = _classify_spots(layout, VisibilityConfig())
    plan: List[Tuple[str, int, int]] = []
    for k in range(seeds):
        for klass in ("A", "B", "C"):
            cands = (groups[klass] or groups["C"] or groups["B"]
                     or groups["A"])
            plan.append((klass, k, cands[k % len(cands)]))
    allspots = groups["A"] + groups["B"] + groups["C"]
    for j in range(3):
        plan.append(("D", 100 + j, allspots[j % len(allspots)]))
    return plan


def rooms_plan(layout: WarehouseLayout, seeds: int) -> List[Tuple[str, int, int]]:
    """Build the ``(class, seed, spot)`` plan for a rooms-family layout.

    Mirrors ``mission_eval._rooms_plan``: every searchable-room spot is hidden
    from the bays, so this emits ``3*seeds`` class-C delegated searches cycling
    through the searchable rooms (storage A/B + back room; the fleet covers its
    own loading room) plus 3 fleet-addressed (D) allocations.
    """
    regions = search_regions_for_layout(layout)
    searchable = [i for i, (x, y) in enumerate(layout.object_spots)
                  if region_name_for_xy(layout, (x, y)) in regions]
    plan: List[Tuple[str, int, int]] = []
    for k in range(seeds):
        for j in range(3):
            plan.append(("C", k, searchable[(3 * k + j) % len(searchable)]))
    for j in range(3):
        plan.append(("D", 100 + j, searchable[j % len(searchable)]))
    return plan


def build_plan(family: str, layout: WarehouseLayout,
               seeds: int) -> List[Tuple[str, int, int]]:
    """Dispatch to the family's plan builder."""
    return rooms_plan(layout, seeds) if family == "rooms" else hero_plan(layout, seeds)


def sample_family_layout(family: str, seed: int) -> WarehouseLayout:
    """Deterministically sample one layout of ``family`` for integer ``seed``."""
    rng = np.random.default_rng(seed)
    return sample_rooms_layout(rng) if family == "rooms" else sample_layout(rng)


# ---------------------------------------------------------------------------
# Summaries
# ---------------------------------------------------------------------------
def _pct(xs: Sequence[float], p: float) -> float:
    """Return the ``p``-quantile (0..100) of ``xs`` (empty -> 0.0)."""
    if not xs:
        return 0.0
    s = sorted(xs)
    idx = min(len(s) - 1, int(round((p / 100.0) * (len(s) - 1))))
    return float(s[idx])


def summarize(results: List[dict]) -> dict:
    """Aggregate a bag of mission dicts into a metrics summary.

    Groups A/B/C fetches (success = on-pad + task-complete + no fall) from D
    allocations (correct = allocator winner matches the A* argmin), and reports
    falls, plan failures (outcome failed/timeout on A-C), and the search-steps
    distribution (total steps + first-found step) for the A-C set.
    """
    ac = [r for r in results if r["class"] in ("A", "B", "C")]
    d = [r for r in results if r["class"] == "D"]
    per_class: Dict[str, Dict[str, int]] = {}
    for r in ac:
        pc = per_class.setdefault(r["class"], {"n": 0, "ok": 0})
        pc["n"] += 1
        pc["ok"] += int(r["success"])
    steps = [r["steps"] for r in ac]
    found = [r["found_step"] for r in ac if r.get("found_step") is not None]
    infer = [r["mean_vla_infer_ms"] for r in results
             if r.get("mean_vla_infer_ms", 0.0) > 0.0]
    plan_failures = [r for r in ac if r["outcome"] in ("failed", "timeout")]
    return {
        "n_missions": len(results),
        "ac_success": sum(r["success"] for r in ac),
        "ac_total": len(ac),
        "per_class": per_class,
        "d_correct": sum(r.get("alloc_correct", False) for r in d),
        "d_total": len(d),
        "n_falls": sum(r["any_fell"] for r in results),
        "plan_failures": len(plan_failures),
        "plan_failure_missions": [
            {"class": r["class"], "seed": r["seed"], "spot": r["target_spot"],
             "outcome": r["outcome"]} for r in plan_failures],
        "n_confirmations": sum(r.get("n_confirmations", 0) for r in results),
        "steps": {
            "min": min(steps) if steps else 0,
            "median": int(median(steps)) if steps else 0,
            "p90": int(_pct(steps, 90)),
            "max": max(steps) if steps else 0,
        },
        "found_step": {
            "min": min(found) if found else 0,
            "median": int(median(found)) if found else 0,
            "p90": int(_pct(found, 90)),
            "max": max(found) if found else 0,
        },
        "mean_vla_infer_ms": round(sum(infer) / len(infer), 3) if infer else 0.0,
    }


def render_layout_bev(layout: WarehouseLayout, path: str,
                      title: Optional[str] = None) -> str:
    """Render a top-down BEV of a layout (walls/rooms/spots/bays) to a PNG.

    Pure-geometry matplotlib figure (no MuJoCo / GPU): perimeter+divider+shelf
    walls as gray/brown rectangles, room boxes as dashed outlines with their
    names, home bays as coloured pads with spawn-heading arrows, the delivery pad
    in green and object spots as numbered dots. Used to attach a picture to any
    layout seed that exposes a systematic failure.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    hx, hy = layout.hall_x / 2.0, layout.hall_y / 2.0
    fig, ax = plt.subplots(figsize=(layout.hall_x / 2.2, layout.hall_y / 2.2))
    ax.set_xlim(-hx - 0.3, hx + 0.3)
    ax.set_ylim(-hy - 0.3, hy + 0.3)
    ax.set_aspect("equal")

    for r in layout.rooms:
        ax.add_patch(Rectangle((r.cx - r.half_x, r.cy - r.half_y),
                               2 * r.half_x, 2 * r.half_y, fill=False,
                               ls="--", ec="#88aacc", lw=1.0, zorder=1))
        ax.text(r.cx, r.cy + r.half_y - 0.5, r.name, ha="center", va="top",
                fontsize=8, color="#4477aa", zorder=1)
    for w in layout.walls:
        col = "#8a6d3b" if w.name.startswith("shelf_") else (
            "#9aa" if "div" in w.name else "#666")
        ax.add_patch(Rectangle((w.cx - w.half_x, w.cy - w.half_y),
                               2 * w.half_x, 2 * w.half_y, facecolor=col,
                               edgecolor="none", zorder=2))
    for z in layout.zones:
        ax.add_patch(Rectangle((z.cx - z.half_x, z.cy - z.half_y),
                               2 * z.half_x, 2 * z.half_y,
                               facecolor=tuple(z.rgba[:3]) + (0.5,),
                               edgecolor="none", zorder=1))
        if z.name == "delivery":
            ax.text(z.cx, z.cy, "PAD", ha="center", va="center", fontsize=7,
                    color="#245", zorder=5)
    for cs, (sx, sy, yaw) in layout.spawn_poses.items():
        ax.arrow(sx, sy, 0.6 * np.cos(yaw), 0.6 * np.sin(yaw), width=0.06,
                 head_width=0.28, color="#333", zorder=6)
        ax.text(sx, sy - 0.55, cs[0], ha="center", va="top", fontsize=7,
                color="#333", zorder=6)
    for i, (x, y) in enumerate(layout.object_spots):
        ax.plot(x, y, "o", ms=8, color="#cc4444", zorder=7)
        ax.text(x + 0.15, y + 0.15, str(i), fontsize=7, color="#772222", zorder=7)

    ax.set_title(title or layout.name, fontsize=10)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def run_gen_eval(family: str, layout_seeds: int, mission_seeds: int,
                 out_dir: str, max_steps: int, stacks: Sequence[str],
                 base_seed: int = 0, vla_ckpt: Optional[str] = None,
                 vla_device: Optional[str] = None,
                 fail_threshold: float = 0.83,
                 only_layout_seed: Optional[int] = None) -> dict:
    """Sample N layouts, run every mission on every stack, tabulate + persist.

    Returns the aggregate summary dict; also writes per-layout/per-stack JSON,
    an all-missions JSONL, the aggregate ``summary.json``, and a BEV PNG for any
    (layout, stack) whose A-C success rate falls below ``fail_threshold``.

    ``only_layout_seed`` restricts the run to exactly that one layout seed (each
    layout is sampled deterministically from its integer seed and every mission
    is run on a fresh :class:`~code.fleet.mission.MissionRunner`, so a single-seed
    run reproduces that seed's slice of a full run byte-for-byte) — a quick,
    deterministic repro handle for a specific failing layout.
    """
    os.makedirs(out_dir, exist_ok=True)
    teachers = {cs: WBCTeacher(use_gpu=True) for cs in CALLSIGNS}
    all_results: List[dict] = []
    per_layout: List[dict] = []
    findings: List[dict] = []
    jsonl = open(os.path.join(out_dir, "missions.jsonl"), "w")
    t0 = time.time()
    total_missions = 0

    seeds = ([only_layout_seed] if only_layout_seed is not None
             else [base_seed + k for k in range(layout_seeds)])
    for seed in seeds:
        layout = sample_family_layout(family, seed)
        plan = build_plan(family, layout, mission_seeds)
        bev_path = os.path.join(out_dir, f"layout_seed{seed:03d}.png")
        render_layout_bev(layout, bev_path,
                          title=f"{family} seed={seed} ({layout.name})")
        print(f"\n=== {family} layout seed={seed}  {layout.name}  "
              f"{len(layout.object_spots)} spots  plan={len(plan)}x{len(stacks)} "
              f"stacks ===", flush=True)
        for stack in stacks:
            cfg = STACKS[stack]
            stack_res: List[dict] = []
            for idx, (klass, mseed, spot) in enumerate(plan):
                r = run_mission(klass, mseed, layout, spot, teachers, max_steps,
                                perception_mode=cfg["perception_mode"],
                                locomotion=cfg["locomotion"],
                                vla_ckpt=vla_ckpt, vla_device=vla_device)
                r.update({"family": family, "layout_seed": seed,
                          "layout_name": layout.name, "stack": stack})
                stack_res.append(r)
                all_results.append(r)
                jsonl.write(json.dumps(r) + "\n")
                jsonl.flush()
                total_missions += 1
                extra = (f" alloc={r.get('alloc_winner')} corr={r.get('alloc_correct')}"
                         if klass == "D" else "")
                print(f"  [{stack:8s} {idx:02d}] {klass} seed={mseed} "
                      f"spot{spot} -> {r['outcome']} pad={r['object_on_pad']} "
                      f"fell={r['any_fell']} ok={r['success']} "
                      f"steps={r['steps']}{extra} ({r['time_s']}s) "
                      f"[{total_missions} done, {time.time()-t0:.0f}s]", flush=True)
            summ = summarize(stack_res)
            rate = summ["ac_success"] / max(1, summ["ac_total"])
            row = {"family": family, "layout_seed": seed,
                   "layout_name": layout.name, "stack": stack,
                   "bev": bev_path, "n_spots": len(layout.object_spots),
                   **summ}
            per_layout.append(row)
            if rate < fail_threshold or summ["n_falls"] > 0 or summ["plan_failures"] > 0:
                findings.append(row)
            print(f"  -> {stack}: A-C {summ['ac_success']}/{summ['ac_total']} "
                  f"D {summ['d_correct']}/{summ['d_total']} falls={summ['n_falls']} "
                  f"plan_fail={summ['plan_failures']}", flush=True)

    jsonl.close()
    aggregate = _aggregate(family, all_results, per_layout, stacks,
                           time.time() - t0)
    aggregate["findings"] = findings
    with open(os.path.join(out_dir, "per_layout.json"), "w") as f:
        json.dump(per_layout, f, indent=2)
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(aggregate, f, indent=2)
    _print_tables(aggregate, per_layout, out_dir)
    return aggregate


def _aggregate(family: str, results: List[dict], per_layout: List[dict],
               stacks: Sequence[str], elapsed: float) -> dict:
    """Per-stack roll-up across all sampled layouts of the family."""
    by_stack: Dict[str, dict] = {}
    for stack in stacks:
        sr = [r for r in results if r["stack"] == stack]
        by_stack[stack] = summarize(sr)
        by_stack[stack]["n_layouts"] = len({r["layout_seed"] for r in sr})
    return {
        "family": family,
        "stacks": list(stacks),
        "n_layouts": len({r["layout_seed"] for r in results}),
        "n_missions": len(results),
        "total_time_s": round(elapsed, 1),
        "by_stack": by_stack,
    }


def _print_tables(agg: dict, per_layout: List[dict], out_dir: str) -> None:
    """Print the per-layout and aggregate generalization tables."""
    print("\n" + "=" * 84)
    print(f"GENERALIZATION EVAL — family={agg['family']}  "
          f"layouts={agg['n_layouts']}  missions={agg['n_missions']}  "
          f"time={agg['total_time_s']}s")
    print("-" * 84)
    print(f"  {'seed':>4} {'stack':<9}{'A-C':>8}{'D':>6}{'falls':>7}"
          f"{'planfail':>9}{'steps p90':>11}")
    for row in per_layout:
        print(f"  {row['layout_seed']:>4} {row['stack']:<9}"
              f"{row['ac_success']:>4}/{row['ac_total']:<3}"
              f"{row['d_correct']:>3}/{row['d_total']:<2}"
              f"{row['n_falls']:>7}{row['plan_failures']:>9}"
              f"{row['steps']['p90']:>11}")
    print("-" * 84)
    for stack, s in agg["by_stack"].items():
        pc = " ".join(f"{k}:{v['ok']}/{v['n']}"
                      for k, v in sorted(s["per_class"].items()))
        print(f"  AGG {stack:<9}: A-C {s['ac_success']}/{s['ac_total']}  "
              f"[{pc}]  D {s['d_correct']}/{s['d_total']}  falls={s['n_falls']}  "
              f"plan_fail={s['plan_failures']}  "
              f"steps(med/p90/max)={s['steps']['median']}/{s['steps']['p90']}/"
              f"{s['steps']['max']}")
    if agg.get("findings"):
        print("-" * 84)
        print(f"  FINDINGS ({len(agg['findings'])} layout/stack rows below target "
              f"or with falls/plan-failures):")
        for row in agg["findings"]:
            print(f"    seed={row['layout_seed']} {row['stack']}: "
                  f"A-C {row['ac_success']}/{row['ac_total']} "
                  f"falls={row['n_falls']} plan_fail={row['plan_failures']} "
                  f"bev={row['bev']}")
    print(f"  artifacts: {out_dir}/")
    print("=" * 84, flush=True)


def main(argv: Optional[List[str]] = None) -> None:
    ap = argparse.ArgumentParser(description="Randomized-layout generalization eval")
    ap.add_argument("--layout-family", choices=("hero", "rooms"), default="rooms")
    ap.add_argument("--layout-seeds", type=int, default=8,
                    help="number of randomized layouts to sample")
    ap.add_argument("--mission-seeds", type=int, default=3,
                    help="seeds per A/B/C class per layout (3*seeds + 3 D)")
    ap.add_argument("--base-seed", type=int, default=0,
                    help="layout RNG base; layout k uses base+k")
    ap.add_argument("--only-layout-seed", type=int, default=None,
                    help="run exactly this one layout seed (deterministic single-"
                         "seed repro; overrides --layout-seeds/--base-seed range)")
    ap.add_argument("--stacks", default="both",
                    help="'both' (default), 'learned', 'baseline', or a comma list")
    ap.add_argument("--out", type=str, default=None,
                    help="artifact dir (default ops/gen_eval/<family>)")
    ap.add_argument("--max-steps", type=int, default=9000)
    ap.add_argument("--ckpt", type=str, default=None,
                    help="GroundedNav checkpoint for the learned stack (default F5)")
    ap.add_argument("--device", type=str, default=None,
                    help="Torch device for the VLA policy (cuda|cpu; default auto)")
    args = ap.parse_args(argv)

    stacks = (list(STACKS) if args.stacks == "both"
              else [s.strip() for s in args.stacks.split(",") if s.strip()])
    for s in stacks:
        if s not in STACKS:
            ap.error(f"unknown stack {s!r}; choose from {list(STACKS)} or 'both'")
    out = args.out or os.path.join(_DEFAULT_OUT, args.layout_family)
    run_gen_eval(args.layout_family, args.layout_seeds, args.mission_seeds,
                 out, args.max_steps, stacks, base_seed=args.base_seed,
                 vla_ckpt=args.ckpt, vla_device=args.device,
                 only_layout_seed=args.only_layout_seed)


if __name__ == "__main__":
    main()
