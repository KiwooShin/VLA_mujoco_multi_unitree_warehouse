"""mission_eval.py — Batch evaluation of end-to-end collaborative fetch missions.

Runs seeded missions across four scenario classes and reports success:

* **A** — addressed fetch, object visible to the owner at start (no peer traffic);
* **B** — addressed fetch, object visible only to a *different* robot (peer query);
* **C** — addressed fetch, object hidden from ALL (full delegated search);
* **D** — fleet-addressed ("someone bring me ..."), scoring whether the
  path-length allocator picked the objectively shortest-path robot.

A mission is a success (A-C) iff the requested object ends on the delivery pad,
the owner sent ``TASK_COMPLETE``, and no robot fell. A D mission is correct iff
the allocator's winner matches an independent A\\* argmin over every robot.

Per-mission + summary JSON land in ``eval/missions/`` and a table is printed.

Usage
-----
PYTHONPATH=. MUJOCO_GL=egl python code/fleet/mission_eval.py --seeds 4 --out eval/missions
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from code.comms.messages import ObjectQuery
from code.fleet.allocator import RobotPose, planned_path_length
from code.fleet.mission import MissionRunner
from code.fleet.search import (SEARCH_REGIONS, free_centroid, region_centroid,
                               region_name_for_xy, search_regions_for_layout)
from code.fleet.visibility import VisibilityConfig, is_object_visible
from code.sim.arena_build import COLORS
from code.sim.teacher import WBCTeacher
from code.warehouse.layout import (CALLSIGNS, WarehouseLayout, hero_layout,
                                   rooms_layout)

_DEFAULT_OUT = str(_REPO / "eval" / "missions")
_LAYOUTS = {"hero": hero_layout, "rooms": rooms_layout}
Point = Tuple[float, float]
_CMAP = dict(COLORS)
_OWNER = "Alpha"
# Filler (color, shape, size) for the non-target spots — none is a red cube.
_FILLER: Tuple[Tuple[str, str, float], ...] = (
    ("orange", "cube", 0.24), ("blue", "cylinder", 0.22), ("green", "ball", 0.24),
    ("yellow", "cone", 0.26), ("purple", "cube", 0.24), ("cyan", "cylinder", 0.22),
    ("blue", "ball", 0.24),
)


def _walls(layout: WarehouseLayout) -> List[dict]:
    return [dataclasses.asdict(w) for w in layout.walls]


def _classify_spots(layout: WarehouseLayout,
                    cfg: VisibilityConfig) -> Dict[str, List[int]]:
    """Group object spots by who can see them from the robots' spawn poses."""
    walls = _walls(layout)
    poses = {cs: layout.spawn_poses[cs] for cs in CALLSIGNS}
    groups: Dict[str, List[int]] = {"A": [], "B": [], "C": []}
    for i, (x, y) in enumerate(layout.object_spots):
        owner_sees = is_object_visible(poses[_OWNER][:2], poses[_OWNER][2], 0.74,
                                       (x, y), walls, cfg=cfg)
        peer_sees = any(is_object_visible(poses[cs][:2], poses[cs][2], 0.74,
                                          (x, y), walls, cfg=cfg)
                        for cs in CALLSIGNS if cs != _OWNER)
        if owner_sees:
            groups["A"].append(i)
        elif peer_sees:
            groups["B"].append(i)
        else:
            groups["C"].append(i)
    return groups


def build_objects(layout: WarehouseLayout, target_spot: int,
                  seed: int) -> List[dict]:
    """Place a red cube at ``target_spot`` and distinct fillers elsewhere."""
    objs: List[dict] = []
    fi = seed % len(_FILLER)
    for i, (x, y) in enumerate(layout.object_spots):
        if i == target_spot:
            c, s, sz = "red", "cube", 0.24
        else:
            c, s, sz = _FILLER[(fi + i) % len(_FILLER)]
        objs.append({"color_name": c, "color_rgb": _CMAP[c], "shape_name": s,
                     "size": float(sz), "x": float(x), "y": float(y)})
    return objs


def _independent_allocation(layout: WarehouseLayout, cfg: dict,
                            query: ObjectQuery, vis: VisibilityConfig,
                            regions: Tuple[str, ...]
                            ) -> Tuple[str, Dict[str, float]]:
    """Hand-compute the path-shortest robot (ground truth for class D)."""
    walls = cfg["walls"]
    rooms = tuple(layout.rooms)
    poses = {cs: RobotPose(layout.spawn_poses[cs][:2], layout.spawn_poses[cs][2], 0.74)
             for cs in CALLSIGNS}
    obj_xy: Optional[Point] = None
    for obj in cfg["objects"]:
        if query.matches(obj):
            oxy = (float(obj["x"]), float(obj["y"]))
            if any(is_object_visible(p.xy, p.yaw, p.base_height, oxy, walls, cfg=vis)
                   for p in poses.values()):
                obj_xy = oxy
            break
    costs: Dict[str, float] = {}
    for cs in CALLSIGNS:
        if obj_xy is not None:
            target = obj_xy
        else:
            best_r = min(regions, key=lambda r: (
                (region_centroid(r, cfg["hall_x"], cfg["hall_y"], rooms)[0] - poses[cs].xy[0]) ** 2
                + (region_centroid(r, cfg["hall_x"], cfg["hall_y"], rooms)[1] - poses[cs].xy[1]) ** 2))
            target = free_centroid(cfg, best_r, rooms=rooms)
        costs[cs] = planned_path_length(cfg, poses[cs].xy, target)
    winner = min(CALLSIGNS, key=lambda cs: costs[cs])
    return winner, costs


def run_mission(klass: str, seed: int, layout: WarehouseLayout, spot: int,
                teachers: Dict[str, WBCTeacher], max_steps: int,
                perception_mode: str = "oracle") -> dict:
    """Run one mission of a class and return its metrics dict."""
    objs = build_objects(layout, spot, seed)
    vis = VisibilityConfig()
    t0 = time.time()
    mr = MissionRunner(layout=layout, objects=objs, teachers=teachers,
                       use_gpu=True, search_deadline_steps=max_steps,
                       perception_mode=perception_mode)
    if klass == "D":
        gt_winner, gt_costs = _independent_allocation(
            layout, mr.scene_cfg, ObjectQuery("red", "cube"), vis,
            search_regions_for_layout(layout))
        mr.submit("someone bring me the red cube")
    else:
        gt_winner, gt_costs = "", {}
        mr.submit(f"{_OWNER}, fetch the red cube to the delivery pad")
    res = mr.run(max_steps)
    elapsed = time.time() - t0

    found = [m for m in mr.bus.transcript if m.performative.name == "REPORT_FOUND"]
    alloc_ok: Optional[bool] = None
    if klass == "D":
        alloc_ok = (mr.allocation is not None
                    and mr.allocation.winner == gt_winner)
    success = res.object_on_pad and res.task_complete_sent and not res.any_fell
    out = {
        "class": klass, "seed": seed, "target_spot": spot,
        "target_xy": list(layout.object_spots[spot]),
        "outcome": res.outcome, "owner": res.owner, "steps": res.steps,
        "object_on_pad": res.object_on_pad, "task_complete": res.task_complete_sent,
        "any_fell": res.any_fell, "success": bool(success),
        "finder": found[0].sender if found else None,
        "found_step": found[0].t_step if found else None,
        "perception_mode": perception_mode,
        "n_confirmations": len(mr.confirmations),
        "time_s": round(elapsed, 1),
    }
    if klass == "D":
        out.update({
            "alloc_winner": mr.allocation.winner if mr.allocation else None,
            "alloc_reason": mr.allocation.reason if mr.allocation else None,
            "gt_winner": gt_winner,
            "gt_costs": {k: round(v, 2) for k, v in gt_costs.items()},
            "alloc_correct": bool(alloc_ok),
        })
    mr.close()
    return out


def _plan(seeds: int) -> List[Tuple[str, int, int]]:
    """Build the (class, seed, spot) plan from the classified hero spots."""
    layout = hero_layout()
    groups = _classify_spots(layout, VisibilityConfig())
    plan: List[Tuple[str, int, int]] = []
    for k in range(seeds):
        for klass in ("A", "B", "C"):
            cands = groups[klass] or groups["C"]
            plan.append((klass, k, cands[k % len(cands)]))
    # Three fleet-addressed allocator checks, one per hidden/visible mix.
    dspots = (groups["A"] + groups["B"] + groups["C"])
    for j in range(3):
        plan.append(("D", 100 + j, dspots[j % len(dspots)]))
    return plan


def _rooms_plan(seeds: int) -> List[Tuple[str, int, int]]:
    """Build the (class, seed, spot) plan for the multi-room layout (F6).

    Every rooms spot is hidden from the fleet's bays (all class C), so each
    searchable-room spot (storage A/B + back room; the fleet's own loading room
    is not a search target) is one class-C room-to-room search mission. Three
    fleet-addressed (D) allocations round it out.
    """
    layout = rooms_layout()
    regions = search_regions_for_layout(layout)
    searchable = [i for i, (x, y) in enumerate(layout.object_spots)
                  if region_name_for_xy(layout, (x, y)) in regions]
    plan: List[Tuple[str, int, int]] = [("C", j, spot)
                                        for j, spot in enumerate(searchable)]
    for j in range(3):
        plan.append(("D", 100 + j, searchable[j % len(searchable)]))
    return plan


def run_eval(seeds: int, out_dir: str, max_steps: int,
             perception_mode: str = "oracle", layout_name: str = "hero") -> dict:
    """Run the full mission suite, write JSON, print the table."""
    os.makedirs(out_dir, exist_ok=True)
    layout = _LAYOUTS.get(layout_name, hero_layout)()
    teachers = {cs: WBCTeacher(use_gpu=True) for cs in CALLSIGNS}
    plan = _rooms_plan(seeds) if layout.rooms else _plan(seeds)

    results: List[dict] = []
    t0 = time.time()
    for idx, (klass, seed, spot) in enumerate(plan):
        r = run_mission(klass, seed, layout, spot, teachers, max_steps,
                        perception_mode=perception_mode)
        results.append(r)
        with open(os.path.join(out_dir, f"mission_{idx:02d}_{klass}.json"), "w") as f:
            json.dump(r, f, indent=2)
        _print_mission(idx, r)

    summary = _summarize(results, time.time() - t0)
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    _print_summary(summary, out_dir)
    return summary


def _summarize(results: List[dict], elapsed: float) -> dict:
    ac = [r for r in results if r["class"] in ("A", "B", "C")]
    d = [r for r in results if r["class"] == "D"]
    per_class: Dict[str, Dict[str, int]] = {}
    for r in ac:
        pc = per_class.setdefault(r["class"], {"n": 0, "ok": 0})
        pc["n"] += 1
        pc["ok"] += int(r["success"])
    return {
        "n_missions": len(results),
        "perception_mode": results[0].get("perception_mode", "oracle") if results else "oracle",
        "ac_success": sum(r["success"] for r in ac),
        "ac_total": len(ac),
        "per_class": per_class,
        "d_correct": sum(r.get("alloc_correct", False) for r in d),
        "d_total": len(d),
        "n_falls": sum(r["any_fell"] for r in results),
        "n_confirmations": sum(r.get("n_confirmations", 0) for r in results),
        "total_time_s": round(elapsed, 1),
    }


def _print_mission(idx: int, r: dict) -> None:
    extra = ""
    if r["class"] == "D":
        extra = (f" alloc={r['alloc_winner']}({r['alloc_reason']}) "
                 f"gt={r['gt_winner']} correct={r['alloc_correct']}")
    print(f"[{idx:02d}] {r['class']} seed={r['seed']} spot{r['target_spot']}{tuple(r['target_xy'])} "
          f"-> {r['outcome']} owner={r['owner']} finder={r['finder']} "
          f"steps={r['steps']} pad={r['object_on_pad']} fell={r['any_fell']} "
          f"success={r['success']}{extra} ({r['time_s']}s)", flush=True)


def _print_summary(s: dict, out_dir: str) -> None:
    print("\n" + "=" * 72)
    print(f"MISSION EVAL  (perception_mode={s.get('perception_mode', 'oracle')})")
    print("-" * 72)
    for klass, pc in sorted(s["per_class"].items()):
        print(f"  class {klass}: {pc['ok']}/{pc['n']} success")
    print(f"  A-C combined: {s['ac_success']}/{s['ac_total']} "
          f"(target >= 8/10)")
    print(f"  D allocations correct: {s['d_correct']}/{s['d_total']} (target 3/3)")
    print(f"  falls: {s['n_falls']}   detector confirmations: "
          f"{s.get('n_confirmations', 0)}   time: {s['total_time_s']}s")
    print(f"  artifacts: {out_dir}/")
    print("=" * 72, flush=True)


def main(argv: Optional[List[str]] = None) -> None:
    ap = argparse.ArgumentParser(description="Collaborative fetch mission eval")
    ap.add_argument("--seeds", type=int, default=4,
                    help="seeds per A/B/C class (3*seeds + 3 D missions)")
    ap.add_argument("--out", type=str, default=_DEFAULT_OUT)
    ap.add_argument("--max-steps", type=int, default=9000)
    ap.add_argument("--perception", choices=("oracle", "groundnet"),
                    default="oracle", help="visibility backend for can_see")
    ap.add_argument("--layout", choices=tuple(_LAYOUTS), default="hero",
                    help="hero thirds or the multi-room rooms_layout (F6)")
    args = ap.parse_args(argv)
    run_eval(args.seeds, args.out, args.max_steps,
             perception_mode=args.perception, layout_name=args.layout)


if __name__ == "__main__":
    main()
