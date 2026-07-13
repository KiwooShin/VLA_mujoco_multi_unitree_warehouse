"""perception_eval.py — GROUND_NET detector vs. the geometric oracle.

The headline Cycle-2a artifact: characterise the REAL learned detector against
the deterministic visibility oracle over the hero layout and N seeded red-cube
placements, rendering the 480x360 26-degree grounding camera from the warehouse
scene (the same walls / objects / light floor the mission's shared viz model
renders) at cameras ringed around each placement.

Two regimes are reported (JSON + printed tables):

* Frames where the ORACLE says VISIBLE (object in clear line of sight, in range
  and inside the grounding cam's FOV): the detector's **detection rate**,
  **false-negative rate** and **world-xy error** stats — how often, and how
  accurately, the learned detector confirms a genuine sighting.
* Frames where the ORACLE says HIDDEN (a wall occludes the object): the
  detector's **false-positive rate** — the critical check that *walls actually
  defeat the detector* (it must not hallucinate an occluded object).

A "detection" here is the confirmer's own criterion: raw peak confidence
``>= tau`` AND the decoded world-xy within ``XY_HIT_M`` of the true object (this
rejects the domain-shift spurious peaks documented in perception_bridge). All
thresholds are reported.

Usage
-----
PYTHONPATH=. MUJOCO_GL=egl python code/fleet/perception_eval.py --seeds 4 \\
    --out eval/perception [--mission-seeds 4]
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import mujoco
import numpy as np

from code.fleet.perception_bridge import (CONFIRM_RANGE_M, CONFIRM_TAU,
                                           GROUNDING_HALF_FOV_DEG,
                                           GroundingCamRenderer,
                                           detection_world_xy,
                                           load_shared_detector)
from code.fleet.viz import _warehouse_base_spec
from code.fleet.visibility import (_DEFAULT_OBJ_RADIUS, VisibilityConfig,
                                    is_object_visible, line_of_sight_clear)
from code.sim.arena_build import COLORS
from code.warehouse.arena import warehouse_scene_cfg
from code.warehouse.layout import hero_layout

_DEFAULT_OUT = str(_REPO / "eval" / "perception")
_CMAP = dict(COLORS)
Point = Tuple[float, float]

# Camera-ring sampling around each placement: cameras face the object at these
# ranges (m) and bearings (deg) about the object->camera direction.
_RING_DISTS: Tuple[float, ...] = (2.5, 3.5, 4.5, 5.5)
_RING_ANGLES_DEG: Tuple[float, ...] = tuple(range(0, 360, 30))  # 12 angles
_BASE_H: float = 0.744               # pelvis/camera-mount height (m)
_XY_HIT_M: float = 1.0               # decoded xy within this of truth == a real hit
# Minimum camera-to-wall clearance for a *physically valid* ring pose (m). The
# naive ring places some cameras inside/against a shelf — poses no robot could
# occupy (nav inflation is 0.40 m). MuJoCo then near-clips the shelf, so its
# render shows an object the 2-D oracle (correctly) calls occluded, manufacturing
# a "through-wall FP". Scoring only reachable poses (camera >= this from every
# tall wall, ~a robot body radius) keeps the wall-occluded FP metric honest.
_MIN_CAM_WALL_CLEARANCE_M: float = 0.35
_FILLER = (("orange", "cube", 0.24), ("blue", "cylinder", 0.22),
           ("green", "ball", 0.24), ("yellow", "cone", 0.26),
           ("purple", "cube", 0.24), ("cyan", "cylinder", 0.22),
           ("blue", "ball", 0.24))


def _build_objects(spot: int) -> List[dict]:
    """Red cube at ``spot`` + distinct fillers elsewhere (hero layout)."""
    objs: List[dict] = []
    fi = 0
    for i, (x, y) in enumerate(hero_layout().object_spots):
        if i == spot:
            c, s, sz = "red", "cube", 0.24
        else:
            c, s, sz = _FILLER[fi % len(_FILLER)]
            fi += 1
        objs.append({"color_name": c, "color_rgb": _CMAP[c], "shape_name": s,
                     "size": float(sz), "x": float(x), "y": float(y)})
    return objs


def _pct(xs: List[float], p: float) -> float:
    """The p-quantile (0..1) of xs, or nan if empty."""
    if not xs:
        return float("nan")
    ys = sorted(xs)
    return ys[min(len(ys) - 1, max(0, int(round(p * (len(ys) - 1)))))]


def _cam_wall_clearance(cam_xy: Point, walls) -> float:
    """Min distance from ``cam_xy`` to any tall (occluding) wall footprint (m)."""
    best = float("inf")
    for w in walls:
        if float(w.get("height", 2.5)) <= 0.25:
            continue  # short props never occlude / never block a robot's stance
        cx, cy, yaw = float(w["cx"]), float(w["cy"]), float(w.get("yaw", 0.0))
        c, s = math.cos(yaw), math.sin(yaw)
        lx = (cam_xy[0] - cx) * c + (cam_xy[1] - cy) * s
        ly = -(cam_xy[0] - cx) * s + (cam_xy[1] - cy) * c
        dx = max(abs(lx) - float(w["half_x"]), 0.0)
        dy = max(abs(ly) - float(w["half_y"]), 0.0)
        best = min(best, math.hypot(dx, dy))
    return best


def _camera_pose_valid(cam_xy: Point, walls, clearance: float) -> bool:
    """Whether a ring camera pose is one a robot could actually occupy.

    Rejects poses inside/against a tall wall (< ``clearance`` from its footprint):
    those are physically unreachable and only produce near-clip see-through
    artifacts, not real sightings.
    """
    return _cam_wall_clearance(cam_xy, walls) >= clearance


def _frame_record(det, renderer, data, cam_xy: Point, yaw: float, obj_xy: Point,
                  obj_z: float, walls, vis_cfg, tau: float,
                  obj_radius: float) -> dict:
    """Render one camera pose and score the detector against the oracle.

    ``obj_radius`` is the object planar half-extent sampled by the LOS oracle
    (0 -> the historical centre-only segment; ~0.12 m -> the partial-visibility
    sampling). It is the single knob that produces the relabeling before/after
    table: a partially visible object flips from oracle-HIDDEN to oracle-VISIBLE.
    """
    oracle_vis = is_object_visible(cam_xy, yaw, _BASE_H, obj_xy, walls,
                                   obj_z=obj_z, obj_radius=obj_radius, cfg=vis_cfg)
    los = line_of_sight_clear(cam_xy, obj_xy, walls,
                              head_z=vis_cfg.head_z(_BASE_H), obj_z=obj_z,
                              obj_radius=obj_radius)
    pelvis = (cam_xy[0], cam_xy[1], _BASE_H)
    rgb, depth, _ = renderer.render(data, pelvis, yaw)
    out = det.infer(rgb, depth, class_name="cube", color_name="red",
                    cam_type="grounding", conf_thresh=0.0)
    conf = float(out["confidence"])
    det_xy = detection_world_xy(cam_xy, yaw, float(out["dist_m"]),
                                float(out["bearing_deg"]))
    xy_err = math.hypot(det_xy[0] - obj_xy[0], det_xy[1] - obj_xy[1])
    hit = bool(conf >= tau and xy_err <= _XY_HIT_M)   # real (on-object) detection
    return dict(oracle_vis=bool(oracle_vis), los_clear=bool(los), conf=conf,
                xy_err=float(xy_err), hit=hit,
                present_raw=bool(conf >= tau))


def eval_frames(seeds: int, tau: float, verbose: bool = True,
                ckpt: Optional[str] = None,
                obj_radius: float = _DEFAULT_OBJ_RADIUS,
                min_cam_clearance: float = _MIN_CAM_WALL_CLEARANCE_M) -> dict:
    """Render + score every sampled frame across ``seeds`` placements.

    Args:
        seeds: Number of red-cube placements (each rings ~40 camera frames).
        tau: Confidence floor for a detection.
        verbose: Print the visible/hidden summary tables.
        ckpt: Explicit GROUND_NET checkpoint to evaluate; when ``None`` the
            fleet's default resolution order is used (see
            :func:`code.fleet.perception_bridge.resolve_ckpt_path`).
        obj_radius: Object planar half-extent the oracle samples for LOS (m).
            ``0.0`` reproduces the historical centre-only labelling (the "before"
            of the partial-visibility relabeling); the default matches the fleet.
        min_cam_clearance: Reject ring camera poses closer than this to any tall
            wall (m). ``0.0`` reproduces the historical (unfiltered) ring that
            scored physically-invalid, near-clip camera poses.
    """
    import dataclasses
    layout = hero_layout()
    walls = [dataclasses.asdict(w) for w in layout.walls]
    vis_cfg = VisibilityConfig()
    det = load_shared_detector(ckpt)
    if det is None:
        raise RuntimeError("GROUND_NET checkpoint unavailable; cannot run perception eval")

    records: List[dict] = []
    n_spots = len(layout.object_spots)
    for s in range(seeds):
        spot = s % n_spots
        obj_xy = tuple(float(v) for v in layout.object_spots[spot])
        objs = _build_objects(spot)
        obj_z = max(0.12, 0.24 / 2.0)
        scene_cfg = warehouse_scene_cfg(layout, objects=objs,
                                        rng=np.random.default_rng(s))
        model = _warehouse_base_spec(scene_cfg).compile()
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        renderer = GroundingCamRenderer(model)
        for d in _RING_DISTS:
            for ang in _RING_ANGLES_DEG:
                a = math.radians(ang)
                cam_xy = (obj_xy[0] + d * math.cos(a), obj_xy[1] + d * math.sin(a))
                if not (layout.hall_x / -2 + 0.4 < cam_xy[0] < layout.hall_x / 2 - 0.4
                        and layout.hall_y / -2 + 0.4 < cam_xy[1] < layout.hall_y / 2 - 0.4):
                    continue  # camera outside the hall
                if not _camera_pose_valid(cam_xy, walls, min_cam_clearance):
                    continue  # camera inside/against a shelf: unreachable pose
                yaw = math.atan2(obj_xy[1] - cam_xy[1], obj_xy[0] - cam_xy[0])
                rec = _frame_record(det, renderer, data, cam_xy, yaw, obj_xy,
                                    obj_z, walls, vis_cfg, tau, obj_radius)
                rec.update(seed=s, spot=spot, dist=float(d), angle=int(ang))
                records.append(rec)
        renderer.close()

    summary = _summarize_frames(records, tau)
    if verbose:
        _print_frames(summary)
    return {"tau": tau, "xy_hit_m": _XY_HIT_M, "n_frames": len(records),
            "summary": summary, "records": records}


def _summarize_frames(records: List[dict], tau: float) -> dict:
    """Fold per-frame records into the visible / hidden tables."""
    vis = [r for r in records if r["oracle_vis"]]
    hid = [r for r in records if not r["oracle_vis"]]
    hid_wall = [r for r in hid if not r["los_clear"]]   # occluded by a wall

    vis_hits = [r for r in vis if r["hit"]]
    xy_errs = [r["xy_err"] for r in vis_hits]
    det_rate = len(vis_hits) / len(vis) if vis else float("nan")

    fp = [r for r in hid if r["hit"]]                   # hallucinated occluded obj
    fp_wall = [r for r in hid_wall if r["hit"]]
    return {
        "visible": {
            "n": len(vis),
            "detections": len(vis_hits),
            "detection_rate": det_rate,
            "false_negative_rate": (1.0 - det_rate) if vis else float("nan"),
            "xy_err_mean": float(np.mean(xy_errs)) if xy_errs else float("nan"),
            "xy_err_p50": _pct(xy_errs, 0.50),
            "xy_err_p90": _pct(xy_errs, 0.90),
            "xy_err_max": max(xy_errs) if xy_errs else float("nan"),
            "conf_mean": float(np.mean([r["conf"] for r in vis])) if vis else float("nan"),
            "conf_p90": _pct([r["conf"] for r in vis], 0.90),
        },
        "hidden": {
            "n": len(hid),
            "n_wall_occluded": len(hid_wall),
            "false_positives": len(fp),
            "false_positive_rate": len(fp) / len(hid) if hid else float("nan"),
            "wall_occluded_fp": len(fp_wall),
            "wall_occluded_fp_rate": (len(fp_wall) / len(hid_wall)
                                      if hid_wall else float("nan")),
            "conf_mean": float(np.mean([r["conf"] for r in hid])) if hid else float("nan"),
            "conf_p90": _pct([r["conf"] for r in hid], 0.90),
        },
    }


def _fmt(x: float) -> str:
    return "  n/a" if x != x else f"{x:.3f}"


def _print_frames(summary: dict) -> None:
    v, h = summary["visible"], summary["hidden"]
    print("\n" + "=" * 68)
    print("PERCEPTION EVAL — GROUND_NET detector vs. geometric oracle")
    print("-" * 68)
    print("  ORACLE = VISIBLE (object in clear LOS + range + grounding FOV)")
    print(f"    frames                : {v['n']}")
    print(f"    detections (on-object): {v['detections']}")
    print(f"    detection rate        : {_fmt(v['detection_rate'])}")
    print(f"    false-negative rate   : {_fmt(v['false_negative_rate'])}")
    print(f"    xy err mean/p50/p90/max: {_fmt(v['xy_err_mean'])} / "
          f"{_fmt(v['xy_err_p50'])} / {_fmt(v['xy_err_p90'])} / {_fmt(v['xy_err_max'])} m")
    print(f"    conf mean/p90         : {_fmt(v['conf_mean'])} / {_fmt(v['conf_p90'])}")
    print("  ORACLE = HIDDEN (walls must defeat the detector)")
    print(f"    frames                : {h['n']}  (wall-occluded: {h['n_wall_occluded']})")
    print(f"    false-positive rate   : {_fmt(h['false_positive_rate'])} "
          f"({h['false_positives']} hallucinated)")
    print(f"    wall-occluded FP rate : {_fmt(h['wall_occluded_fp_rate'])} "
          f"({h['wall_occluded_fp']} through-wall)")
    print(f"    conf mean/p90         : {_fmt(h['conf_mean'])} / {_fmt(h['conf_p90'])}")
    print("=" * 68, flush=True)


def compare_missions(mission_seeds: int, out_dir: str, max_steps: int) -> dict:
    """Run mission_eval in oracle + groundnet modes and tabulate the comparison."""
    from code.fleet.mission_eval import run_eval
    print(f"\n[perception-eval] running mission_eval oracle vs groundnet "
          f"(--seeds {mission_seeds}) ...", flush=True)
    oracle = run_eval(mission_seeds, os.path.join(out_dir, "missions_oracle"),
                      max_steps, perception_mode="oracle")
    groundnet = run_eval(mission_seeds, os.path.join(out_dir, "missions_groundnet"),
                         max_steps, perception_mode="groundnet")
    _print_mission_compare(oracle, groundnet)
    return {"oracle": oracle, "groundnet": groundnet}


def _print_mission_compare(oracle: dict, groundnet: dict) -> None:
    print("\n" + "=" * 68)
    print("MISSION EVAL — oracle vs groundnet (same protocol/message flow)")
    print("-" * 68)
    print(f"  {'metric':<26}{'oracle':>12}{'groundnet':>14}")
    for key, label in (("ac_success", "A/B/C fetch success"),
                       ("d_correct", "D allocations correct"),
                       ("n_falls", "robot falls")):
        o = oracle.get(key); g = groundnet.get(key)
        odenom = f"/{oracle.get('ac_total')}" if key == "ac_success" else (
            f"/{oracle.get('d_total')}" if key == "d_correct" else "")
        print(f"  {label:<26}{str(o) + odenom:>12}{str(g) + odenom:>14}")
    print(f"  {'groundnet confirmations':<26}{'-':>12}"
          f"{groundnet.get('n_confirmations', 0):>14}")
    print("=" * 68, flush=True)


def main(argv: Optional[List[str]] = None) -> None:
    ap = argparse.ArgumentParser(description="GROUND_NET detector vs oracle eval")
    ap.add_argument("--seeds", type=int, default=4,
                    help="red-cube placements (each rings ~40 camera frames)")
    ap.add_argument("--tau", type=float, default=CONFIRM_TAU,
                    help=f"confidence floor for a detection (default {CONFIRM_TAU})")
    ap.add_argument("--out", type=str, default=_DEFAULT_OUT)
    ap.add_argument("--ckpt", type=str, default=None,
                    help="GROUND_NET checkpoint to evaluate (default: fleet "
                         "resolution order — env var > warehouse fine-tune > baseline)")
    ap.add_argument("--mission-seeds", type=int, default=0,
                    help="if >0, also run mission_eval oracle vs groundnet")
    ap.add_argument("--max-steps", type=int, default=9000)
    ap.add_argument("--obj-radius", type=float, default=_DEFAULT_OBJ_RADIUS,
                    help="object planar half-extent the oracle samples for LOS "
                         "(0 = historical centre-only labelling)")
    ap.add_argument("--min-cam-clearance", type=float,
                    default=_MIN_CAM_WALL_CLEARANCE_M,
                    help="reject ring camera poses closer than this to a tall wall "
                         "(0 = historical unfiltered ring)")
    args = ap.parse_args(argv)

    os.makedirs(args.out, exist_ok=True)
    from code.fleet.perception_bridge import resolve_ckpt_path
    ckpt = args.ckpt or resolve_ckpt_path()
    print(f"[perception-eval] tau={args.tau} range<={CONFIRM_RANGE_M}m "
          f"grounding_half_fov={GROUNDING_HALF_FOV_DEG}deg obj_radius={args.obj_radius}m "
          f"min_cam_clearance={args.min_cam_clearance}m ckpt={ckpt!r}", flush=True)
    result = eval_frames(args.seeds, args.tau, ckpt=args.ckpt,
                         obj_radius=args.obj_radius,
                         min_cam_clearance=args.min_cam_clearance)
    result["ckpt"] = ckpt
    result["obj_radius"] = args.obj_radius
    result["min_cam_clearance"] = args.min_cam_clearance
    # Drop the bulky per-frame records from the on-disk JSON summary.
    disk = {k: v for k, v in result.items() if k != "records"}
    if args.mission_seeds > 0:
        disk["missions"] = compare_missions(args.mission_seeds, args.out,
                                            args.max_steps)
    with open(os.path.join(args.out, "perception_eval.json"), "w") as f:
        json.dump(disk, f, indent=2)
    print(f"[perception-eval] wrote {args.out}/perception_eval.json", flush=True)


if __name__ == "__main__":
    main()
