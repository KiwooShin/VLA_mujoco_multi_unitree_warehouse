"""gen_warehouse_dart.py — Warehouse-domain DART dataset generator (F5, phase-1).

Generates teacher-driven DART episodes in the warehouse arena (A*-planned routes
+ baseline-style primitive segments), recording the EXACT baseline DART parquet
schema (loadable by ``code.data.dataset_phase.PhaseParquetDataset``) PLUS a
per-episode ego RGB mp4 (GPU-rendered) for visual-domain fine-tuning
(``code.data.dataset.ParquetDataset``).

Output layout (matches the baseline DART generator + videos/):
  <out>/
    data/chunk-000/episode_NNNNNN.parquet
    videos/episode_NNNNNN_ego.mp4
    meta/{info.json, stats.json, episodes.jsonl, tasks.jsonl, manifest.jsonl}
    meta/ep_meta/episode_NNNNNN.json      # per-episode resume sidecar

Determinism / idempotency:
  * Every episode is a pure function of (--seed, episode_index): scene, spawn,
    route and DART noise are all derived from it (noise rng = derive_rng(seed+10000,
    ep_idx), per-episode — unlike the baseline's single running stream — so
    skipping/resuming episodes never changes any already-written episode).
  * An episode is "done" iff its parquet (and mp4, when rendering) exists; done
    episodes are skipped. Safe to re-run / resume after a crash without loss.

Chunked execution (verify GPU each chunk): pass --chunk-size K to generate at
most K new episodes per invocation (the next K undone indices), then re-assemble
meta and exit; call repeatedly to fill --episodes in K-sized chunks.

Usage
-----
PYTHONPATH=. MUJOCO_GL=egl python code/apps/warehouse_datagen/gen_warehouse_dart.py \
    generate --episodes 200 --seed 7 --chunk-size 50 --out runs/warehouse_dart/2026-07-12
"""

from __future__ import annotations

# --- GPU EGL fix MUST run before any MuJoCo import ---
from code.apps.warehouse_datagen.egl_gpu import force_nvidia_egl, gpu_utilization
force_nvidia_egl()

import argparse
import datetime as _dt
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from code.scene import derive_rng
from code.teacher import WBCTeacher
from code.apps.warehouse_datagen.rollout import FPS, run_warehouse_dart_episode
from code.apps.warehouse_datagen.scene import sample_episode_plan

_NOISE_SEED_OFFSET = 10000


def _default_out() -> str:
    return f"runs/warehouse_dart/{_dt.date.today().isoformat()}"


def _episode_done(out: Path, i: int, render: bool) -> bool:
    parq = out / "data" / "chunk-000" / f"episode_{i:06d}.parquet"
    side = out / "meta" / "ep_meta" / f"episode_{i:06d}.json"
    if not (parq.exists() and side.exists()):
        return False
    if render and not (out / "videos" / f"episode_{i:06d}_ego.mp4").exists():
        return False
    return True


def _write_episode(out: Path, i: int, result: dict, plan, render: bool) -> dict:
    """Write parquet (+mp4) + sidecar for one finished episode; return sidecar meta."""
    rows = result["rows"]
    df = pd.DataFrame(rows)
    parq = out / "data" / "chunk-000" / f"episode_{i:06d}.parquet"
    df.to_parquet(parq, index=False)

    if render and result["ego_rgb_seq"]:
        import imageio.v2 as imageio
        vid = out / "videos" / f"episode_{i:06d}_ego.mp4"
        imageio.mimwrite(str(vid), result["ego_rgb_seq"], fps=FPS, macro_block_size=1)

    final_dist = float(rows[-1]["goal"][0]) if rows else 99.0
    meta = {
        "episode_index":   i,
        "length":          len(rows),
        "success":         bool(result["reached"]),
        "final_goal_dist": round(final_dist, 3),
        "mode":            plan.mode,
        "route_len":       round(plan.route_len, 3),
        "layout_name":     plan.layout_name,
        "instruction":     plan.instruction,
        "spawn_xy":        [round(plan.spawn_xy[0], 3), round(plan.spawn_xy[1], 3)],
        "goal_xy":         [round(plan.goal_xy[0], 3), round(plan.goal_xy[1], 3)],
        "seed":            plan.seed,
        "is_dart":         True,
        "has_video":       bool(render and result["ego_rgb_seq"]),
    }
    with open(out / "meta" / "ep_meta" / f"episode_{i:06d}.json", "w") as f:
        json.dump(meta, f)
    return meta


def assemble_meta(out: Path, *, seed: int, noise: float, maxsteps: int,
                  render: bool, total_target: int) -> dict:
    """(Re)build meta/{info,stats,episodes,tasks,manifest} from all present episodes."""
    ep_meta_dir = out / "meta" / "ep_meta"
    data_dir = out / "data" / "chunk-000"
    metas = []
    for p in sorted(ep_meta_dir.glob("episode_*.json")):
        with open(p) as f:
            metas.append(json.load(f))
    metas.sort(key=lambda m: m["episode_index"])

    # tasks map (deterministic: sorted unique instructions)
    instrs = sorted({m["instruction"] for m in metas})
    task_map = {ins: k for k, ins in enumerate(instrs)}

    # proprio/action stats by scanning parquets
    all_p, all_a = [], []
    total_frames = 0
    for m in metas:
        i = m["episode_index"]
        parq = data_dir / f"episode_{i:06d}.parquet"
        if not parq.exists():
            continue
        df = pd.read_parquet(parq, columns=["proprio", "action"])
        all_p.extend(df["proprio"].tolist())
        all_a.extend(df["action"].tolist())
        total_frames += len(df)

    def _stat(a: np.ndarray) -> dict:
        return {"mean": a.mean(0).tolist(), "std": (a.std(0) + 1e-6).tolist(),
                "min": a.min(0).tolist(), "max": a.max(0).tolist()}

    arr_p = np.array(all_p, dtype=np.float32) if all_p else np.zeros((1, 55), np.float32)
    arr_a = np.array(all_a, dtype=np.float32) if all_a else np.zeros((1, 15), np.float32)
    stats = {"proprio": _stat(arr_p), "action": _stat(arr_a)}

    n_ep = len(metas)
    n_success = sum(1 for m in metas if m["success"])
    n_route = sum(1 for m in metas if m["mode"] == "route")
    info = {
        "codebase_version": "warehouse_dart_v1",
        "fps":              FPS,
        "robot":            "unitree_g1_lowerbody",
        "domain":           "warehouse",
        "difficulty":       "warehouse",
        "seed":             seed,
        "noise_sigma":      noise,
        "maxsteps_per_ep":  maxsteps,
        "proprio_dim":      55,
        "phase_dim":        2,
        "action_dim":       15,
        "total_episodes":   n_ep,
        "total_frames":     total_frames,
        "n_route":          n_route,
        "n_primitive":      n_ep - n_route,
        "success_rate":     round(n_success / max(1, n_ep), 3),
        "has_video":        render,
        "target_episodes":  total_target,
        "no_render":        not render,
    }

    with open(out / "meta" / "info.json", "w") as f:
        json.dump(info, f, indent=2)
    with open(out / "meta" / "stats.json", "w") as f:
        json.dump(stats, f, indent=2)
    with open(out / "meta" / "episodes.jsonl", "w") as f:
        for m in metas:
            em = dict(m)
            em["task_index"] = task_map[m["instruction"]]
            em["tasks"] = [m["instruction"]]
            f.write(json.dumps(em) + "\n")
    with open(out / "meta" / "tasks.jsonl", "w") as f:
        for ins in instrs:
            f.write(json.dumps({"task_index": task_map[ins], "task": ins}) + "\n")
    with open(out / "meta" / "manifest.jsonl", "w") as f:
        for m in metas:
            f.write(json.dumps(
                {"path": f"data/chunk-000/episode_{m['episode_index']:06d}.parquet"}) + "\n")
    return info


def main_generate(args: argparse.Namespace) -> None:
    out = Path(args.out)
    for d in ["data/chunk-000", "videos", "meta/ep_meta"]:
        (out / d).mkdir(parents=True, exist_ok=True)
    render = not args.no_render
    if args.workers and args.workers > 1:
        print(f"[wh-dart] NOTE: baseline generation is single-process (shared "
              f"WBCTeacher/EGL); --workers={args.workers} ignored, running serial.",
              flush=True)

    print(f"[wh-dart] out={out}  episodes(target)={args.episodes}  seed={args.seed}")
    print(f"[wh-dart] noise={args.noise}  maxsteps={args.maxsteps}  render={render}  "
          f"primitive_frac={args.primitive_frac}", flush=True)
    print(f"[wh-dart] Loading WBCTeacher ({'GPU' if False else 'CPU-ONNX'} teacher; "
          f"MuJoCo rendering on GPU)...", flush=True)
    teacher = WBCTeacher()
    print(f"[wh-dart] Teacher loaded ({teacher.device_str}).", flush=True)

    already = sum(1 for i in range(args.episodes) if _episode_done(out, i, render))
    todo = [i for i in range(args.episodes) if not _episode_done(out, i, render)]
    if args.chunk_size:
        todo = todo[:args.chunk_size]
    print(f"[wh-dart] {already} episodes already done; generating {len(todo)} this run "
          f"(indices {todo[:1]}..{todo[-1:]}).", flush=True)

    t0 = time.time()
    n_new = n_fell = n_frames = 0
    for i in todo:
        plan = sample_episode_plan(args.seed, i, primitive_frac=args.primitive_frac)
        noise_rng = derive_rng(args.seed + _NOISE_SEED_OFFSET, i)
        ep_t0 = time.time()
        result = run_warehouse_dart_episode(
            teacher, plan, episode_idx=i, global_frame_offset=0,
            noise_sigma=args.noise, hard_maxsteps=args.maxsteps,
            rng_noise=noise_rng, render=render, verbose=args.verbose,
        )
        ep_dt = time.time() - ep_t0
        if result is None:
            n_fell += 1
            print(f"  ep {i:06d} [{plan.mode}] FELL/empty (discarded) t={ep_dt:.1f}s",
                  flush=True)
            continue
        m = _write_episode(out, i, result, plan, render)
        n_new += 1
        n_frames += m["length"]
        sps = m["length"] / max(ep_dt, 1e-6)
        print(f"  ep {i:06d} [{plan.mode:9s}] steps={m['length']:4d} reached={m['success']} "
              f"len={plan.route_len:.1f}m t={ep_dt:.1f}s ({sps:.0f}stp/s)", flush=True)

    elapsed = time.time() - t0
    info = assemble_meta(out, seed=args.seed, noise=args.noise,
                         maxsteps=args.maxsteps, render=render,
                         total_target=args.episodes)

    ms_per_frame = (elapsed * 1000.0 / n_frames) if n_frames else 0.0
    print(f"\n{'='*66}")
    print(f"[wh-dart] chunk done: +{n_new} new eps, {n_fell} fell, {n_frames} new frames")
    print(f"          dataset now: {info['total_episodes']} eps / {info['total_frames']} "
          f"frames / success={info['success_rate']}")
    print(f"          route/primitive = {info['n_route']}/{info['n_primitive']}")
    print(f"          throughput: {elapsed:.0f}s  ({ms_per_frame:.2f} ms/frame incl. "
          f"teacher+render)  gpu_util now={gpu_utilization()}%")
    print(f"          output: {out}/")
    remaining = info["target_episodes"] - info["total_episodes"]
    if remaining > 0:
        print(f"          {remaining} episodes remaining — re-run to continue.")
    print(f"{'='*66}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description="Warehouse-domain DART dataset generator (F5)")
    sub = ap.add_subparsers(dest="cmd")
    g = sub.add_parser("generate", help="Generate warehouse DART episodes")
    g.add_argument("--episodes", type=int, default=200, help="TOTAL target episode count")
    g.add_argument("--seed", type=int, default=7)
    g.add_argument("--out", default=None, help="Output dir (default runs/warehouse_dart/<date>)")
    g.add_argument("--noise", type=float, default=0.07, help="DART joint-target noise std (rad)")
    g.add_argument("--maxsteps", type=int, default=1400, help="Hard per-episode step cap")
    g.add_argument("--primitive-frac", type=float, default=0.3,
                   help="Fraction of baseline-style direct-steer (primitive) episodes")
    g.add_argument("--chunk-size", type=int, default=0,
                   help="Generate at most this many NEW episodes this run (0=all remaining)")
    g.add_argument("--no-render", action="store_true", help="Skip ego RGB rendering / mp4")
    g.add_argument("--workers", type=int, default=1, help="(serial only; see --help)")
    g.add_argument("--verbose", action="store_true")

    args = ap.parse_args()
    if args.cmd != "generate":
        ap.print_help()
        sys.exit(1)
    if args.out is None:
        args.out = _default_out()
    main_generate(args)


if __name__ == "__main__":
    main()
