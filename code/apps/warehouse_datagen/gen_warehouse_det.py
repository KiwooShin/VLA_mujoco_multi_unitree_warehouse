"""gen_warehouse_det.py — F-GN: warehouse-domain labeled dataset for GROUND_NET.

Replicates the baseline NX-6 detector-dataset recipe
(``code/datagen/gen_det_dataset.py`` + siblings) *byte-compatibly* — same label
format / parquet+npz schema, so the baseline det trainer
(``code.train.nx6_heatmap``) and loader (``code.perception.detector.data``)
consume it directly — but renders the **warehouse visual domain** instead of the
blue-checker playground:

  * Scenes are built from BOTH the fixed ``hero_layout`` / jittered
    ``sample_layout`` single-hall warehouses AND the multi-room ``rooms_layout``
    (light-gray tiled floor, brown shelves, coloured pads — the exact domain the
    fleet mission's shared viz model renders).
  * Objects are placed at the layout ``object_spots`` (one unique (colour, shape)
    per spot, exactly like ``code.warehouse.arena._default_objects``) PLUS a few
    randomised free-space poses for distractor/viewpoint diversity.
  * Frames come from two families (same as the baseline recipe):
      1. TRAJECTORY — the WBC teacher walks from a random free spawn toward a
         random object, subsampling frames along a natural far->near approach and
         caching a handful of mid-gait qpos snapshots.
      2. TELEPORT — the robot is teleported (mj_forward, cached gait pose) to
         controlled (distance, bearing) offsets from a random object (log-uniform
         0.3-8 m, wide bearings) AND to fully random free-cell "confusion" poses
         across aisles / doorways / alcoves / deep corners.
  * NEGATIVES come for free from the segmentation labels: a frame where a wall /
    shelf occludes an object yields no mask pixels for it, so that (class, colour)
    query becomes a hard "target wall-occluded" negative — exactly the
    through-wall signal Cycle-2a flagged the playground-trained detector missing.

Labels are derived by the SHARED baseline label pipeline
(``code.datagen.gen_det_labels.derive_object_labels`` via
``gen_det_capture.capture_frame``) — MuJoCo instance segmentation on the
``obj_{i}`` geoms (which ``build_warehouse_arena`` names identically to
``build_arena``), so every visible object's class/colour/bbox/centroid/GT-(dist,
bearing) is exact.

Cameras: grounding (480x360, 26 deg pitch) + proximity (320x240, 58 deg) — the
same two ArenaRenderer streams the baseline dataset used and the deploy confirmer
renders.

Usage
-----
  PYTHONPATH=. MUJOCO_GL=egl python -m code.apps.warehouse_datagen.gen_warehouse_det --smoke
  PYTHONPATH=. MUJOCO_GL=egl python -m code.apps.warehouse_datagen.gen_warehouse_det \
      --n-hero 210 --n-rooms 160 --seed 8100 --out dataset/det_warehouse

  # merge with a fraction of the original playground det dataset (anti-forgetting):
  PYTHONPATH=. python -m code.apps.warehouse_datagen.gen_warehouse_det combine \
      --out dataset/det_wh_mix --part dataset/det_warehouse:1.0 \
      --part /home/kiwoos/work/unitree_vla/dataset/det_v1:0.35
"""

from __future__ import annotations

# GPU-rendering fix MUST run before mujoco initialises EGL.
from code.apps.warehouse_datagen.egl_gpu import force_nvidia_egl

force_nvidia_egl()

import argparse
import json
import math
import time
from pathlib import Path

import cv2
import mujoco
import numpy as np
import pandas as pd

from code.arena import ArenaRenderer
from code.datagen.gen_det_capture import SegRenderer, capture_frame
from code.datagen.gen_det_common import (
    COLOR_NAMES, DUAL_RENDER_PROB, FALL_HEIGHT, MAX_GAIT_SNAPSHOTS, MIN_PIXELS,
    SETTLE_STEPS, SHAPE_NAMES, pick_cam,
)
from code.datagen.gen_det_labels import build_id_to_obj
from code.sim.arena_build import COLORS, SHAPES
from code.sim.scene import derive_rng
from code.steer import steer as steer_cmd
from code.teacher import WBCTeacher, _yaw_of
from code.warehouse.arena import (_default_objects, build_warehouse_arena,
                                  warehouse_scene_cfg)
from code.warehouse.layout import (CALLSIGNS, hero_layout, rooms_layout,
                                   sample_layout)
from code.apps.warehouse_demo.planning import build_inflated_grid
from code.apps.warehouse_datagen.scene import (GRID_RES, INFLATE_RADIUS,
                                               free_cells_world)

# ---------------------------------------------------------------------------
# Per-scene frame budget (mirrors the baseline det recipe's proportions but
# leans on cheap teleport frames for warehouse aisle/doorway/corner coverage).
# ---------------------------------------------------------------------------
WH_MAXSTEPS_TRAJ: int = 220         # bounded walk (warehouse walls make long walks flaky)
N_TRAJ_TARGET: int = 10             # aim for ~this many trajectory samples per scene
N_TELEPORT_FOCUS: int = 12          # controlled (dist, bearing) offsets from a random object
N_TELEPORT_AIMED: int = 8           # free-cell pose AIMED at a random object (dense occlusion negs)
N_TELEPORT_RANDOM: int = 8          # fully random free-cell confusion poses
EXTRA_FREE_OBJECTS_MAX: int = 3     # random free-space objects added on top of object_spots
FOCUS_LOG_LO, FOCUS_LOG_HI = math.log(0.3), math.log(8.0)
_ALL_COMBOS = [(ci, si) for ci in range(len(COLORS)) for si in range(len(SHAPES))]

# How many random frames to stash for the heatmap-overlay contact sheet.
_N_CONTACT = 16


# ---------------------------------------------------------------------------
# Scene construction
# ---------------------------------------------------------------------------
def _pick_layout(family: str, rng: np.random.Generator):
    """Return a warehouse layout for the requested family.

    Args:
        family: "hero" (fixed hero_layout or a jittered sample_layout) or
            "rooms" (the fixed multi-room layout).
        rng: RNG used to decide hero-vs-jitter and to seed sample_layout.

    Returns:
        A validated :class:`code.warehouse.layout.WarehouseLayout`.
    """
    if family == "rooms":
        return rooms_layout()
    # hero family: mostly jittered variants, sometimes the canonical hero.
    if rng.random() < 0.85:
        return sample_layout(rng)
    return hero_layout()


def _scene_objects(layout, rng: np.random.Generator) -> list:
    """Objects at every layout spot + a few random free-space poses.

    Each object keeps a UNIQUE (colour, shape) pair (baseline invariant), sized
    from the canonical SHAPES table so the label radius correction matches.

    Args:
        layout: Source warehouse layout.
        rng: RNG for combo choice + free-space placement.

    Returns:
        List of object dicts (color_name, color_rgb, shape_name, size, x, y).
    """
    objects = _default_objects(layout, rng)
    used = {(o["color_name"], o["shape_name"]) for o in objects}

    # Random free-space objects, clear of walls and of one another / the spots.
    n_extra = int(rng.integers(0, EXTRA_FREE_OBJECTS_MAX + 1))
    if n_extra > 0:
        cfg0 = warehouse_scene_cfg(layout, objects=[], rng=rng)
        grid = build_inflated_grid(cfg0, GRID_RES, 0.35)
        cells = free_cells_world(grid)
        rng.shuffle(cells)
        placed = [(o["x"], o["y"]) for o in objects]
        avail = [(c, s) for (ci, si), (c, cr), (s, sz)
                 in [((ci, si), COLORS[ci], SHAPES[si]) for ci in range(len(COLORS))
                     for si in range(len(SHAPES))]
                 if (c, s) not in used]
        rng.shuffle(avail)
        ci_iter = iter(avail)
        for cx, cy in cells:
            if n_extra <= 0:
                break
            if any(math.hypot(cx - px, cy - py) < 0.8 for px, py in placed):
                continue
            try:
                color_name, shape_name = next(ci_iter)
            except StopIteration:
                break
            color_rgb = dict(COLORS)[color_name]
            size = dict(SHAPES)[shape_name]
            objects.append({"color_name": color_name, "color_rgb": color_rgb,
                            "shape_name": shape_name, "size": float(size),
                            "x": float(cx), "y": float(cy)})
            placed.append((cx, cy))
            n_extra -= 1
    return objects


def run_warehouse_scene(scene_id: int, family: str, seed: int):
    """Build one warehouse scene and capture its trajectory + teleport frames.

    Args:
        scene_id: Scene index (drives the deterministic per-scene RNG).
        family: "hero" or "rooms".
        seed: Base dataset seed.

    Returns:
        Tuple (scene_cfg, frame_records, preview_samples). ``frame_records`` is
        empty if the robot fell during the initial settle. ``preview_samples`` is
        a small reservoir of (rgb, labels, meta) dicts for the contact sheet.
    """
    rng = derive_rng(seed, scene_id)
    rng_s = np.random.default_rng(np.random.SeedSequence([seed, 0xA11CE, scene_id]))

    layout = _pick_layout(family, rng)
    callsign = CALLSIGNS[int(rng.integers(len(CALLSIGNS)))]
    objects = _scene_objects(layout, rng)
    n_obj = len(objects)
    target_idx = int(rng.integers(n_obj)) if n_obj else 0
    scene_cfg = warehouse_scene_cfg(layout, robot=callsign, objects=objects,
                                    rng=rng, target_index=target_idx)
    scene_cfg["family"] = family

    model = build_warehouse_arena(scene_cfg)
    model.opt.timestep = 0.005
    id_to_obj = build_id_to_obj(model, n_obj)

    # Free-space grid (inflated by robot radius) for spawn + teleport poses.
    grid = build_inflated_grid(scene_cfg, GRID_RES, INFLATE_RADIUS)
    free = free_cells_world(grid)

    teacher = WBCTeacher(use_gpu=False)
    teacher.model = model
    teacher.data = mujoco.MjData(model)
    teacher._nj = model.nq - 7
    teacher._pelvis_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")

    spawn = free[int(rng_s.integers(len(free)))]
    spawn_yaw = float(rng_s.uniform(-math.pi, math.pi))
    teacher.reset(pos_xy=(float(spawn[0]), float(spawn[1])), yaw=spawn_yaw)

    data = teacher.data
    for _ in range(SETTLE_STEPS):
        teacher.step(vel_cmd=(0.0, 0.0, 0.0))
        if teacher.base_height < FALL_HEIGHT:
            return scene_cfg, [], []

    renderer = ArenaRenderer(model)
    seg_rend = SegRenderer(model)

    frame_records: list = []
    previews: list = []
    gait_snapshots = [data.qpos.copy()]

    def _reservoir(rec):
        """Reservoir-sample a preview copy of an interesting (labeled) frame."""
        if not rec["labels"]:
            return
        item = dict(rgb=rec["rgb"].copy(), labels=rec["labels"],
                    cam_type=rec["cam_type"], source=rec["source"],
                    family=family, scene_id=scene_id)
        if len(previews) < _N_CONTACT:
            previews.append(item)
        elif rng_s.random() < 0.25:
            previews[int(rng_s.integers(_N_CONTACT))] = item

    def _cams(cam_type):
        cams = [cam_type]
        if rng_s.random() < DUAL_RENDER_PROB:
            cams.append("proximity" if cam_type == "grounding" else "grounding")
        return cams

    # ---- 1) TRAJECTORY: walk from spawn toward a random object ----
    tgt = objects[int(rng_s.integers(n_obj))]
    target_xy = np.array([tgt["x"], tgt["y"]], dtype=np.float64)
    sample_every = max(3, WH_MAXSTEPS_TRAJ // N_TRAJ_TARGET)
    step, last_sample = 0, -999
    while step < WH_MAXSTEPS_TRAJ:
        robot_xy = data.qpos[0:2].copy()
        robot_yaw = _yaw_of(data.qpos[3:7])
        dist_to_target = float(np.linalg.norm(robot_xy - target_xy))
        if step - last_sample >= sample_every:
            for ct in _cams(pick_cam(dist_to_target)):
                rec = capture_frame(renderer, seg_rend, data, robot_yaw, ct,
                                    objects, id_to_obj)
                rec["source"] = "trajectory"
                frame_records.append(rec)
                _reservoir(rec)
            last_sample = step
            if len(gait_snapshots) < MAX_GAIT_SNAPSHOTS and step > 0:
                gait_snapshots.append(data.qpos.copy())
            if dist_to_target < 0.35:
                break
        vel_cmd, _, _ = steer_cmd(robot_xy, robot_yaw, target_xy, stop_r=0.25)
        teacher.step(vel_cmd=tuple(float(v) for v in vel_cmd))
        if teacher.base_height < FALL_HEIGHT:
            break
        step += 1

    # ---- Teleport helpers ----
    def _apply_pose(rx, ry, yaw):
        snap = gait_snapshots[int(rng_s.integers(len(gait_snapshots)))]
        data.qpos[:] = snap
        data.qpos[0], data.qpos[1] = rx, ry
        data.qpos[3] = math.cos(yaw / 2.0)
        data.qpos[4] = 0.0
        data.qpos[5] = 0.0
        data.qpos[6] = math.sin(yaw / 2.0)
        mujoco.mj_forward(model, data)

    # ---- 2) TELEPORT focus: controlled (dist, bearing) to a random object ----
    n_focus = 0
    for _ in range(N_TELEPORT_FOCUS * 4):
        if n_focus >= N_TELEPORT_FOCUS:
            break
        focus = objects[int(rng_s.integers(n_obj))]
        dist = math.exp(rng_s.uniform(FOCUS_LOG_LO, FOCUS_LOG_HI))
        wide = rng_s.random() < 0.3
        bearing_off = float(rng_s.uniform(-85, 85) if wide else rng_s.uniform(-45, 45))
        approach = float(rng_s.uniform(-math.pi, math.pi))
        rx = focus["x"] + dist * math.cos(approach)
        ry = focus["y"] + dist * math.sin(approach)
        if not grid.is_free((rx, ry)):
            continue
        yaw = (approach + math.pi) - math.radians(bearing_off)
        _apply_pose(rx, ry, yaw)
        for ct in _cams(pick_cam(dist)):
            rec = capture_frame(renderer, seg_rend, data, yaw, ct, objects, id_to_obj)
            rec["source"] = "teleport_focus"
            frame_records.append(rec)
            _reservoir(rec)
        n_focus += 1

    # ---- 2b) TELEPORT aimed: free-cell pose AIMED at a random object ----
    # Points the camera where an object actually is. When the line of sight is
    # clear the object lands centred (a positive); when a wall / shelf intervenes
    # the object is segmentation-occluded and the (class, colour) query becomes a
    # hard "target present but wall-occluded" negative — the exact
    # camera-pointed-at-the-object-but-blocked distribution the deploy confirmer
    # and perception eval probe, which the random-yaw poses above under-sample.
    n_aim = 0
    for _ in range(N_TELEPORT_AIMED * 4):
        if n_aim >= N_TELEPORT_AIMED:
            break
        cell = free[int(rng_s.integers(len(free)))]
        rx, ry = float(cell[0]), float(cell[1])
        obj = objects[int(rng_s.integers(n_obj))]
        d = math.hypot(obj["x"] - rx, obj["y"] - ry)
        if d < 0.5 or d > 9.0:
            continue
        yaw = math.atan2(obj["y"] - ry, obj["x"] - rx) + float(rng_s.uniform(-0.22, 0.22))
        _apply_pose(rx, ry, yaw)
        for ct in _cams(pick_cam(d)):
            rec = capture_frame(renderer, seg_rend, data, yaw, ct, objects, id_to_obj)
            rec["source"] = "teleport_aimed"
            frame_records.append(rec)
            _reservoir(rec)
        n_aim += 1

    # ---- 3) TELEPORT random: free-cell confusion poses (aisles/doorways/corners) ----
    n_rand = 0
    for _ in range(N_TELEPORT_RANDOM * 4):
        if n_rand >= N_TELEPORT_RANDOM:
            break
        cell = free[int(rng_s.integers(len(free)))]
        rx, ry = float(cell[0]), float(cell[1])
        yaw = float(rng_s.uniform(-math.pi, math.pi))
        _apply_pose(rx, ry, yaw)
        ct = "proximity" if rng_s.random() < 0.4 else "grounding"
        rec = capture_frame(renderer, seg_rend, data, yaw, ct, objects, id_to_obj)
        rec["source"] = "teleport_random"
        frame_records.append(rec)
        _reservoir(rec)
        n_rand += 1

    renderer.close()
    seg_rend.close()
    return scene_cfg, frame_records, previews


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def generate(args: argparse.Namespace) -> tuple[dict, Path]:
    """Generate warehouse scenes and write the baseline-compatible det dataset.

    Args:
        args: Parsed CLI args (n_hero, n_rooms, seed, out, smoke, smoke_scenes).

    Returns:
        Tuple (meta, out_dir).
    """
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[gen_warehouse_det] MUJOCO_GL={__import__('os').environ.get('MUJOCO_GL')} "
          f"mujoco={mujoco.__version__}", flush=True)

    plan = ["hero"] * args.n_hero + ["rooms"] * args.n_rooms
    rng_split = np.random.default_rng(np.random.SeedSequence([args.seed, 0xD5]))
    order = rng_split.permutation(len(plan))
    n_train = int(round(0.8 * len(plan)))
    n_val = int(round(0.1 * len(plan)))
    split_of = {}
    for rank, idx in enumerate(order):
        split_of[idx] = ("train" if rank < n_train
                         else "val" if rank < n_train + n_val else "test")

    scenes_meta: dict = {}
    per_split: dict = {"train": [], "val": [], "test": []}
    all_previews: list = []

    t0 = time.perf_counter()
    n_ok = n_fell = 0
    for scene_id, family in enumerate(plan):
        scene_cfg, recs, previews = run_warehouse_scene(scene_id, family, args.seed)
        split = split_of[scene_id]
        if not recs:
            n_fell += 1
            print(f"  [scene {scene_id:4d}] {family:5s} SKIP (fell during settle)", flush=True)
            continue
        n_ok += 1
        scenes_meta[scene_id] = dict(
            style=family, split=split, arena_size=scene_cfg["arena_size"],
            layout_name=scene_cfg["layout_name"], objects=scene_cfg["objects"],
            target_index=scene_cfg["target_index"], instruction=scene_cfg["instruction"],
            lighting=scene_cfg.get("lighting", {}))
        per_split[split].append((scene_id, family, scene_cfg, recs))
        all_previews.extend(previews)
        print(f"  [scene {scene_id:4d}] {family:5s} split={split:5s} "
              f"n_obj={len(scene_cfg['objects'])} frames={len(recs):3d} "
              f"elapsed={time.perf_counter()-t0:6.1f}s", flush=True)
        if args.smoke and n_ok >= args.smoke_scenes:
            break

    print(f"\n[gen] scenes ok={n_ok} fell={n_fell} "
          f"wall={time.perf_counter()-t0:.1f}s", flush=True)

    # ---- Write per-split artifacts (identical schema to gen_det_dataset) ----
    all_frame_rows, all_label_rows = [], []
    frame_uid = 0
    geom_err_dist, geom_err_bearing = [], []
    classes_counts = {s: 0 for s in SHAPE_NAMES}

    for split in ("train", "val", "test"):
        split_dir = out_dir / split
        split_dir.mkdir(parents=True, exist_ok=True)
        buf = {"grounding": {"rgb": [], "depth": []}, "proximity": {"rgb": [], "depth": []}}
        idx_counter = {"grounding": 0, "proximity": 0}
        for scene_id, family, scene_cfg, recs in per_split[split]:
            target_idx = scene_cfg["target_index"]
            for rec in recs:
                ct = rec["cam_type"]
                arr_idx = idx_counter[ct]
                idx_counter[ct] += 1
                buf[ct]["rgb"].append(rec["rgb"])
                buf[ct]["depth"].append(rec["depth"])
                all_frame_rows.append(dict(
                    frame_uid=frame_uid, scene_id=scene_id, split=split, difficulty=family,
                    cam_type=ct, array_idx=arr_idx, source=rec["source"],
                    robot_x=rec["robot_x"], robot_y=rec["robot_y"], robot_yaw=rec["robot_yaw"],
                    qpos=rec["qpos"].tolist(), instruction=scene_cfg["instruction"],
                    target_index=int(target_idx), n_objects_visible=rec["n_objects_visible"],
                    lighting_ambient=float(scene_cfg.get("lighting", {}).get("ambient", 0.5))))
                for lb in rec["labels"]:
                    all_label_rows.append(dict(
                        frame_uid=frame_uid, scene_id=scene_id, split=split, cam_type=ct,
                        is_instructed_target=(lb["obj_idx"] == target_idx), **lb))
                    classes_counts[lb["class_name"]] = classes_counts.get(lb["class_name"], 0) + 1
                    good = ((not lb["clipped"]) and lb["area_px"] >= 200
                            and not math.isnan(lb["err_dist_m"] or np.nan))
                    if good:
                        geom_err_dist.append(lb["err_dist_m"])
                        geom_err_bearing.append(lb["err_bearing_deg"])
                frame_uid += 1
        for ct in ("grounding", "proximity"):
            if not buf[ct]["rgb"]:
                continue
            rgb_arr = np.stack(buf[ct]["rgb"], axis=0).astype(np.uint8)
            depth_arr = np.stack(buf[ct]["depth"], axis=0).astype(np.float16)
            np.savez_compressed(split_dir / f"images_{ct}.npz", rgb=rgb_arr, depth=depth_arr)
            print(f"[gen] {split}/images_{ct}.npz rgb={rgb_arr.shape} "
                  f"file={(split_dir / f'images_{ct}.npz').stat().st_size/1e6:.1f}MB", flush=True)

    frames_df = pd.DataFrame(all_frame_rows)
    labels_df = pd.DataFrame(all_label_rows)
    for split in ("train", "val", "test"):
        split_dir = out_dir / split
        frames_df[frames_df["split"] == split].to_parquet(split_dir / "frames.parquet", index=False)
        labels_df[labels_df["split"] == split].to_parquet(split_dir / "labels.parquet", index=False)
        print(f"[gen] {split}: frames={int((frames_df['split']==split).sum())} "
              f"labels={int((labels_df['split']==split).sum()) if len(labels_df) else 0}", flush=True)

    with open(out_dir / "scenes.json", "w") as f:
        json.dump(scenes_meta, f, indent=1)

    ged = np.array(geom_err_dist or [0.0], dtype=np.float64)
    geb = np.array(geom_err_bearing or [0.0], dtype=np.float64)
    meta = dict(
        dataset="warehouse_det", frames_total=int(len(frames_df)),
        frames_grounding_cam=int((frames_df["cam_type"] == "grounding").sum()) if len(frames_df) else 0,
        frames_proximity_cam=int((frames_df["cam_type"] == "proximity").sum()) if len(frames_df) else 0,
        scenes=int(n_ok), scenes_fell=int(n_fell),
        n_hero=int(sum(1 for _, fam, _, _ in sum(per_split.values(), []) if fam == "hero")),
        n_rooms=int(sum(1 for _, fam, _, _ in sum(per_split.values(), []) if fam == "rooms")),
        classes=SHAPE_NAMES, colors=COLOR_NAMES, classes_counts=classes_counts,
        n_labels_total=int(len(labels_df)),
        label_geometry_err_m_p95=float(np.percentile(ged, 95)),
        label_geometry_err_deg_p95=float(np.percentile(geb, 95)),
        label_geometry_n_checked=int(ged.size), seed=args.seed,
        source_counts=(frames_df["source"].value_counts().to_dict() if len(frames_df) else {}),
        split_scene_counts={s: len(per_split[s]) for s in ("train", "val", "test")})
    with open(out_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"\n[gen] META: {json.dumps(meta, indent=2)}", flush=True)

    if all_previews:
        _contact_sheet(all_previews, Path(args.ops_dir), seed=args.seed)
    return meta, out_dir


# ---------------------------------------------------------------------------
# Contact sheet: RGB + gaussian heatmap-label overlay + bbox + text
# ---------------------------------------------------------------------------
def _gauss(h: int, w: int, cx: float, cy: float, sigma: float) -> np.ndarray:
    ys = np.arange(h, dtype=np.float32)[:, None]
    xs = np.arange(w, dtype=np.float32)[None, :]
    return np.exp(-((xs - cx) ** 2 + (ys - cy) ** 2) / (2.0 * sigma * sigma)).astype(np.float32)


def _contact_sheet(previews: list, ops_dir: Path, seed: int, n: int = 12) -> str:
    """Write a grid PNG of sampled frames with heatmap-label overlays.

    For every visible object in each sampled frame, a Gaussian peaked at its
    labelled centroid (the exact supervision target the trainer builds) is
    blended in as a warm overlay, plus the bbox + a colour/shape/distance
    caption — so a human can eyeball that labels land on the right pixels.
    """
    ops_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed ^ 0xC0FFEE)
    idx = rng.choice(len(previews), size=min(n, len(previews)), replace=False)
    tiles = []
    for k in idx:
        it = previews[int(k)]
        rgb = it["rgb"].astype(np.float32)
        h, w = rgb.shape[:2]
        heat = np.zeros((h, w), dtype=np.float32)
        vis = rgb.copy()
        for lb in it["labels"]:
            if lb["area_px"] < MIN_PIXELS:
                continue
            heat = np.maximum(heat, _gauss(h, w, lb["centroid_px_x"], lb["centroid_px_y"],
                                           sigma=max(3.0, w / 60.0)))
            x, y, bw, bh = int(lb["bbox_x"]), int(lb["bbox_y"]), int(lb["bbox_w"]), int(lb["bbox_h"])
            cv2.rectangle(vis, (x, y), (x + bw, y + bh), (0, 255, 0), 1)
            cv2.putText(vis, f"{lb['color_name']} {lb['class_name']} d={lb['dist_gt_m']:.1f}",
                        (x, max(10, y - 3)), cv2.FONT_HERSHEY_SIMPLEX, 0.33, (255, 255, 0), 1,
                        cv2.LINE_AA)
        heat_rgb = np.zeros_like(vis)
        heat_rgb[..., 0] = (heat * 255).astype(np.uint8)  # red heat
        heat_rgb[..., 1] = (heat * 90).astype(np.uint8)
        vis = np.clip(vis * (1 - 0.45 * heat[..., None]) + heat_rgb * (0.45 * heat[..., None]),
                      0, 255).astype(np.uint8)
        header = f"{it['family']} sc{it['scene_id']} {it['cam_type']} {it['source']} nobj={len(it['labels'])}"
        canvas = np.zeros((h + 18, w, 3), dtype=np.uint8)
        canvas[18:] = vis
        cv2.putText(canvas, header, (2, 13), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (255, 255, 255), 1,
                    cv2.LINE_AA)
        tiles.append(cv2.resize(canvas, (384, 306)))
    cols = 4
    rows = int(math.ceil(len(tiles) / cols))
    sheet = np.full((rows * 306, cols * 384, 3), 20, dtype=np.uint8)
    for i, t in enumerate(tiles):
        r, c = divmod(i, cols)
        sheet[r * 306:(r + 1) * 306, c * 384:(c + 1) * 384] = t
    png = str(ops_dir / "det_contact_sheet.png")
    cv2.imwrite(png, cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR))
    print(f"[contact-sheet] wrote {len(tiles)} tiles -> {png}", flush=True)
    return png


# ---------------------------------------------------------------------------
# Validation: load with the baseline det loader
# ---------------------------------------------------------------------------
def validate_with_loader(out_dir: str) -> dict:
    """Load every split with the baseline SplitCache and report counts."""
    from code.perception.detector.data import SplitCache, build_example_index
    report = {}
    for split in ("train", "val", "test"):
        if not (Path(out_dir) / split / "frames.parquet").exists():
            continue
        cache = SplitCache(out_dir, split, verbose=False)
        ex = build_example_index(cache, np.random.default_rng(0))
        n_pos = sum(1 for e in ex if e[3] is not None)
        report[split] = dict(frames=len(cache), examples=len(ex), positives=n_pos,
                             negatives=len(ex) - n_pos)
        print(f"[validate] {split}: frames={len(cache)} examples={len(ex)} "
              f"pos={n_pos} neg={len(ex)-n_pos}", flush=True)
    return report


# ---------------------------------------------------------------------------
# Combine: merge warehouse + a fraction of another det dataset (anti-forgetting)
# ---------------------------------------------------------------------------
def _read_split(root: str, split: str):
    """Return (frames_df, labels_df, npz{cam->{rgb,depth}}) for one split, or None."""
    base = Path(root) / split
    if not (base / "frames.parquet").exists():
        return None
    frames = pd.read_parquet(base / "frames.parquet")
    labels = pd.read_parquet(base / "labels.parquet")
    npz = {}
    for cam in ("grounding", "proximity"):
        p = base / f"images_{cam}.npz"
        if p.exists():
            with np.load(p) as z:
                npz[cam] = dict(rgb=z["rgb"], depth=z["depth"])
    return frames, labels, npz


def combine_datasets(out: str, parts: list, seed: int = 0) -> dict:
    """Merge several det datasets (each subsampled by a fraction) into ``out``.

    Frames are subsampled by ``frac`` per split (frame granularity), array_idx +
    frame_uid are reindexed so the merged npz/parquet stay self-consistent and
    load with the baseline SplitCache unchanged.

    Args:
        out: Output dataset dir.
        parts: List of (root, frac) — dataset dir and keep-fraction in [0, 1].
        seed: RNG seed for the subsampling.

    Returns:
        The merged meta dict.
    """
    out_dir = Path(out)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    totals = {}
    for split in ("train", "val", "test"):
        split_dir = out_dir / split
        split_dir.mkdir(parents=True, exist_ok=True)
        merged_frames, merged_labels = [], []
        buf = {"grounding": {"rgb": [], "depth": []}, "proximity": {"rgb": [], "depth": []}}
        idx_counter = {"grounding": 0, "proximity": 0}
        uid = 0
        for root, frac in parts:
            got = _read_split(root, split)
            if got is None:
                continue
            frames, labels, npz = got
            frames = frames.sort_values("frame_uid").reset_index(drop=True)
            keep_uids = set(frames["frame_uid"].tolist())
            if frac < 1.0:
                all_uids = frames["frame_uid"].values
                k = int(round(frac * len(all_uids)))
                keep_uids = set(rng.choice(all_uids, size=k, replace=False).tolist())
            lab_by_uid = {}
            for r in labels.itertuples():
                lab_by_uid.setdefault(int(r.frame_uid), []).append(r)
            for row in frames.itertuples():
                if int(row.frame_uid) not in keep_uids:
                    continue
                cam = row.cam_type
                if cam not in npz:
                    continue
                new_idx = idx_counter[cam]
                idx_counter[cam] += 1
                buf[cam]["rgb"].append(npz[cam]["rgb"][int(row.array_idx)])
                buf[cam]["depth"].append(npz[cam]["depth"][int(row.array_idx)])
                fr = {c: getattr(row, c) for c in frames.columns}
                fr["frame_uid"] = uid
                fr["array_idx"] = new_idx
                fr["split"] = split
                merged_frames.append(fr)
                for lr in lab_by_uid.get(int(row.frame_uid), []):
                    ld = {c: getattr(lr, c) for c in labels.columns}
                    ld["frame_uid"] = uid
                    ld["split"] = split
                    merged_labels.append(ld)
                uid += 1
        for cam in ("grounding", "proximity"):
            if not buf[cam]["rgb"]:
                continue
            np.savez_compressed(split_dir / f"images_{cam}.npz",
                                rgb=np.stack(buf[cam]["rgb"]).astype(np.uint8),
                                depth=np.stack(buf[cam]["depth"]).astype(np.float16))
        pd.DataFrame(merged_frames).to_parquet(split_dir / "frames.parquet", index=False)
        pd.DataFrame(merged_labels).to_parquet(split_dir / "labels.parquet", index=False)
        totals[split] = len(merged_frames)
        print(f"[combine] {split}: {len(merged_frames)} frames from "
              f"{[(Path(r).name, f) for r, f in parts]}", flush=True)
    meta = dict(dataset="combined", parts=[(str(r), f) for r, f in parts],
                split_frame_counts=totals, seed=seed)
    with open(out_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    return meta


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> None:
    """CLI: default subcommand generates; ``combine`` merges datasets."""
    ap = argparse.ArgumentParser(description="warehouse GROUND_NET detector dataset")
    sub = ap.add_subparsers(dest="cmd")

    ap.add_argument("--n-hero", type=int, default=210)
    ap.add_argument("--n-rooms", type=int, default=160)
    ap.add_argument("--seed", type=int, default=8100)
    ap.add_argument("--out", default="dataset/det_warehouse")
    ap.add_argument("--ops-dir", default="ops/gn_ft")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--smoke-scenes", type=int, default=2)
    ap.add_argument("--no-validate", action="store_true")

    cp = sub.add_parser("combine", help="merge det datasets (root:frac)")
    cp.add_argument("--out", required=True)
    cp.add_argument("--part", action="append", required=True,
                    help="repeatable ROOT:FRAC, e.g. dataset/det_warehouse:1.0")
    cp.add_argument("--seed", type=int, default=0)

    args = ap.parse_args()

    if args.cmd == "combine":
        parts = []
        for spec in args.part:
            root, _, frac = spec.rpartition(":")
            parts.append((root, float(frac)))
        combine_datasets(args.out, parts, seed=args.seed)
        validate_with_loader(args.out)
        return

    if args.smoke:
        args.n_hero = min(args.n_hero, 2)
        args.n_rooms = min(args.n_rooms, 2)

    t0 = time.perf_counter()
    meta, out_dir = generate(args)
    if not args.no_validate and meta["frames_total"] > 0:
        validate_with_loader(str(out_dir))
    dt = time.perf_counter() - t0
    print(f"\n[gen_warehouse_det] DONE in {dt/60:.1f} min "
          f"({meta['frames_total']} frames, {meta['scenes']} scenes)", flush=True)
    if args.smoke and meta["scenes"]:
        print(f"[smoke] {dt/meta['scenes']:.1f}s/scene "
              f"{dt/max(1,meta['frames_total'])*1000:.0f}ms/frame", flush=True)


if __name__ == "__main__":
    main()
