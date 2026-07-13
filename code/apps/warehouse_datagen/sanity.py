"""sanity.py — Post-generation validation for the warehouse DART dataset (F5).

Three checks, run automatically after generation (and standalone):

  1. schema conformance — load the dataset with the BASELINE dataset classes
     (``PhaseParquetDataset`` for 57-d proprio+phase, ``ParquetDataset`` for ego
     RGB from mp4) and assert every tensor's shape/dtype/range, incl. that the
     rendered ego frames are non-zero (real warehouse pixels, not placeholders).
  2. contact sheet — a 3x3 PNG of sampled ego frames (ops/f5/) to eyeball the
     visual domain (gray tiled floor + brown shelves, no blue playground).
  3. distribution stats — action / vel_cmd / goal distributions, printed
     side-by-side with a slice of the ORIGINAL DART dataset when available.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

ORIGINAL_DART_DIR = "/home/kiwoos/work/unitree_vla/dataset/dart_demo"


# ---------------------------------------------------------------------------
# 1. Schema conformance via the baseline dataset classes
# ---------------------------------------------------------------------------
def check_schema(out_dir: str) -> dict:
    """Load with the baseline dataset classes and validate shapes/dtypes/ranges."""
    from code.data.dataset_phase import PhaseParquetDataset, PROPRIO_DIM_PHASE
    from code.data.dataset import ParquetDataset

    res: dict = {"ok": True, "errors": []}

    def _fail(msg: str) -> None:
        res["ok"] = False
        res["errors"].append(msg)

    ds = PhaseParquetDataset([out_dir], split="train", train_fraction=0.9)
    if len(ds) == 0:
        _fail("PhaseParquetDataset produced 0 samples")
        return res
    s = ds[0]
    K = ds.K
    if tuple(s["proprio_h"].shape) != (K, PROPRIO_DIM_PHASE):
        _fail(f"proprio_h shape {tuple(s['proprio_h'].shape)} != ({K},{PROPRIO_DIM_PHASE})")
    if s["action"].shape[-1] != 15:
        _fail(f"action dim {s['action'].shape[-1]} != 15")
    if tuple(s["goal"].shape) != (3,):
        _fail(f"goal shape {tuple(s['goal'].shape)} != (3,)")
    if tuple(s["vel_cmd"].shape) != (3,):
        _fail(f"vel_cmd shape {tuple(s['vel_cmd'].shape)} != (3,)")
    ph = s["proprio_h"][-1, -2:].numpy()  # [sin, cos] of last frame
    if not (np.all(np.abs(ph) <= 1.0 + 1e-4)):
        _fail(f"phase sin/cos out of [-1,1]: {ph}")
    if abs(float(np.hypot(ph[0], ph[1])) - 1.0) > 0.05 and not np.allclose(ph, 0):
        _fail(f"phase not on unit circle: {ph} (|.|={np.hypot(*ph):.3f})")
    for k in ("proprio_h", "action", "goal", "vel_cmd"):
        if not bool(np.isfinite(s[k].numpy()).all()):
            _fail(f"{k} has non-finite values")
    if float(s["goal"][0]) < 0:
        _fail(f"goal dist negative: {float(s['goal'][0])}")
    res["phase_samples_with_phase_col"] = int(
        sum(1 for ep in ds._episodes if ep["has_phase"]))
    res["n_samples_phase"] = len(ds)

    # ego RGB from mp4 (visual-domain readiness)
    pds = ParquetDataset(out_dir, split="train", train_fraction=0.9, load_video=True)
    if len(pds) > 0:
        sp = pds[len(pds) // 2]
        rgb = sp["ego_rgb"].numpy()
        if tuple(rgb.shape) != (3, 128, 128):
            _fail(f"ego_rgb shape {tuple(rgb.shape)} != (3,128,128)")
        res["ego_rgb_nonzero_frac"] = float((rgb > 0).mean())
        res["ego_rgb_mean"] = float(rgb.mean())
        if res["ego_rgb_nonzero_frac"] < 0.5:
            _fail(f"ego_rgb looks empty (nonzero frac {res['ego_rgb_nonzero_frac']:.3f}) "
                  f"— rendering/video missing?")
    else:
        res["ego_rgb_nonzero_frac"] = None
    return res


# ---------------------------------------------------------------------------
# 2. Contact sheet of sampled ego frames
# ---------------------------------------------------------------------------
def contact_sheet(out_dir: str, png_path: str, n: int = 9) -> Optional[str]:
    """Write a 3x3 PNG of ego frames sampled across episodes. Returns path or None."""
    import cv2

    vids = sorted((Path(out_dir) / "videos").glob("episode_*_ego.mp4"))
    if not vids:
        return None
    side = int(math.sqrt(n))
    cell = 240
    sheet = np.full((side * cell, side * cell, 3), 30, dtype=np.uint8)
    picks = [vids[int(round(k * (len(vids) - 1) / max(1, n - 1)))] for k in range(n)]
    for idx, vp in enumerate(picks):
        cap = cv2.VideoCapture(str(vp))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
        cap.set(cv2.CAP_PROP_POS_FRAMES, total // 2)
        ok, frame = cap.read()
        cap.release()
        if not ok:
            continue
        frame = cv2.resize(frame, (cell, cell))
        cv2.putText(frame, vp.stem.replace("_ego", ""), (6, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
        r, c = divmod(idx, side)
        sheet[r * cell:(r + 1) * cell, c * cell:(c + 1) * cell] = frame
    Path(png_path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(png_path, sheet)
    return png_path


# ---------------------------------------------------------------------------
# 3. Distribution stats vs the original DART dataset
# ---------------------------------------------------------------------------
def _load_slice(repo: str, max_frames: int = 40000) -> Optional[dict]:
    chunk = Path(repo) / "data" / "chunk-000"
    if not chunk.exists():
        return None
    acts, vels, goals = [], [], []
    for p in sorted(chunk.glob("episode_*.parquet")):
        df = pd.read_parquet(p, columns=["action", "vel_cmd", "goal"])
        acts.extend(df["action"].tolist())
        vels.extend(df["vel_cmd"].tolist())
        goals.extend(df["goal"].tolist())
        if len(acts) >= max_frames:
            break
    if not acts:
        return None
    a = np.array(acts, np.float32)
    v = np.array(vels, np.float32)
    g = np.array(goals, np.float32)
    return {
        "n_frames": len(a),
        "action_mean_abs": float(np.abs(a).mean()),
        "action_std_mean": float(a.std(0).mean()),
        "vx_mean": float(v[:, 0].mean()), "vx_std": float(v[:, 0].std()),
        "wz_mean": float(v[:, 2].mean()), "wz_std": float(v[:, 2].std()),
        "turn_frac": float((np.abs(v[:, 2]) > 0.05).mean()),
        "moving_frac": float((v[:, 0] > 0.05).mean()),
        "goal_dist_mean": float(g[:, 0].mean()), "goal_dist_max": float(g[:, 0].max()),
    }


def distribution_stats(out_dir: str, original_dir: Optional[str] = ORIGINAL_DART_DIR) -> dict:
    """Compute our distribution stats and, when available, the original's."""
    ours = _load_slice(out_dir)
    orig = _load_slice(original_dir) if original_dir else None
    return {"ours": ours, "original": orig, "original_dir": original_dir}


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
def run_sanity(out_dir: str, ops_dir: str = "ops/f5",
               original_dir: Optional[str] = ORIGINAL_DART_DIR) -> dict:
    """Run all three checks; print a report and save ops/f5/sanity.json."""
    report: dict = {}
    print("\n[sanity] === schema conformance (baseline dataset classes) ===", flush=True)
    report["schema"] = check_schema(out_dir)
    print(f"[sanity] schema ok={report['schema']['ok']}  "
          f"phase_samples={report['schema'].get('n_samples_phase')}  "
          f"ego_rgb_nonzero_frac={report['schema'].get('ego_rgb_nonzero_frac')}")
    for e in report["schema"]["errors"]:
        print(f"[sanity]   ERROR: {e}")

    print("\n[sanity] === contact sheet ===", flush=True)
    png = str(Path(ops_dir) / "contact_sheet.png")
    report["contact_sheet"] = contact_sheet(out_dir, png)
    print(f"[sanity] contact sheet -> {report['contact_sheet']}")

    print("\n[sanity] === distribution stats vs original DART ===", flush=True)
    report["dist"] = distribution_stats(out_dir, original_dir)
    ours, orig = report["dist"]["ours"], report["dist"]["original"]
    if ours:
        cols = ["n_frames", "action_mean_abs", "action_std_mean", "vx_mean", "vx_std",
                "wz_mean", "wz_std", "turn_frac", "moving_frac", "goal_dist_mean",
                "goal_dist_max"]
        print(f"[sanity] {'metric':>16} | {'ours(warehouse)':>16} | {'orig(dart_demo)':>16}")
        for c in cols:
            ov = ours.get(c)
            gv = orig.get(c) if orig else None
            os_ = f"{ov:.4f}" if isinstance(ov, float) else str(ov)
            gs_ = (f"{gv:.4f}" if isinstance(gv, float) else str(gv)) if orig else "n/a"
            print(f"[sanity] {c:>16} | {os_:>16} | {gs_:>16}")

    Path(ops_dir).mkdir(parents=True, exist_ok=True)
    with open(Path(ops_dir) / "sanity.json", "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\n[sanity] saved -> {Path(ops_dir) / 'sanity.json'}", flush=True)
    return report


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--ops-dir", default="ops/f5")
    ap.add_argument("--original-dir", default=ORIGINAL_DART_DIR)
    a = ap.parse_args()
    run_sanity(a.out, a.ops_dir, a.original_dir)
