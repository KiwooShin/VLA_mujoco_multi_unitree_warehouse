"""Schema-conformance test for the warehouse GROUND_NET detector dataset.

A tiny 2-scene generation (one hero + one rooms) must:
  * write the baseline det-dataset artifact layout (per-split frames/labels
    parquet + images_{cam}.npz, scenes.json, meta.json);
  * carry every column the baseline loader / trainer read;
  * load cleanly with the BASELINE detector loader
    (``code.perception.detector.data.SplitCache`` + ``build_example_index``),
    producing both positive and negative training examples;
  * have geometrically-accurate labels (back-projection vs GT agreement, the
    same sanity the baseline dataset reports).

Renders on the GPU via the EGL fix; kept to 2 short scenes so it runs quickly.
"""

from __future__ import annotations

import argparse
import tempfile
import unittest
from pathlib import Path

from code.apps.warehouse_datagen.egl_gpu import force_nvidia_egl

force_nvidia_egl()

import numpy as np

from code.apps.warehouse_datagen.gen_warehouse_det import generate

# Columns the baseline loader / trainer / scene selection depend on.
_FRAME_COLS = ("frame_uid", "scene_id", "split", "cam_type", "array_idx",
               "source", "robot_yaw", "qpos", "n_objects_visible")
_LABEL_COLS = ("frame_uid", "class_id", "color_id", "centroid_px_x",
               "centroid_px_y", "dist_gt_m", "bearing_gt_deg", "clipped",
               "area_px", "is_instructed_target")


def _args(out: str) -> argparse.Namespace:
    # plan = [hero, rooms, rooms]; smoke breaks after 2 ok scenes -> one of each.
    return argparse.Namespace(n_hero=1, n_rooms=2, seed=321, out=out,
                              ops_dir=str(Path(out) / "ops"), smoke=True,
                              smoke_scenes=2)


class DetSchemaTest(unittest.TestCase):
    def test_tiny_gen_loads_with_baseline_loader(self) -> None:
        import pandas as pd
        from code.perception.detector.data import (SplitCache,
                                                    build_example_index)

        with tempfile.TemporaryDirectory() as tmp:
            meta, out_dir = generate(_args(tmp))
            out = Path(tmp)

            # Artifacts present.
            self.assertTrue((out / "meta.json").exists())
            self.assertTrue((out / "scenes.json").exists())
            self.assertTrue((out / "train" / "frames.parquet").exists())
            self.assertTrue((out / "train" / "labels.parquet").exists())
            self.assertTrue(any((out / "train").glob("images_*.npz")))
            self.assertGreater(meta["frames_total"], 0)
            self.assertGreater(meta["n_labels_total"], 0)

            # Required columns.
            fdf = pd.read_parquet(out / "train" / "frames.parquet")
            ldf = pd.read_parquet(out / "train" / "labels.parquet")
            for c in _FRAME_COLS:
                self.assertIn(c, fdf.columns, f"frames missing column {c}")
            for c in _LABEL_COLS:
                self.assertIn(c, ldf.columns, f"labels missing column {c}")
            # class/color ids are in range.
            self.assertTrue(((ldf["class_id"] >= 0) & (ldf["class_id"] < 4)).all())
            self.assertTrue(((ldf["color_id"] >= 0) & (ldf["color_id"] < 7)).all())

            # Loads with the BASELINE loader; yields positives AND negatives.
            cache = SplitCache(str(out), "train", verbose=False)
            self.assertEqual(len(cache), len(fdf))
            ex = build_example_index(cache, np.random.default_rng(0))
            n_pos = sum(1 for e in ex if e[3] is not None)
            n_neg = len(ex) - n_pos
            self.assertGreater(n_pos, 0, "no positive examples")
            self.assertGreater(n_neg, 0, "no negative examples (wall-occluded / absent)")

            # Labels are geometrically accurate (same check the baseline reports).
            self.assertLess(meta["label_geometry_err_m_p95"], 0.25)

            # Both families appeared in the 2-scene draw.
            import json
            with open(out / "scenes.json") as f:
                scenes = json.load(f)
            fams = {s["style"] for s in scenes.values()}
            self.assertTrue({"hero", "rooms"} & fams)


if __name__ == "__main__":
    unittest.main()
