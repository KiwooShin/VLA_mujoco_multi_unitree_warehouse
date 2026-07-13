"""End-to-end schema-conformance test: a tiny 2-episode generation must load
cleanly with the BASELINE dataset classes and pass the sanity schema check.

This exercises the full pipeline (scene -> rollout -> parquet + ego mp4 -> meta
assembly -> PhaseParquetDataset / ParquetDataset). Kept small (short episodes)
so it runs in well under a minute; renders on the GPU via the EGL fix.
"""

from __future__ import annotations

import argparse
import tempfile
import unittest
from pathlib import Path

from code.apps.warehouse_datagen.egl_gpu import force_nvidia_egl
force_nvidia_egl()

from code.apps.warehouse_datagen.gen_warehouse_dart import main_generate
from code.apps.warehouse_datagen.sanity import check_schema


def _args(out: str) -> argparse.Namespace:
    return argparse.Namespace(
        episodes=2, seed=123, out=out, noise=0.07, maxsteps=60,
        primitive_frac=0.5, chunk_size=0, no_render=False, workers=1, verbose=False,
    )


class SchemaConformanceTest(unittest.TestCase):
    def test_two_episode_dataset_loads_with_baseline_classes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            main_generate(_args(tmp))

            out = Path(tmp)
            self.assertTrue((out / "meta" / "info.json").exists())
            self.assertTrue((out / "meta" / "episodes.jsonl").exists())
            parqs = list((out / "data" / "chunk-000").glob("episode_*.parquet"))
            self.assertGreaterEqual(len(parqs), 1)
            vids = list((out / "videos").glob("episode_*_ego.mp4"))
            self.assertGreaterEqual(len(vids), 1)

            # exact baseline DART columns present
            import pandas as pd
            df = pd.read_parquet(parqs[0])
            for col in ("frame_index", "episode_index", "index", "task_index",
                        "timestamp", "proprio", "action", "goal", "vel_cmd", "done",
                        "task_description", "phase"):
                self.assertIn(col, df.columns, f"missing column {col}")
            self.assertEqual(len(df["proprio"].iloc[0]), 55)
            self.assertEqual(len(df["action"].iloc[0]), 15)
            self.assertEqual(len(df["phase"].iloc[0]), 2)

            report = check_schema(tmp)
            self.assertTrue(report["ok"], f"schema errors: {report['errors']}")
            self.assertIsNotNone(report.get("ego_rgb_nonzero_frac"))
            self.assertGreater(report["ego_rgb_nonzero_frac"], 0.5)


if __name__ == "__main__":
    unittest.main()
