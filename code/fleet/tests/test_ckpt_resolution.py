"""Checkpoint-resolution-order test for the fleet perception bridge.

Cycle-2b makes the warehouse-domain fine-tune the default GROUND_NET weights
WHEN they exist, while keeping the original playground baseline as the fallback
and letting ``GROUND_NET_CKPT`` override both. This pins that precedence:

    env var  >  runs/nx6_warehouse_ft/model_best.pt  >  runs/nx6_heatmap_B/...

Pure-logic (no torch / mujoco context / weights): it drives
``resolve_ckpt_path`` with temp files + a patched env var only.
"""

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

from code.fleet.perception_bridge import resolve_ckpt_path


class CkptResolutionOrderTest(unittest.TestCase):
    def setUp(self) -> None:
        self._saved = os.environ.pop("GROUND_NET_CKPT", None)

    def tearDown(self) -> None:
        if self._saved is not None:
            os.environ["GROUND_NET_CKPT"] = self._saved
        else:
            os.environ.pop("GROUND_NET_CKPT", None)

    def test_env_var_wins_even_if_missing(self) -> None:
        os.environ["GROUND_NET_CKPT"] = "/some/explicit/override.pt"
        with tempfile.TemporaryDirectory() as tmp:
            ft = Path(tmp) / "ft.pt"
            ft.write_bytes(b"x")  # exists, but env var must still win
            got = resolve_ckpt_path(ft_ckpt=str(ft), orig_ckpt=str(Path(tmp) / "orig.pt"))
        self.assertEqual(got, "/some/explicit/override.pt")

    def test_finetune_used_when_present(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            ft = Path(tmp) / "ft.pt"
            orig = Path(tmp) / "orig.pt"
            ft.write_bytes(b"x")
            orig.write_bytes(b"x")
            got = resolve_ckpt_path(ft_ckpt=str(ft), orig_ckpt=str(orig))
        self.assertEqual(got, str(ft))

    def test_falls_back_to_original_when_no_finetune(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            ft = Path(tmp) / "ft.pt"        # deliberately NOT created
            orig = Path(tmp) / "orig.pt"
            orig.write_bytes(b"x")
            got = resolve_ckpt_path(ft_ckpt=str(ft), orig_ckpt=str(orig))
        self.assertEqual(got, str(orig))

    def test_default_paths_point_at_expected_runs(self) -> None:
        # The import-time snapshot resolves to a path under runs/ (either the
        # fine-tune or the baseline, depending on what exists in this checkout).
        from code.fleet.perception_bridge import (_CKPT_DEFAULT,
                                                  _ORIGINAL_CKPT,
                                                  _WAREHOUSE_FT_CKPT)
        self.assertTrue(_WAREHOUSE_FT_CKPT.endswith(
            os.path.join("runs", "nx6_warehouse_ft", "model_best.pt")))
        self.assertTrue(_ORIGINAL_CKPT.endswith(
            os.path.join("runs", "nx6_heatmap_B", "model_best.pt")))
        self.assertIn(_CKPT_DEFAULT, (_WAREHOUSE_FT_CKPT, _ORIGINAL_CKPT))


if __name__ == "__main__":
    unittest.main()
