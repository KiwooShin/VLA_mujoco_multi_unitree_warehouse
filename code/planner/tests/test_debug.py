"""Unit tests for code.planner.debug.render_grid_png (skipped without cv2)."""

from __future__ import annotations

import os
import tempfile
import unittest

import numpy as np

from code.planner import debug
from code.planner.astar import plan_path
from code.planner.grid import OccupancyGrid


@unittest.skipUnless(debug.HAVE_CV2, "cv2 not installed")
class RenderGridPngTest(unittest.TestCase):
    def _og(self) -> OccupancyGrid:
        g = np.zeros((20, 30), dtype=np.bool_)
        g[8:12, 10:20] = True
        return OccupancyGrid(g, 0.1, (0.0, 0.0))

    def test_writes_png_with_path(self) -> None:
        og = self._og()
        path = plan_path(og, og.cell_to_world((2, 2)), og.cell_to_world((17, 27)))
        with tempfile.TemporaryDirectory() as d:
            out = os.path.join(d, "maze.png")
            ret = debug.render_grid_png(og, path, out)
            self.assertEqual(ret, out)
            self.assertTrue(os.path.exists(out))
            self.assertGreater(os.path.getsize(out), 0)

    def test_writes_png_without_path(self) -> None:
        og = self._og()
        with tempfile.TemporaryDirectory() as d:
            out = os.path.join(d, "grid.png")
            debug.render_grid_png(og, None, out)
            self.assertTrue(os.path.exists(out))

    def test_bad_scale_raises(self) -> None:
        og = self._og()
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(ValueError):
                debug.render_grid_png(og, None, os.path.join(d, "x.png"), scale=0)


if __name__ == "__main__":
    unittest.main()
