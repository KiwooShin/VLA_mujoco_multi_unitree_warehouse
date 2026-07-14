"""Layout math: non-overlapping regions, adaptive tiles (n=4/6), fit helpers."""

from __future__ import annotations

import itertools
import unittest

from code.apps.demos import style
from code.apps.demos.layout import (Rect, contain_fit, ego_area, overlaps,
                                    regions, tile_rects)


class TestRegions(unittest.TestCase):
    def test_regions_are_non_overlapping_and_in_canvas(self):
        reg = regions()
        canvas = Rect(0, 0, style.CANVAS_W, style.CANVAS_H)
        self.assertEqual(set(reg), {"bev", "panel", "strip"})
        for r in reg.values():
            self.assertTrue(canvas.contains_rect(r))
        for a, b in itertools.combinations(reg.values(), 2):
            self.assertFalse(overlaps(a, b))

    def test_regions_tile_left_column_and_full_height_panel(self):
        reg = regions()
        self.assertEqual(reg["panel"].h, style.CANVAS_H)          # full-height panel
        self.assertEqual(reg["panel"].x1, style.CANVAS_W)
        self.assertEqual(reg["bev"].x, 0)
        self.assertEqual(reg["strip"].x, 0)
        self.assertEqual(reg["bev"].y, 0)
        self.assertEqual(reg["bev"].y1, reg["strip"].y)           # stacked, no gap
        self.assertEqual(reg["strip"].y1, style.CANVAS_H)
        self.assertEqual(reg["bev"].x1, reg["panel"].x)


class TestTiles(unittest.TestCase):
    def test_tiles_fit_strip_and_do_not_overlap(self):
        strip = regions()["strip"]
        for n in (4, 6):
            with self.subTest(n=n):
                tiles = tile_rects(strip, n)
                self.assertEqual(len(tiles), n)
                for t in tiles:
                    self.assertTrue(strip.contains_rect(t))
                    self.assertGreater(t.w, 0)
                    self.assertGreater(t.h, 0)
                    self.assertTrue(strip.contains_rect(ego_area(t)))
                for a, b in zip(tiles, tiles[1:]):
                    self.assertLessEqual(a.x1, b.x)

    def test_more_robots_yield_smaller_tiles(self):
        strip = regions()["strip"]
        self.assertLess(tile_rects(strip, 6)[0].w, tile_rects(strip, 4)[0].w)

    def test_tile_rects_rejects_bad_counts(self):
        strip = regions()["strip"]
        with self.assertRaises(ValueError):
            tile_rects(strip, 0)


class TestContainFit(unittest.TestCase):
    def test_contain_fit_preserves_aspect_and_centers(self):
        off_x, off_y, w, h = contain_fit(960, 600, 1000, 800)
        self.assertAlmostEqual(w / h, 960 / 600, delta=0.02)
        self.assertLessEqual(w, 1000)
        self.assertLessEqual(h, 800)
        self.assertTrue(w == 1000 or h == 800)     # one dimension maxed
        self.assertGreaterEqual(off_x, 0)
        self.assertGreaterEqual(off_y, 0)

    def test_contain_fit_rejects_bad_source(self):
        with self.assertRaises(ValueError):
            contain_fit(0, 100, 50, 50)


if __name__ == "__main__":
    unittest.main()
