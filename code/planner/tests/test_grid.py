"""Unit tests for the code.planner.grid contract (OccupancyGrid + inflate).

Covers world/cell round-trips, bounds handling, is_free semantics, inflate disk
correctness (cell counts + clearance guarantee), and degenerate-input errors.
"""

from __future__ import annotations

import math
import unittest

import numpy as np

from code.planner.grid import OccupancyGrid, inflate


def _empty(ny: int = 20, nx: int = 30, res: float = 0.1,
           origin=(0.0, 0.0)) -> OccupancyGrid:
    return OccupancyGrid(np.zeros((ny, nx), dtype=np.bool_), res, origin)


class WorldCellRoundTripTest(unittest.TestCase):
    def test_cell_to_world_to_cell_identity(self) -> None:
        og = _empty(origin=(-1.5, 2.0))
        for iy in range(0, 20, 3):
            for ix in range(0, 30, 5):
                xy = og.cell_to_world((iy, ix))
                self.assertEqual(og.world_to_cell(xy), (iy, ix))

    def test_world_to_cell_rounds_to_nearest_center(self) -> None:
        og = _empty(origin=(0.0, 0.0))  # res 0.1
        # A point 0.03 m off cell (5, 7) center still rounds to it.
        cx, cy = og.cell_to_world((5, 7))
        self.assertEqual(og.world_to_cell((cx + 0.03, cy - 0.02)), (5, 7))

    def test_origin_is_cell_zero_center(self) -> None:
        og = _empty(origin=(3.0, -4.0))
        self.assertEqual(og.cell_to_world((0, 0)), (3.0, -4.0))


class BoundsTest(unittest.TestCase):
    def test_world_to_cell_out_of_bounds_raises(self) -> None:
        og = _empty(ny=10, nx=10, origin=(0.0, 0.0))
        with self.assertRaises(ValueError):
            og.world_to_cell((-1.0, 0.0))
        with self.assertRaises(ValueError):
            og.world_to_cell((0.0, 5.0))  # beyond ny*res

    def test_is_free_out_of_bounds_is_false(self) -> None:
        og = _empty(ny=10, nx=10)
        self.assertFalse(og.is_free((-1.0, -1.0)))
        self.assertFalse(og.is_free((100.0, 100.0)))


class IsFreeTest(unittest.TestCase):
    def test_free_and_occupied(self) -> None:
        g = np.zeros((10, 10), dtype=np.bool_)
        g[4, 6] = True
        og = OccupancyGrid(g, 0.1, (0.0, 0.0))
        self.assertTrue(og.is_free(og.cell_to_world((0, 0))))
        self.assertFalse(og.is_free(og.cell_to_world((4, 6))))


class InflateTest(unittest.TestCase):
    def test_radius_zero_is_identity_copy(self) -> None:
        g = np.zeros((10, 10), dtype=np.bool_)
        g[5, 5] = True
        og = OccupancyGrid(g, 0.1, (0.0, 0.0))
        out = inflate(og, 0.0)
        self.assertIsNot(out.grid, og.grid)  # copy, not alias
        np.testing.assert_array_equal(out.grid, og.grid)

    def test_single_cell_disk_count(self) -> None:
        # One occupied cell centered far from borders; count matches the disk.
        res = 0.1
        radius = 0.25
        g = np.zeros((41, 41), dtype=np.bool_)
        g[20, 20] = True
        og = OccupancyGrid(g, res, (0.0, 0.0))
        out = inflate(og, radius)

        r_cells = int(math.ceil(radius / res))
        expected = 0
        for dy in range(-r_cells, r_cells + 1):
            for dx in range(-r_cells, r_cells + 1):
                if (dy * dy + dx * dx) * res * res <= radius * radius + 1e-9:
                    expected += 1
        self.assertEqual(int(out.grid.sum()), expected)

    def test_clearance_guarantee(self) -> None:
        # Every free cell within `radius` of a wall must become occupied.
        res = 0.1
        radius = 0.3
        g = np.zeros((40, 40), dtype=np.bool_)
        g[:, 20] = True  # vertical wall
        og = OccupancyGrid(g, res, (0.0, 0.0))
        out = inflate(og, radius)
        occ_iy, occ_ix = np.nonzero(g)
        r_cells = int(math.ceil(radius / res))
        ny, nx = g.shape
        for iy in range(ny):
            for ix in range(nx):
                # Distance to nearest wall cell (column 20).
                d = abs(ix - 20) * res
                if d <= radius - 1e-9:
                    self.assertTrue(
                        out.grid[iy, ix],
                        f"cell ({iy},{ix}) at {d:.2f} m should be inflated",
                    )
        # Sanity: a cell well beyond the radius stays free.
        self.assertFalse(out.grid[0, 0])
        self.assertEqual((r_cells, ny, nx), (3, 40, 40))

    def test_preserves_frame(self) -> None:
        og = _empty(origin=(1.0, -2.0))
        out = inflate(og, 0.2)
        self.assertEqual(out.resolution, og.resolution)
        self.assertEqual(out.origin_xy, og.origin_xy)
        self.assertEqual(out.shape, og.shape)


class DegenerateInputTest(unittest.TestCase):
    def test_non_bool_dtype_raises(self) -> None:
        with self.assertRaises(ValueError):
            OccupancyGrid(np.zeros((5, 5), dtype=np.float32), 0.1, (0.0, 0.0))

    def test_wrong_ndim_raises(self) -> None:
        with self.assertRaises(ValueError):
            OccupancyGrid(np.zeros((5,), dtype=np.bool_), 0.1, (0.0, 0.0))

    def test_bad_resolution_raises(self) -> None:
        with self.assertRaises(ValueError):
            OccupancyGrid(np.zeros((5, 5), dtype=np.bool_), 0.0, (0.0, 0.0))
        with self.assertRaises(ValueError):
            OccupancyGrid(np.zeros((5, 5), dtype=np.bool_), -0.1, (0.0, 0.0))

    def test_negative_inflate_radius_raises(self) -> None:
        with self.assertRaises(ValueError):
            inflate(_empty(), -0.1)


if __name__ == "__main__":
    unittest.main()
