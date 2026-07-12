"""Unit tests for code.warehouse.occupancy (wall rasterization to a grid)."""

import math
import unittest

from code.planner.grid import OccupancyGrid
from code.warehouse.layout import WallSpec, WarehouseLayout, hero_layout
from code.warehouse.occupancy import occupancy_grid


class TestGridShapeAndFrame(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.layout = hero_layout()
        cls.og = occupancy_grid(cls.layout, resolution=0.1)

    def test_returns_occupancygrid(self) -> None:
        self.assertIsInstance(self.og, OccupancyGrid)

    def test_covers_full_hall(self) -> None:
        # 16 x 12 m hall at 0.1 m -> 160 x 120 cells (ny, nx).
        self.assertEqual(self.og.shape, (120, 160))
        self.assertEqual(self.og.resolution, 0.1)

    def test_origin_at_first_cell_center(self) -> None:
        ox, oy = self.og.origin_xy
        self.assertAlmostEqual(ox, -7.95)
        self.assertAlmostEqual(oy, -5.95)

    def test_resolution_scales_shape(self) -> None:
        og2 = occupancy_grid(self.layout, resolution=0.2)
        self.assertEqual(og2.shape, (60, 80))

    def test_invalid_resolution_raises(self) -> None:
        with self.assertRaises(ValueError):
            occupancy_grid(self.layout, resolution=0.0)


class TestRasterizationCorrectness(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.layout = hero_layout()
        cls.og = occupancy_grid(cls.layout, resolution=0.1)

    def test_shelf_center_occupied(self) -> None:
        # A hero shelf block centre (row A west block ~ (-1.475, 2.0)).
        self.assertFalse(self.og.is_free((-1.475, 2.0)))

    def test_aisle_and_crossover_free(self) -> None:
        self.assertTrue(self.og.is_free((0.0, 0.0)))    # central middle aisle
        self.assertTrue(self.og.is_free((0.0, 2.0)))    # mid-row crossover gap
        self.assertTrue(self.og.is_free((-5.0, -5.0)))  # Alpha home bay

    def test_perimeter_border_occupied(self) -> None:
        # Perimeter walls straddle the hall edge -> the outer cell ring is solid.
        self.assertTrue(self.og.grid[:, -1].all())   # east wall column
        self.assertTrue(self.og.grid[:, 0].all())    # west wall column
        self.assertTrue(self.og.grid[-1, :].all())   # north wall row
        self.assertTrue(self.og.grid[0, :].all())    # south wall row

    def test_occupied_fraction_reasonable(self) -> None:
        frac = self.og.grid.mean()
        self.assertGreater(frac, 0.02)  # walls present
        self.assertLess(frac, 0.30)     # hall still mostly open


class TestYawedWallRasterization(unittest.TestCase):
    """A 45-deg-yawed thin wall must be rasterized along its rotated long axis."""

    @classmethod
    def setUpClass(cls) -> None:
        wall = WallSpec(0.0, 0.0, 1.5, 0.1, yaw=math.pi / 4.0, name="diag")
        cls.layout = WarehouseLayout(hall_x=4.0, hall_y=4.0, walls=[wall])
        cls.og = occupancy_grid(cls.layout, resolution=0.1)

    def test_point_on_rotated_long_axis_occupied(self) -> None:
        # (0.7, 0.7) lies along the +45 deg long axis, well inside half-length.
        self.assertFalse(self.og.is_free((0.7, 0.7)))

    def test_point_off_axis_free(self) -> None:
        # (0.6, -0.6) lies along the short (perpendicular) axis -> outside.
        self.assertTrue(self.og.is_free((0.6, -0.6)))

    def test_point_beyond_long_axis_free(self) -> None:
        # (1.4, 1.4): distance ~1.98 m > 1.5 m half-length -> off the wall end.
        self.assertTrue(self.og.is_free((1.4, 1.4)))


if __name__ == "__main__":
    unittest.main()
