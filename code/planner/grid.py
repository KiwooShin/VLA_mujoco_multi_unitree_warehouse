"""Occupancy-grid contract shared by the warehouse and planner packages.

The warehouse layout (``code.warehouse``) rasterizes its wall list into an
:class:`OccupancyGrid`; the A* planner (``code.planner.astar``) consumes it.
Both sides build against this one type so the simulated geometry and the
planning world can never skew (single source of truth).

Frame convention: world coordinates are hall-centered (x right, y up in
top-down view). ``grid[iy, ix]`` is True where occupied. ``origin_xy`` is the
world position of the CENTER of cell ``(iy=0, ix=0)``.
"""

from __future__ import annotations

import dataclasses
from typing import Tuple

import numpy as np


@dataclasses.dataclass(frozen=True)
class OccupancyGrid:
    """A 2-D boolean occupancy grid in world coordinates.

    Attributes:
        grid: Bool array of shape (ny, nx); True means occupied.
        resolution: Cell edge length in meters.
        origin_xy: World (x, y) of the center of cell (iy=0, ix=0).
    """

    grid: np.ndarray
    resolution: float
    origin_xy: Tuple[float, float]

    def __post_init__(self) -> None:
        if self.grid.ndim != 2 or self.grid.dtype != np.bool_:
            raise ValueError(
                f"grid must be 2-D bool, got shape={self.grid.shape} "
                f"dtype={self.grid.dtype}"
            )
        if self.resolution <= 0.0:
            raise ValueError(f"resolution must be > 0, got {self.resolution}")

    @property
    def shape(self) -> Tuple[int, int]:
        """(ny, nx) cell counts."""
        return self.grid.shape  # type: ignore[return-value]

    def world_to_cell(self, xy: Tuple[float, float]) -> Tuple[int, int]:
        """Maps world (x, y) to the nearest cell index (iy, ix).

        Args:
            xy: World coordinates in meters.

        Returns:
            (iy, ix) cell index of the nearest cell center.

        Raises:
            ValueError: If xy falls outside the grid bounds.
        """
        ix = int(round((xy[0] - self.origin_xy[0]) / self.resolution))
        iy = int(round((xy[1] - self.origin_xy[1]) / self.resolution))
        ny, nx = self.grid.shape
        if not (0 <= ix < nx and 0 <= iy < ny):
            raise ValueError(f"world point {xy} outside grid ({ny}x{nx})")
        return iy, ix

    def cell_to_world(self, iy_ix: Tuple[int, int]) -> Tuple[float, float]:
        """Maps a cell index (iy, ix) to the world (x, y) of its center."""
        iy, ix = iy_ix
        return (
            self.origin_xy[0] + ix * self.resolution,
            self.origin_xy[1] + iy * self.resolution,
        )

    def is_free(self, xy: Tuple[float, float]) -> bool:
        """True if the world point lies in bounds and its cell is unoccupied."""
        try:
            iy, ix = self.world_to_cell(xy)
        except ValueError:
            return False
        return not bool(self.grid[iy, ix])


def inflate(og: OccupancyGrid, radius_m: float) -> OccupancyGrid:
    """Returns a copy with occupied cells dilated by a disk of radius_m.

    Used to add robot-body clearance so A* paths keep the pelvis center at
    least radius_m away from any wall.

    Args:
        og: Source grid.
        radius_m: Dilation radius in meters (>= 0).

    Returns:
        A new OccupancyGrid with the same frame and dilated occupancy.

    Raises:
        ValueError: If radius_m is negative.
    """
    if radius_m < 0.0:
        raise ValueError(f"radius_m must be >= 0, got {radius_m}")
    r_cells = int(np.ceil(radius_m / og.resolution))
    if r_cells == 0:
        return OccupancyGrid(og.grid.copy(), og.resolution, og.origin_xy)
    yy, xx = np.mgrid[-r_cells : r_cells + 1, -r_cells : r_cells + 1]
    disk = (yy * yy + xx * xx) * (og.resolution**2) <= radius_m**2 + 1e-9
    occ_iy, occ_ix = np.nonzero(og.grid)
    ny, nx = og.grid.shape
    out = np.zeros_like(og.grid)
    for dy, dx in zip(*np.nonzero(disk)):
        sy, sx = int(dy) - r_cells, int(dx) - r_cells
        ys = np.clip(occ_iy + sy, 0, ny - 1)
        xs = np.clip(occ_ix + sx, 0, nx - 1)
        out[ys, xs] = True
    return OccupancyGrid(out, og.resolution, og.origin_xy)
