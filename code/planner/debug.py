"""Debug visualization helpers for the grid planner (optional cv2 dependency).

:func:`render_grid_png` rasterizes an occupancy grid plus a planned path to a
PNG for eyeballing A* output. cv2 is imported lazily; callers/tests should skip
when :data:`HAVE_CV2` is False. Running this module as a script builds a small
demo maze, plans a path across it and writes ``ops/phase1/planner_maze.png``.
"""

from __future__ import annotations

from typing import Optional, Sequence, Tuple

import numpy as np

from code.planner.grid import OccupancyGrid

try:  # cv2 is only needed for debug rendering.
    import cv2  # type: ignore

    HAVE_CV2: bool = True
except ImportError:  # pragma: no cover - environment dependent
    cv2 = None  # type: ignore
    HAVE_CV2 = False

Point = Tuple[float, float]


def render_grid_png(
    og: OccupancyGrid,
    path: Optional[Sequence[Point]],
    out_path: str,
    *,
    scale: int = 6,
) -> str:
    """Renders an occupancy grid and an optional path to a PNG file.

    The image is drawn world-y-up (row 0 at the bottom). Occupied cells are dark
    gray, free cells white; the path is a green polyline with a green start dot
    and a red goal dot.

    Args:
        og: Occupancy grid to render.
        path: World-coordinate waypoints to overlay, or None.
        out_path: Destination PNG path.
        scale: Integer pixel magnification per cell (>= 1).

    Returns:
        ``out_path``.

    Raises:
        RuntimeError: If cv2 is not importable.
        ValueError: If ``scale`` < 1.
    """
    if not HAVE_CV2:
        raise RuntimeError("cv2 is required for render_grid_png but is not installed")
    if scale < 1:
        raise ValueError(f"scale must be >= 1, got {scale}")

    ny, nx = og.grid.shape
    base = np.full((ny, nx, 3), 255, dtype=np.uint8)
    base[og.grid] = (40, 40, 40)  # BGR dark gray for occupied
    canvas = cv2.resize(
        base, (nx * scale, ny * scale), interpolation=cv2.INTER_NEAREST
    )

    if path:
        pix = []
        for x, y in path:
            ix = int(round((x - og.origin_xy[0]) / og.resolution))
            iy = int(round((y - og.origin_xy[1]) / og.resolution))
            ix = min(max(ix, 0), nx - 1)
            iy = min(max(iy, 0), ny - 1)
            pix.append((ix * scale + scale // 2, iy * scale + scale // 2))
        pts = np.asarray(pix, dtype=np.int32)
        if len(pts) >= 2:
            cv2.polylines(canvas, [pts], False, (0, 180, 0), max(1, scale // 3))
        cv2.circle(canvas, tuple(pts[0]), scale, (0, 200, 0), -1)
        cv2.circle(canvas, tuple(pts[-1]), scale, (0, 0, 220), -1)

    canvas = cv2.flip(canvas, 0)  # world-y-up
    if not cv2.imwrite(out_path, canvas):
        raise RuntimeError(f"cv2.imwrite failed for {out_path}")
    return out_path


def _demo() -> str:
    """Builds a demo maze, plans a path and writes the example PNG.

    Returns:
        The written PNG path.
    """
    import os

    from code.planner.astar import plan_path, shortcut_path

    ny, nx = 120, 160  # 12 m x 16 m at 0.1 m
    grid = np.zeros((ny, nx), dtype=np.bool_)
    grid[:, 0] = grid[:, -1] = grid[0, :] = grid[-1, :] = True  # perimeter
    # Two shelf rows with a mid gap (racetrack topology).
    grid[30:34, 20:70] = True
    grid[30:34, 90:140] = True
    grid[86:90, 20:70] = True
    grid[86:90, 90:140] = True
    # An L-shaped alcove partition in the upper-right.
    grid[55:100, 120:124] = True
    grid[55:59, 120:150] = True

    og = OccupancyGrid(grid, 0.1, (0.0, 0.0))
    start = og.cell_to_world((10, 10))
    goal = og.cell_to_world((110, 145))
    path = shortcut_path(og, plan_path(og, start, goal))

    out_dir = os.path.join("ops", "phase1")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "planner_maze.png")
    return render_grid_png(og, path, out_path)


if __name__ == "__main__":  # pragma: no cover
    print(_demo())
