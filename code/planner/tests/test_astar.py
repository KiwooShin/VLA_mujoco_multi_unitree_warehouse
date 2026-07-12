"""Unit tests for code.planner.astar (grid A*, no corner cut, smoothing).

Synthetic mazes cover: open-grid straight lines, a wall with a single gap,
unreachable goals, the no-corner-cutting invariant (diagonal-gap traps),
endpoint snapping, supercover-clean smoothing that never lengthens the path,
and determinism.
"""

from __future__ import annotations

import math
import unittest

import numpy as np

from code.planner.astar import (
    PathNotFoundError,
    _raw_cell,
    _segment_clear,
    _supercover_cells,
    path_length,
    plan_path,
    shortcut_path,
)
from code.planner.grid import OccupancyGrid


def _grid(ny: int, nx: int, res: float = 0.1) -> np.ndarray:
    return np.zeros((ny, nx), dtype=np.bool_)


def _og(g: np.ndarray, res: float = 0.1) -> OccupancyGrid:
    return OccupancyGrid(g, res, (0.0, 0.0))


def _cells(og: OccupancyGrid, path):
    return [og.world_to_cell(p) for p in path]


def _assert_path_in_free(og: OccupancyGrid, path) -> None:
    for p in path:
        assert og.is_free(p), f"waypoint {p} not free"


def _assert_no_corner_cut(og: OccupancyGrid, path) -> None:
    """Every diagonal step must have both shared cardinal cells free."""
    free = ~og.grid
    cells = _cells(og, path)
    for (ay, ax), (by, bx) in zip(cells, cells[1:]):
        diy, dix = by - ay, bx - ax
        assert max(abs(diy), abs(dix)) == 1, f"non-unit step {(ay,ax)}->{(by,bx)}"
        if diy != 0 and dix != 0:
            assert free[ay + diy, ax] and free[ay, ax + dix], (
                f"corner cut on diagonal {(ay,ax)}->{(by,bx)}"
            )


class SupercoverTest(unittest.TestCase):
    def test_pure_diagonal_includes_both_straddled_cells(self) -> None:
        cells = _supercover_cells((0, 0), (2, 2))
        # Conservative supercover: both off-diagonal cells appear at each corner.
        for c in [(0, 0), (0, 1), (1, 0), (1, 1), (1, 2), (2, 1), (2, 2)]:
            self.assertIn(c, cells)

    def test_horizontal_line(self) -> None:
        cells = _supercover_cells((3, 0), (3, 4))
        self.assertEqual(cells, [(3, 0), (3, 1), (3, 2), (3, 3), (3, 4)])

    def test_single_cell(self) -> None:
        self.assertEqual(_supercover_cells((2, 2), (2, 2)), [(2, 2)])


class StraightLineTest(unittest.TestCase):
    def test_open_grid_diagonal(self) -> None:
        og = _og(_grid(30, 40))
        start = og.cell_to_world((2, 2))
        goal = og.cell_to_world((25, 35))
        path = plan_path(og, start, goal)
        self.assertEqual(path[0], start)
        self.assertEqual(path[-1], goal)
        _assert_path_in_free(og, path)
        _assert_no_corner_cut(og, path)
        # Length is the octile optimum for an open grid.
        dy, dx = 23, 33
        expected = (dx - dy) + math.sqrt(2.0) * dy  # in cells
        self.assertAlmostEqual(path_length(path), expected * og.resolution, places=6)

    def test_same_cell_is_single_waypoint(self) -> None:
        og = _og(_grid(10, 10))
        p = og.cell_to_world((4, 4))
        path = plan_path(og, p, p)
        self.assertEqual(path, [p])
        self.assertEqual(path_length(path), 0.0)


class WallGapTest(unittest.TestCase):
    def test_single_gap_forces_route_through_gap(self) -> None:
        g = _grid(40, 40)
        g[:, 20] = True
        g[20, 20] = False  # single gap
        og = _og(g)
        path = plan_path(og, og.cell_to_world((20, 5)), og.cell_to_world((20, 35)))
        _assert_path_in_free(og, path)
        _assert_no_corner_cut(og, path)
        # The only free cell in column 20 is the gap (20, 20).
        crossings = [(iy, ix) for (iy, ix) in _cells(og, path) if ix == 20]
        self.assertTrue(crossings)
        for c in crossings:
            self.assertEqual(c, (20, 20))


class UnreachableTest(unittest.TestCase):
    def test_sealed_wall_raises(self) -> None:
        g = _grid(30, 30)
        g[:, 15] = True  # full-height wall, no gap
        og = _og(g)
        with self.assertRaises(PathNotFoundError):
            plan_path(og, og.cell_to_world((15, 5)), og.cell_to_world((15, 25)))

    def test_error_message_has_stats(self) -> None:
        g = _grid(20, 20)
        g[:, 10] = True
        og = _og(g)
        with self.assertRaises(PathNotFoundError) as ctx:
            plan_path(og, og.cell_to_world((10, 3)), og.cell_to_world((10, 17)))
        msg = str(ctx.exception)
        self.assertIn("free cells", msg)


class NoCornerCutTest(unittest.TestCase):
    def test_diagonal_pinch_trap_is_unreachable(self) -> None:
        # Center (2,2) blocked on all four cardinals; reachable only by a
        # corner-cutting diagonal, which the planner forbids -> unreachable.
        g = _grid(5, 5)
        for c in [(1, 2), (3, 2), (2, 1), (2, 3)]:
            g[c] = True
        og = _og(g)
        with self.assertRaises(PathNotFoundError):
            plan_path(og, og.cell_to_world((0, 0)), og.cell_to_world((2, 2)))

    def test_enters_pinched_cell_via_open_cardinal(self) -> None:
        # Center blocked on 3 cardinals; the one open cardinal (left) is the
        # only legal entrance, so the final step must be cardinal, not diagonal.
        g = _grid(5, 5)
        for c in [(1, 2), (3, 2), (2, 3)]:  # top, bottom, right blocked
            g[c] = True
        og = _og(g)
        path = plan_path(og, og.cell_to_world((2, 0)), og.cell_to_world((2, 2)))
        _assert_no_corner_cut(og, path)
        cells = _cells(og, path)
        self.assertEqual(cells[-2:], [(2, 1), (2, 2)])


class SnappingTest(unittest.TestCase):
    def test_free_start_not_moved(self) -> None:
        og = _og(_grid(20, 20))
        start = og.cell_to_world((5, 5))
        goal = og.cell_to_world((5, 15))
        path = plan_path(og, start, goal)
        self.assertEqual(path[0], start)

    def test_occupied_start_snaps(self) -> None:
        g = _grid(20, 20)
        g[5, 5] = True
        og = _og(g)
        start = og.cell_to_world((5, 5))  # occupied
        goal = og.cell_to_world((5, 15))
        path = plan_path(og, start, goal)
        self.assertTrue(og.is_free(path[0]))
        d = math.hypot(path[0][0] - start[0], path[0][1] - start[1])
        self.assertLessEqual(d, 0.3 + 1e-9)

    def test_occupied_goal_snaps(self) -> None:
        g = _grid(20, 20)
        g[5, 15] = True
        og = _og(g)
        path = plan_path(og, og.cell_to_world((5, 5)), og.cell_to_world((5, 15)))
        self.assertTrue(og.is_free(path[-1]))

    def test_deep_occupied_start_raises(self) -> None:
        # Start buried in a thick block; nearest free is > 0.3 m away.
        g = _grid(20, 20)
        g[3:12, 3:12] = True
        og = _og(g)  # res 0.1 -> center is 0.4 m from block edge
        with self.assertRaises(PathNotFoundError):
            plan_path(og, og.cell_to_world((7, 7)), og.cell_to_world((5, 15)))


class ShortcutTest(unittest.TestCase):
    def _maze(self) -> OccupancyGrid:
        g = _grid(40, 60)
        g[10:14, 10:40] = True
        g[26:30, 20:50] = True
        return _og(g)

    def test_shortcut_stays_collision_free(self) -> None:
        og = self._maze()
        path = plan_path(og, og.cell_to_world((3, 3)), og.cell_to_world((37, 55)))
        sp = shortcut_path(og, path)
        free = ~og.grid
        for a, b in zip(sp, sp[1:]):
            self.assertTrue(
                _segment_clear(free, _raw_cell(og, a), _raw_cell(og, b)),
                f"smoothed segment {a}->{b} crosses an occupied cell",
            )

    def test_shortcut_never_lengthens(self) -> None:
        og = self._maze()
        path = plan_path(og, og.cell_to_world((3, 3)), og.cell_to_world((37, 55)))
        sp = shortcut_path(og, path)
        self.assertLessEqual(len(sp), len(path))
        self.assertLessEqual(path_length(sp), path_length(path) + 1e-9)
        self.assertEqual(sp[0], tuple(path[0]))
        self.assertEqual(sp[-1], tuple(path[-1]))

    def test_shortcut_open_grid_is_two_points(self) -> None:
        og = _og(_grid(30, 30))
        path = plan_path(og, og.cell_to_world((2, 2)), og.cell_to_world((25, 25)))
        sp = shortcut_path(og, path)
        self.assertEqual(len(sp), 2)

    def test_shortcut_short_path_passthrough(self) -> None:
        og = _og(_grid(10, 10))
        one = [og.cell_to_world((3, 3))]
        self.assertEqual(shortcut_path(og, one), one)


class DeterminismTest(unittest.TestCase):
    def test_same_input_same_path(self) -> None:
        g = _grid(40, 40)
        g[15:25, 20] = True
        og = _og(g)
        s = og.cell_to_world((5, 5))
        goal = og.cell_to_world((35, 35))
        p1 = plan_path(og, s, goal)
        p2 = plan_path(og, s, goal)
        self.assertEqual(p1, p2)
        self.assertEqual(shortcut_path(og, p1), shortcut_path(og, p2))


if __name__ == "__main__":
    unittest.main()
