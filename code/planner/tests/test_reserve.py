"""Unit tests for code.planner.reserve (space-time reservations + ST-A*).

No simulator is stepped. Covers the reservation table semantics (book / release /
conflict windows / footprint inflation / first_conflict), the speed model, and the
space-time A* on hand-built two-robot scenarios (head-on corridor swap that plain
A* would collide on, and a spatial reroute around a permanently blocked cell),
plus empty-table parity with plain A*, determinism, and the node-budget fallback.
"""

from __future__ import annotations

import unittest

import numpy as np

from code.planner.astar import plan_path
from code.planner.grid import OccupancyGrid
from code.planner.reserve import (
    DEFAULT_SPEED_MPS,
    ReservationTable,
    cell_times_for_path,
    plan_path_st,
    steps_per_cell,
)


def _og(free_mask: np.ndarray, res: float = 0.10) -> OccupancyGrid:
    """Build a grid from a FREE mask (True == free) at origin (0, 0)."""
    return OccupancyGrid(~free_mask, res, (0.0, 0.0))


def _cells(og: OccupancyGrid, path):
    return [og.world_to_cell(p) for p in path]


def _route_clashes(table: ReservationTable, cells, times, t0, rid) -> bool:
    """True if the booked (cells, times) route hits another robot's reservation."""
    return any(table.is_reserved(c, t0 + ct, rid) for c, ct in zip(cells, times))


class TestSpeedModel(unittest.TestCase):
    def test_steps_per_cell_half_mps(self) -> None:
        # 0.5 m/s at 50 Hz -> 0.01 m/step; 0.10 m cell -> 10 steps, diag ~14.
        self.assertEqual(steps_per_cell(0.10, 0.5), (10, 14))

    def test_faster_is_fewer_steps(self) -> None:
        c_slow, _ = steps_per_cell(0.10, 0.4)
        c_fast, _ = steps_per_cell(0.10, 0.8)
        self.assertGreater(c_slow, c_fast)

    def test_min_one_step_per_cell(self) -> None:
        card, diag = steps_per_cell(0.10, 1000.0)
        self.assertGreaterEqual(card, 1)
        self.assertGreaterEqual(diag, 1)

    def test_cell_times_cardinal_and_diagonal(self) -> None:
        cells = [(0, 0), (0, 1), (1, 2)]  # cardinal then diagonal
        times = cell_times_for_path(cells, 0.10, 0.5)
        self.assertEqual(times, [0, 10, 24])  # +10 cardinal, +14 diagonal


class TestReservationTable(unittest.TestCase):
    def test_book_and_query_window(self) -> None:
        t = ReservationTable(0.10, footprint_radius_m=0.0, t_pad=2)
        t.reserve([(5, 5), (5, 6)], t_start=100, cell_times=[0, 10], robot_id="A")
        self.assertTrue(t.is_reserved((5, 5), 100))
        self.assertTrue(t.is_reserved((5, 5), 98))   # window edge (100 - t_pad)
        self.assertTrue(t.is_reserved((5, 5), 102))
        self.assertFalse(t.is_reserved((5, 5), 103))  # just past the +t_pad edge
        self.assertTrue(t.is_reserved((5, 6), 110))    # second cell at offset 10

    def test_ignore_id_excludes_own_booking(self) -> None:
        t = ReservationTable(0.10, footprint_radius_m=0.0, t_pad=1)
        t.reserve([(3, 3)], 50, [0], "A")
        self.assertTrue(t.is_reserved((3, 3), 50))
        self.assertFalse(t.is_reserved((3, 3), 50, ignore_id="A"))

    def test_release_removes_booking(self) -> None:
        t = ReservationTable(0.10, footprint_radius_m=0.0, t_pad=1)
        t.reserve([(3, 3)], 50, [0], "A")
        t.release("A")
        self.assertFalse(t.is_reserved((3, 3), 50))
        self.assertEqual(t.active_robots(), set())
        t.release("A")  # idempotent

    def test_release_keeps_other_robots(self) -> None:
        t = ReservationTable(0.10, footprint_radius_m=0.0, t_pad=1)
        t.reserve([(3, 3)], 50, [0], "A")
        t.reserve([(3, 3)], 60, [0], "B")
        t.release("A")
        self.assertFalse(t.is_reserved((3, 3), 50))  # A gone
        self.assertTrue(t.is_reserved((3, 3), 60))   # B stays
        self.assertEqual(t.active_robots(), {"B"})

    def test_footprint_inflation(self) -> None:
        t = ReservationTable(0.10, footprint_radius_m=0.25, t_pad=0)  # ~2-3 cells
        t.reserve([(10, 10)], 0, [0], "A")
        self.assertTrue(t.is_reserved((10, 12), 0))   # within footprint disk
        self.assertTrue(t.is_reserved((12, 10), 0))
        self.assertFalse(t.is_reserved((10, 20), 0))  # far outside

    def test_reserve_length_mismatch_raises(self) -> None:
        t = ReservationTable(0.10)
        with self.assertRaises(ValueError):
            t.reserve([(0, 0), (0, 1)], 0, [0], "A")

    def test_first_conflict_reports_index(self) -> None:
        t = ReservationTable(0.10, footprint_radius_m=0.0, t_pad=2)
        # Block cell (0, 3) around t = 30 (== the 0.5 m/s arrival time of index 3).
        t.reserve([(0, 3)], 30, [0], "A")
        path = [(0, i) for i in range(6)]  # times 0,10,20,30,40,50
        hit = t.first_conflict(path, t0=0, speed=0.5, ignore_id="B")
        self.assertIsNotNone(hit)
        self.assertEqual(hit[0], 3)
        self.assertEqual(hit[1], (0, 3))

    def test_first_conflict_clear(self) -> None:
        t = ReservationTable(0.10, footprint_radius_m=0.0, t_pad=2)
        t.reserve([(0, 3)], 500, [0], "A")  # far in the future
        path = [(0, i) for i in range(6)]
        self.assertIsNone(t.first_conflict(path, 0, 0.5, ignore_id="B"))


class TestPlanPathSTParity(unittest.TestCase):
    def test_empty_table_is_direct_optimal(self) -> None:
        # With no reservations the ST route must be a direct optimal path: no
        # fallback, no waits/detour, same move count as plain A* (ST's integer
        # step costs give 14/10 != sqrt(2), so the exact tie-break may differ —
        # cost-equivalence, not byte-identity, is what "no reservations" buys).
        og = _og(np.ones((10, 12), dtype=np.bool_))  # fully open
        table = ReservationTable(0.10)
        start = og.cell_to_world((1, 1))
        goal = og.cell_to_world((8, 10))
        plain = _cells(og, plan_path(og, start, goal))
        st = plan_path_st(og, table, start, goal, t0=0, speed=0.5)
        self.assertFalse(st.fell_back)
        self.assertTrue(st.conflict_free)
        self.assertEqual(len(st.cells), len(plain))       # same # of moves
        self.assertEqual(st.cells[0], plain[0])
        self.assertEqual(st.cells[-1], plain[-1])
        for c in st.cells:                                # every cell free
            self.assertFalse(og.grid[c])

    def test_pure_diagonal_matches_plain(self) -> None:
        # A pure diagonal has a unique shortest route -> ST must reproduce it.
        og = _og(np.ones((8, 8), dtype=np.bool_))
        start = og.cell_to_world((0, 0))
        goal = og.cell_to_world((7, 7))
        st = plan_path_st(og, None, start, goal, t0=0, speed=0.5)
        self.assertEqual(st.cells, _cells(og, plan_path(og, start, goal)))

    def test_determinism(self) -> None:
        og = _og(np.ones((10, 12), dtype=np.bool_))
        table = ReservationTable(0.10, footprint_radius_m=0.0, t_pad=3)
        table.reserve([(4, c) for c in range(12)], 0,
                      cell_times_for_path([(4, c) for c in range(12)], 0.10, 0.5),
                      "A")
        s = og.cell_to_world((0, 0))
        g = og.cell_to_world((9, 11))
        a = plan_path_st(og, table, s, g, 0, 0.5, ignore_id="B")
        b = plan_path_st(og, table, s, g, 0, 0.5, ignore_id="B")
        self.assertEqual(a.cells, b.cells)
        self.assertEqual(a.cell_times, b.cell_times)


class TestHeadOnCorridorSwap(unittest.TestCase):
    """A single-row corridor: A walks L->R, B walks R->L at the same time.

    Plain A* would put B straight through the cells A occupies. The ST search must
    return a route B can walk without ever sharing a space-time cell with A.
    """

    def _corridor(self):
        free = np.zeros((5, 11), dtype=np.bool_)
        free[2, :] = True
        return _og(free)

    def test_st_avoids_booked_corridor(self) -> None:
        og = self._corridor()
        table = ReservationTable(0.10, footprint_radius_m=0.0, t_pad=3)
        a_cells = _cells(og, plan_path(og, og.cell_to_world((2, 0)),
                                       og.cell_to_world((2, 10))))
        a_times = cell_times_for_path(a_cells, 0.10, 0.5)
        table.reserve(a_cells, 0, a_times, "A")

        st = plan_path_st(og, table, og.cell_to_world((2, 10)),
                          og.cell_to_world((2, 0)), t0=0, speed=0.5, ignore_id="B")
        self.assertFalse(st.fell_back)
        self.assertTrue(st.conflict_free)
        # B ends where it should, and its booked route never clashes with A.
        self.assertEqual(st.cells[-1], (2, 0))
        self.assertFalse(_route_clashes(table, st.cells, st.cell_times, 0, "B"))

    def test_st_solution_bookable_and_clears(self) -> None:
        og = self._corridor()
        table = ReservationTable(0.10, footprint_radius_m=0.0, t_pad=3)
        a_cells = _cells(og, plan_path(og, og.cell_to_world((2, 0)),
                                       og.cell_to_world((2, 10))))
        table.reserve(a_cells, 0, cell_times_for_path(a_cells, 0.10, 0.5), "A")
        st = plan_path_st(og, table, og.cell_to_world((2, 10)),
                          og.cell_to_world((2, 0)), 0, 0.5, ignore_id="B")
        table.reserve(st.cells, 0, st.cell_times, "B")  # must not raise
        self.assertEqual(table.active_robots(), {"A", "B"})


class TestSpatialReroute(unittest.TestCase):
    """A ring corridor with two arms: block one arm and the ST search must detour."""

    def _ring(self):
        # 5x7 perimeter ring; interior blocked.
        free = np.zeros((5, 7), dtype=np.bool_)
        free[0, :] = True   # top arm
        free[4, :] = True   # bottom arm
        free[:, 0] = True   # left connector
        free[:, 6] = True   # right connector
        return _og(free)

    def test_reroute_around_blocked_cell(self) -> None:
        og = self._ring()
        # Plain A* takes the top arm (row 0).
        plain = _cells(og, plan_path(og, og.cell_to_world((0, 0)),
                                     og.cell_to_world((0, 6))))
        self.assertTrue(all(c[0] == 0 for c in plain), "plain route should hug row 0")

        # Permanently block the middle of the top arm (huge t_pad covers horizon).
        table = ReservationTable(0.10, footprint_radius_m=0.0, t_pad=10_000)
        table.reserve([(0, 3)], 0, [0], "wall")
        st = plan_path_st(og, table, og.cell_to_world((0, 0)),
                          og.cell_to_world((0, 6)), 0, 0.5, ignore_id="B")
        self.assertFalse(st.fell_back)
        # The detour must use the bottom arm (row 4) and never enter (0, 3).
        self.assertTrue(any(c[0] == 4 for c in st.cells), "expected a bottom-arm detour")
        self.assertNotIn((0, 3), st.cells)


class TestNodeBudgetFallback(unittest.TestCase):
    def test_tiny_budget_falls_back_to_plain(self) -> None:
        og = _og(np.ones((10, 12), dtype=np.bool_))
        table = ReservationTable(0.10)
        start = og.cell_to_world((0, 0))
        goal = og.cell_to_world((9, 11))
        st = plan_path_st(og, table, start, goal, 0, 0.5, node_budget=1)
        self.assertTrue(st.fell_back)
        # Fallback is a real, bookable plain-A* route to the goal.
        self.assertEqual(st.cells, _cells(og, plan_path(og, start, goal)))
        self.assertEqual(st.cells[-1], og.world_to_cell(goal))


class TestDefaults(unittest.TestCase):
    def test_default_speed_is_conservative(self) -> None:
        self.assertAlmostEqual(DEFAULT_SPEED_MPS, 0.5)


if __name__ == "__main__":
    unittest.main()
