"""Tests for F7 space-time reservation integration in RobotUnit / Fleet.

Two layers:

* Flag-off byte-identity and flag-on booking lifecycle are exercised on a *fake*
  navigator (``object.__new__(RobotUnit)`` + a stub ``_nav``) so no physics is
  stepped — the plan/book/release wiring is pure bookkeeping.
* A guarded end-to-end smoke builds a real 2-robot fleet with reservations ON,
  steps it, and checks the booking flow survives real planning and nobody falls
  (skips on a fresh clone without the WBC ONNX / EGL).
"""

from __future__ import annotations

import unittest

from code.fleet.robot_unit import RobotState, RobotUnit
from code.planner.reserve import ReservationContext, ReservationTable


class _FakeNav:
    """Records plan() calls and returns a preset booking (no physics)."""

    def __init__(self, ok=True, booking=None, fell_back=False):
        self.ok = ok
        self.last_booking = booking if booking is not None else (
            [(0, 0), (0, 1)], [0, 10])
        self.st_fell_back = fell_back
        self.plan_calls = []
        self.cleared = 0

    def plan(self, goal_xy, *, reserve=None):
        self.plan_calls.append((goal_xy, reserve))
        return self.ok

    def clear_goal(self):
        self.cleared += 1

    @property
    def xy(self):
        return (0.0, 0.0)


def _bare_unit(*, name="Alpha", resv_on=False, table=None,
               now=None, nav=None, state=RobotState.IDLE):
    """A RobotUnit with its physics bypassed (fake nav, no arena built)."""
    u = object.__new__(RobotUnit)
    u.name = name
    u.state = state
    u.goal_xy = None
    u.plan_ok = None
    u._resv_on = resv_on
    u._resv_table = table
    u._resv_speed = 0.5
    u._resv_now = now if now is not None else (lambda: 0)
    u.st_fallbacks = 0
    u.st_replans = 0
    u._nav = nav if nav is not None else _FakeNav()
    return u


class TestFlagOffByteIdentity(unittest.TestCase):
    def test_assign_goal_uses_plain_plan(self) -> None:
        nav = _FakeNav()
        u = _bare_unit(resv_on=False, nav=nav)
        self.assertTrue(u.assign_goal((1.0, 2.0)))
        # Plain planner path: plan() called with reserve=None, nothing booked.
        self.assertEqual(len(nav.plan_calls), 1)
        self.assertIsNone(nav.plan_calls[0][1])
        self.assertEqual(u.st_replans, 0)
        self.assertEqual(u.st_fallbacks, 0)
        self.assertEqual(u.state, RobotState.WALKING)

    def test_release_is_noop_when_off(self) -> None:
        u = _bare_unit(resv_on=False)
        u.release_reservation()  # must not raise / touch anything

    def test_halt_when_off_does_not_touch_table(self) -> None:
        nav = _FakeNav()
        u = _bare_unit(resv_on=False, nav=nav, state=RobotState.WALKING)
        u.halt()
        self.assertEqual(u.state, RobotState.IDLE)
        self.assertEqual(nav.cleared, 1)


class TestFlagOnBookingLifecycle(unittest.TestCase):
    def test_assign_books_route(self) -> None:
        table = ReservationTable(0.10, footprint_radius_m=0.0, t_pad=1)
        nav = _FakeNav(booking=([(3, 3), (3, 4)], [0, 10]))
        u = _bare_unit(resv_on=True, table=table, nav=nav, now=lambda: 0)
        self.assertTrue(u.assign_goal((5.0, 5.0)))
        # A ReservationContext was passed through, and the route is booked.
        ctx = nav.plan_calls[0][1]
        self.assertIsInstance(ctx, ReservationContext)
        self.assertEqual(ctx.robot_id, "Alpha")
        self.assertEqual(ctx.t0, 0)
        self.assertEqual(u.st_replans, 1)
        self.assertEqual(table.active_robots(), {"Alpha"})
        self.assertTrue(table.is_reserved((3, 3), 0))

    def test_replan_releases_and_rebooks(self) -> None:
        table = ReservationTable(0.10, footprint_radius_m=0.0, t_pad=1)
        clock = {"t": 0}
        nav = _FakeNav(booking=([(3, 3)], [0]))
        u = _bare_unit(resv_on=True, table=table, nav=nav, now=lambda: clock["t"])
        u.assign_goal((5.0, 5.0))
        self.assertTrue(table.is_reserved((3, 3), 0))
        # Replan later: the old booking (t0=0) is released, the new one is at t0=100.
        clock["t"] = 100
        u.assign_goal((6.0, 6.0))
        self.assertFalse(table.is_reserved((3, 3), 0), "stale booking not released")
        self.assertTrue(table.is_reserved((3, 3), 100), "not rebooked at new t0")
        self.assertEqual(u.st_replans, 2)
        self.assertEqual(table.active_robots(), {"Alpha"})

    def test_halt_releases_booking(self) -> None:
        table = ReservationTable(0.10, footprint_radius_m=0.0, t_pad=1)
        nav = _FakeNav(booking=([(3, 3)], [0]))
        u = _bare_unit(resv_on=True, table=table, nav=nav, state=RobotState.IDLE)
        u.assign_goal((5.0, 5.0))  # -> WALKING, booked
        self.assertEqual(table.active_robots(), {"Alpha"})
        u.halt()
        self.assertEqual(u.state, RobotState.IDLE)
        self.assertEqual(table.active_robots(), set())

    def test_release_reservation_frees_booking(self) -> None:
        table = ReservationTable(0.10, footprint_radius_m=0.0, t_pad=1)
        u = _bare_unit(resv_on=True, table=table, nav=_FakeNav(booking=([(3, 3)], [0])))
        u.assign_goal((5.0, 5.0))
        u.release_reservation()
        self.assertEqual(table.active_robots(), set())

    def test_fallback_is_counted_and_still_booked(self) -> None:
        table = ReservationTable(0.10, footprint_radius_m=0.0, t_pad=1)
        nav = _FakeNav(booking=([(3, 3)], [0]), fell_back=True)
        u = _bare_unit(resv_on=True, table=table, nav=nav)
        u.assign_goal((5.0, 5.0))
        self.assertEqual(u.st_fallbacks, 1)
        self.assertEqual(u.st_replans, 1)
        self.assertEqual(table.active_robots(), {"Alpha"})  # fallback still booked

    def test_failed_plan_books_nothing(self) -> None:
        table = ReservationTable(0.10, footprint_radius_m=0.0, t_pad=1)
        nav = _FakeNav(ok=False)
        u = _bare_unit(resv_on=True, table=table, nav=nav)
        self.assertFalse(u.assign_goal((5.0, 5.0)))
        self.assertEqual(u.state, RobotState.IDLE)
        self.assertEqual(table.active_robots(), set())
        self.assertEqual(u.st_replans, 0)


class TestFleetReservationSmoke(unittest.TestCase):
    """Real 2-robot fleet with reservations ON (guarded)."""

    @classmethod
    def setUpClass(cls) -> None:
        try:
            from code.fleet.fleet import Fleet
            from code.warehouse.layout import hero_layout

            layout = hero_layout()
            spots = layout.object_spots
            goals = {"Alpha": spots[7], "Bravo": spots[4]}
            cls.fleet = Fleet(layout, goals, callsigns=["Alpha", "Bravo"],
                              use_gpu=True, build_viz=False, reservations=True)
        except unittest.SkipTest:
            raise
        except Exception as e:  # pragma: no cover - environment-dependent
            raise unittest.SkipTest(f"WBC/MuJoCo/EGL unavailable: {e}")

    def test_00_flag_on_builds_table_and_books(self) -> None:
        self.assertIsInstance(self.fleet.reservation_table, ReservationTable)
        self.assertTrue(self.fleet.reservations)
        # Both robots planned and booked their route at construction (t0=0).
        self.assertEqual(self.fleet.st_replans, 2)
        self.assertEqual(self.fleet.reservation_table.active_robots(),
                         {"Alpha", "Bravo"})

    def test_01_short_run_no_falls(self) -> None:
        self.fleet.run(80)
        self.assertFalse(self.fleet.any_fell, "a robot fell with reservations ON")
        moved = [u.walked_length for u in self.fleet.units.values()]
        self.assertTrue(any(w > 0.05 for w in moved), f"nobody moved: {moved}")

    def test_02_flag_off_builds_no_table(self) -> None:
        from code.fleet.fleet import Fleet
        from code.warehouse.layout import hero_layout

        layout = hero_layout()
        spots = layout.object_spots
        f = Fleet(layout, {"Alpha": spots[7]}, callsigns=["Alpha"],
                  use_gpu=True, build_viz=False, reservations=False)
        try:
            self.assertIsNone(f.reservation_table)
            self.assertFalse(f.reservations)
            self.assertEqual(f.st_replans, 0)
            self.assertFalse(f.units["Alpha"]._resv_on)
        finally:
            f.close()


if __name__ == "__main__":
    unittest.main()
