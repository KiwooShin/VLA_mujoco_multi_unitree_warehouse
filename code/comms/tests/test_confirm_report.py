"""Regression tests for CONFIRM-THEN-REPORT (gen_eval rooms seed-6 fix).

A learned detector's world-xy error grows with range (the confirmer's bearing
gate admits a detection whose lever-arm error scales as range*sin(22 deg)), so a
FIRST sighting made far away can be metres off even though it passed the gate.
Reporting/committing that raw long-range estimate stranded the fetcher outside
pickup range (rooms seed 6). The fix: in groundnet mode a sighting beyond the
detector's reliable-report range is NOT reported immediately — the searcher (or
the owner on its own first sighting) walks toward it to a standoff, re-confirms
at close range, and only THEN reports/commits the refined estimate. Bounded to a
single approach leg: reaching the standoff or losing sight of the object falls
back to the long-range estimate flagged ``approx`` (which the owner then refines
against with a wider gate). Oracle mode (``confirm_report_range_m() is None``) is
byte-identical — a sighting is reported the instant it is seen.

All exercised with scripted fakes (no physics), matching the fleet bridge's
contract: ``confirm_range`` sets the reliable range and ``approach_location`` /
``approach_after_polls`` model the fresh close-range look a robot gets after
walking toward a long-range sighting.
"""

from __future__ import annotations

import unittest

from code.comms.bus import MessageBus
from code.comms.messages import (ObjectQuery, Performative,
                                 relative_report_payload, reconstruct_location)
from code.comms.protocol import (CONFIRM_MAX_NO_SIGHT_STEPS,
                                  GOAL_REFINE_MAX_DELTA_APPROX_M,
                                  GOAL_REFINE_MAX_DELTA_M,
                                  RobotProtocol, RobotState)
from code.comms.tests._helpers import FakeActions, FakeClock, pump
from code.comms.tests.test_protocol import (CALLSIGNS, LOC, _FleetFixture,
                                            _seq, _task)
from code.comms.tests.test_protocol_robustness import (_perfs, _single_owner)

P = Performative
Q = ObjectQuery("red", "cube")

# A long-range first sighting (5 m out) and the fresh close-range estimate the
# robot gets after walking toward it (3 m out) — 5 m > 4.5 m reliable range so it
# must be close-confirmed; 3 m <= 4.5 m so the close look is reported/committed.
FAR = (5.0, 0.0)
NEAR = (3.0, 0.0)
STANDOFF = (2.0, 0.0)   # 3 m short of FAR from the origin pose == (2, 0)
RELIABLE = 4.5


def _searcher(act: FakeActions):
    """A lone searcher already commanded to patrol 'north' for the red cube."""
    clock = FakeClock()
    bus = MessageBus(clock.now)
    bravo = RobotProtocol("Bravo", bus, act, peers=("Alpha",),
                          search_regions=("north",))
    bus.post("Alpha", "Bravo", P.COMMAND_SEARCH,
             {"query": Q, "region": "north", "cancel": False})
    return clock, bus, bravo


def _reports(bus: MessageBus):
    return [m for m in bus.transcript if m.performative is P.REPORT_FOUND]


# ---------------------------------------------------------------------------
# Searcher: confirm-then-report state machine
# ---------------------------------------------------------------------------
class TestSearcherConfirmThenReport(unittest.TestCase):
    """A searcher close-confirms a long-range find before REPORT_FOUND."""

    def test_long_range_sighting_walks_in_then_reports_refined_xy(self) -> None:
        act = FakeActions(search_location=FAR, approach_location=NEAR,
                          find_after_polls=0, confirm_range=RELIABLE)
        clock, bus, bravo = _searcher(act)
        for _ in range(6):
            bravo.step(clock.now()); clock.tick()

        reps = _reports(bus)
        self.assertEqual(len(reps), 1)
        r = reps[0]
        self.assertEqual((r.sender, r.recipient), ("Bravo", "Alpha"))
        # Reports the REFINED close-range estimate (3, 0), not the raw 5 m sighting.
        self.assertEqual(reconstruct_location(r.payload), NEAR)
        self.assertNotIn("approx", r.payload)      # a real close confirm, not approx
        # It planned an approach leg to a standoff 3 m short of the sighting.
        self.assertIn(STANDOFF, act.calls("goto"))
        self.assertTrue(bravo.is_idle())           # done after reporting

    def test_no_resight_falls_back_to_approx_long_range_estimate(self) -> None:
        # Sights at 5 m, starts the approach, then never re-sights (lost line of
        # sight, never arrives): after the grace window it reports approx.
        act = FakeActions(search_location=FAR, approach_location=None,
                          static_location=None, find_after_polls=0,
                          arrives=False, confirm_range=RELIABLE)
        clock, bus, bravo = _searcher(act)
        for _ in range(CONFIRM_MAX_NO_SIGHT_STEPS + 8):
            bravo.step(clock.now()); clock.tick()

        reps = _reports(bus)
        self.assertEqual(len(reps), 1)
        r = reps[0]
        self.assertTrue(r.payload.get("approx"))            # flagged approximate
        self.assertEqual(reconstruct_location(r.payload), FAR)  # the long-range estimate
        self.assertTrue(bravo.is_idle())

    def test_reaching_standoff_without_close_confirm_reports_approx(self) -> None:
        # Sights at 5 m, walks in, but the close look never gets inside the
        # reliable range before arriving at the standoff -> approx fallback.
        act = FakeActions(search_location=FAR, approach_location=None,
                          static_location=FAR, find_after_polls=0,
                          arrives=True, confirm_range=RELIABLE)
        clock, bus, bravo = _searcher(act)
        for _ in range(6):
            bravo.step(clock.now()); clock.tick()

        reps = _reports(bus)
        self.assertEqual(len(reps), 1)
        self.assertTrue(reps[0].payload.get("approx"))
        self.assertEqual(reconstruct_location(reps[0].payload), FAR)

    def test_oracle_mode_reports_immediately_no_approach(self) -> None:
        # confirm_range=None (oracle) -> the discipline is off: report at once,
        # no approach leg, no approx flag — byte-identical to the historical path.
        act = FakeActions(search_location=FAR, approach_location=NEAR,
                          find_after_polls=0, confirm_range=None)
        clock, bus, bravo = _searcher(act)
        for _ in range(4):
            bravo.step(clock.now()); clock.tick()

        reps = _reports(bus)
        self.assertEqual(len(reps), 1)
        self.assertEqual(reconstruct_location(reps[0].payload), FAR)
        self.assertNotIn("approx", reps[0].payload)
        self.assertNotIn(STANDOFF, act.calls("goto"))       # never planned a standoff

    def test_confirm_uses_no_new_performatives(self) -> None:
        # The transcript stays sensible: COMMAND_SEARCH/ACCEPT/REPORT_FOUND only,
        # the find just arrives a few steps later.
        act = FakeActions(search_location=FAR, approach_location=NEAR,
                          find_after_polls=0, confirm_range=RELIABLE)
        clock, bus, bravo = _searcher(act)
        for _ in range(6):
            bravo.step(clock.now()); clock.tick()
        perfs = set(_perfs(bus))
        self.assertTrue(perfs <= {P.COMMAND_SEARCH, P.ACCEPT, P.REPORT_FOUND})


# ---------------------------------------------------------------------------
# Searcher: full-fleet integration (owner recovers via the confirmed report)
# ---------------------------------------------------------------------------
class TestSearcherConfirmIntegration(unittest.TestCase):
    """End-to-end: the owner fetches the confirmed (not the raw) location."""

    def test_owner_completes_on_confirmed_report(self) -> None:
        actions = {
            "Alpha": FakeActions(confirm_range=RELIABLE),
            "Bravo": FakeActions(search_location=FAR, approach_location=NEAR,
                                 find_after_polls=0, confirm_range=RELIABLE),
            "Charlie": FakeActions(confirm_range=RELIABLE),
            "Delta": FakeActions(confirm_range=RELIABLE)}
        fx = _FleetFixture(actions, regions=("north", "middle", "south"))
        fx.bus.post("user", "Alpha", P.REQUEST_TASK, {"task": _task()})
        pump(fx.clock, fx.order(CALLSIGNS), 30)

        reps = _reports(fx.bus)
        self.assertEqual(len(reps), 1)
        self.assertEqual((reps[0].sender, reps[0].recipient), ("Bravo", "Alpha"))
        self.assertEqual(reconstruct_location(reps[0].payload), NEAR)  # confirmed xy
        self.assertNotIn("approx", reps[0].payload)
        self.assertEqual(fx.alpha.last_result, "complete")
        # Owner navigated to the confirmed close estimate.
        self.assertIn(NEAR, actions["Alpha"].calls("goto"))


# ---------------------------------------------------------------------------
# Owner: range discipline on its OWN first sighting
# ---------------------------------------------------------------------------
class TestOwnerOwnSightingRangeDiscipline(unittest.TestCase):
    """The owner walks in to confirm its own long-range sighting before fetching."""

    def test_owner_confirms_then_commits_refined_goal(self) -> None:
        act = FakeActions(static_location=FAR, approach_location=NEAR,
                          approach_after_polls=1, confirm_range=RELIABLE,
                          arrives=False)
        clock, bus, alpha = _single_owner(act)

        alpha.step(clock.now()); clock.tick()   # own long-range sighting -> confirm leg
        self.assertEqual(alpha.state, RobotState.OWNER_NAVIGATING)
        self.assertIsNone(alpha.located_target)          # not committed while confirming
        self.assertNotIn(P.STATUS_UPDATE, _perfs(bus))   # no "found it" announced yet
        self.assertEqual(act.calls("goto"), [STANDOFF])  # standoff, not onto (5, 0)

        alpha.step(clock.now()); clock.tick()   # close confirm -> commit refined goal
        self.assertEqual(alpha.located_target, NEAR)
        self.assertEqual(act.calls("goto"), [STANDOFF, NEAR])
        self.assertIn(P.STATUS_UPDATE, _perfs(bus))      # now it announces the find

    def test_owner_oracle_mode_commits_immediately(self) -> None:
        # confirm_range=None -> byte-identical old behaviour: begin nav straight to
        # the sighting, announce it, no standoff.
        act = FakeActions(static_location=FAR, approach_location=NEAR,
                          confirm_range=None)
        clock, bus, alpha = _single_owner(act)
        alpha.step(clock.now()); clock.tick()
        self.assertEqual(alpha.located_target, FAR)
        self.assertEqual(act.calls("goto"), [FAR])
        self.assertIn(P.STATUS_UPDATE, _perfs(bus))


# ---------------------------------------------------------------------------
# Owner: an approx report gets a wider refinement gate
# ---------------------------------------------------------------------------
class TestApproxReportWiderRefineGate(unittest.TestCase):
    """A report flagged ``approx`` widens the owner's same-object refine gate."""

    def _owner_navigating_from_report(self, approx: bool):
        # Drive Alpha into OWNER_DELEGATING (nobody sees), then feed a REPORT_FOUND.
        actions = {cs: FakeActions() for cs in CALLSIGNS}
        fx = _FleetFixture(actions, regions=("north", "middle", "south"))
        fx.bus.post("user", "Alpha", P.REQUEST_TASK, {"task": _task()})
        pump(fx.clock, fx.order(CALLSIGNS), 12)
        self.assertEqual(fx.alpha.state, RobotState.OWNER_DELEGATING)
        extra = {"object": Q, "approx": True} if approx else {"object": Q}
        payload = relative_report_payload((0.0, 0.0), "north", LOC, extra=extra)
        fx.bus.post("Bravo", "Alpha", P.REPORT_FOUND, payload)
        fx.alpha.step(fx.clock.now()); fx.clock.tick()
        self.assertEqual(fx.alpha.state, RobotState.OWNER_NAVIGATING)
        return fx.alpha

    def test_approx_report_accepts_a_wider_nudge(self) -> None:
        alpha = self._owner_navigating_from_report(approx=True)
        # A 3 m nudge is beyond the normal 2.5 m gate but inside the 4.0 m approx
        # gate -> accepted only because the report was flagged approx.
        self.assertGreater(3.0, GOAL_REFINE_MAX_DELTA_M)
        self.assertLessEqual(3.0, GOAL_REFINE_MAX_DELTA_APPROX_M)
        nudge = (LOC[0] + 3.0, LOC[1])
        t = 1000  # well past the refine rate-limit window (set when nav began)
        self.assertTrue(alpha.refine_nav_goal(nudge, t, robot_xy=(0.0, 0.0)))
        self.assertEqual(alpha.located_target, nudge)

    def test_exact_report_rejects_the_same_wide_nudge(self) -> None:
        alpha = self._owner_navigating_from_report(approx=False)
        nudge = (LOC[0] + 3.0, LOC[1])
        t = 1000
        self.assertFalse(alpha.refine_nav_goal(nudge, t, robot_xy=(0.0, 0.0)))
        self.assertEqual(alpha.located_target, LOC)   # goal unchanged


def _visibility_replies(bus: MessageBus):
    return [m for m in bus.transcript
            if m.performative is P.REPORT_VISIBILITY and m.payload.get("visible")]


def _peer_reply(act: FakeActions):
    """A lone peer that answers one QUERY_VISIBILITY for the red cube."""
    clock = FakeClock()
    bus = MessageBus(clock.now)
    bravo = RobotProtocol("Bravo", bus, act, peers=("Alpha",))
    bus.post("Alpha", "Bravo", P.QUERY_VISIBILITY, {"query": Q})
    bravo.step(clock.now()); clock.tick()
    return bus, bravo


# ---------------------------------------------------------------------------
# Peer QUERY_VISIBILITY reply: CONFIRM-THEN-REPORT parity (approx flag)
# ---------------------------------------------------------------------------
class TestPeerVisibilityReplyApprox(unittest.TestCase):
    """A peer's long-range visibility reply is flagged approx (no walk-in)."""

    def test_long_range_reply_flagged_approx_no_walk_in(self) -> None:
        act = FakeActions(static_location=FAR, confirm_range=RELIABLE,
                          pose=(0.0, 0.0))
        bus, bravo = _peer_reply(act)
        reps = _visibility_replies(bus)
        self.assertEqual(len(reps), 1)
        self.assertTrue(reps[0].payload.get("approx"))       # 5 m > 4.5 m reliable
        self.assertEqual(reconstruct_location(reps[0].payload), FAR)
        # A peer answers a question; it does NOT walk in to close-confirm.
        self.assertEqual(act.calls("goto"), [])
        self.assertTrue(bravo.is_idle())

    def test_close_range_reply_not_flagged(self) -> None:
        act = FakeActions(static_location=NEAR, confirm_range=RELIABLE,
                          pose=(0.0, 0.0))
        bus, _ = _peer_reply(act)
        self.assertNotIn("approx", _visibility_replies(bus)[0].payload)  # 3 m <= 4.5 m

    def test_oracle_mode_reply_byte_identical(self) -> None:
        # confirm_range=None (oracle): no approx key even for a far sighting.
        act = FakeActions(static_location=FAR, confirm_range=None, pose=(0.0, 0.0))
        bus, _ = _peer_reply(act)
        self.assertNotIn("approx", _visibility_replies(bus)[0].payload)


class TestOwnerCommitsPeerApprox(unittest.TestCase):
    """The owner commits a long-range peer reply with the WIDER refine gate."""

    def test_owner_commits_approx_and_widens_refine_gate(self) -> None:
        actions = {"Alpha": FakeActions(confirm_range=RELIABLE, arrives=False),
                   "Bravo": FakeActions(static_location=FAR, confirm_range=RELIABLE),
                   "Charlie": FakeActions(confirm_range=RELIABLE),
                   "Delta": FakeActions(confirm_range=RELIABLE)}
        fx = _FleetFixture(actions, regions=("north", "middle", "south"))
        fx.bus.post("user", "Alpha", P.REQUEST_TASK, {"task": _task()})
        pump(fx.clock, fx.order(CALLSIGNS), 8)

        reps = _visibility_replies(fx.bus)
        self.assertTrue(reps and reps[0].payload.get("approx"))
        self.assertEqual(fx.alpha.state, RobotState.OWNER_NAVIGATING)
        self.assertEqual(fx.alpha.located_target, FAR)   # committed the raw estimate
        # Because the commit was flagged approx, a 3 m nudge (beyond the 2.5 m
        # exact gate, inside the 4.0 m approx gate) is now accepted during approach.
        self.assertGreater(3.0, GOAL_REFINE_MAX_DELTA_M)
        self.assertLessEqual(3.0, GOAL_REFINE_MAX_DELTA_APPROX_M)
        wide = (FAR[0] + 3.0, FAR[1])
        self.assertTrue(fx.alpha.refine_nav_goal(wide, 1000, robot_xy=(0.0, 0.0)))
        self.assertEqual(fx.alpha.located_target, wide)


# ---------------------------------------------------------------------------
# Searcher fall mid confirm-walk-in -> REJECT (not an approx REPORT_FOUND)
# ---------------------------------------------------------------------------
class TestSearcherFallDuringConfirm(unittest.TestCase):
    """A fall mid confirm-approach surfaces as a REJECT, not a weak find."""

    def test_fall_mid_confirm_rejects_and_does_not_report(self) -> None:
        act = FakeActions(search_location=FAR, approach_location=None,
                          static_location=FAR, find_after_polls=0, arrives=False,
                          confirm_range=RELIABLE)
        clock, bus, bravo = _searcher(act)
        for _ in range(4):                       # find long-range -> enter confirm
            bravo.step(clock.now()); clock.tick()
            if STANDOFF in act.calls("goto"):
                break
        self.assertIn(STANDOFF, act.calls("goto"))
        self.assertEqual(_reports(bus), [])      # nothing reported yet
        # Fall while walking in to confirm.
        act.nav_fails, act.fail_reason = True, "searcher fell"
        bravo.step(clock.now()); clock.tick()

        rej = [m for m in bus.transcript if m.performative is P.REJECT]
        self.assertEqual(len(rej), 1)
        self.assertEqual((rej[0].sender, rej[0].recipient), ("Bravo", "Alpha"))
        self.assertEqual(rej[0].payload["reason"], "searcher fell")
        # It must NOT emit an (approx) REPORT_FOUND — that would cancel healthy
        # co-searchers and commit the owner to a fallen robot's weak estimate.
        self.assertEqual(_reports(bus), [])
        self.assertTrue(bravo.is_idle())


if __name__ == "__main__":
    unittest.main()
