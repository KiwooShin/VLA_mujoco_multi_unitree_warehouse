"""Shared test doubles for the code.comms unit tests.

FakeClock is a deterministic step counter; FakeActions is a scripted
:class:`~code.comms.protocol.RobotActions` that records every call and can be
configured to "see" an object either at query time or only after it has been
searching for a while; ``pump`` drives a set of protocols for a fixed number of
steps in a deterministic order.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

from code.comms.protocol import RobotActions, RobotProtocol

XY = Tuple[float, float]


class FakeClock:
    """A deterministic monotonic step counter."""

    def __init__(self) -> None:
        self.t = 0

    def now(self) -> int:
        """Return the current step."""
        return self.t

    def tick(self) -> None:
        """Advance one step."""
        self.t += 1


class FakeActions(RobotActions):
    """A scripted, call-recording RobotActions.

    Args:
        static_location: What :meth:`can_see` returns when *not* searching
            (models a robot that can see the object from where it stands).
        search_location: What :meth:`can_see` returns once a search has been
            running for more than ``find_after_polls`` polls (models a robot
            that finds the object by searching). ``None`` -> never finds.
        find_after_polls: Number of search polls that return ``None`` before
            ``search_location`` is returned.
        arrives: What :meth:`arrived` returns (nav completes on the next poll).
        nav_fails: What :meth:`failed` returns (models a fall / unreachable goal).
        fail_reason: The reason string surfaced by :meth:`failure_reason`.
        can_search: What :meth:`start_search` returns (False models a region with
            no reachable patrol, so the searcher should REJECT).
        can_pickup: What :meth:`pickup` returns (False models a missed grasp).
        reconfirm_location: What :meth:`reconfirm_target` returns (a fresh
            close-range perception fix before a pickup re-approach). ``None`` (the
            default) models the oracle path — no fresh fix — leaving the retry's
            goal unchanged.
        confirm_range: What :meth:`confirm_report_range_m` returns — the
            CONFIRM-THEN-REPORT reliable-report range. ``None`` (default) disables
            the discipline (the oracle path: a sighting is reported immediately).
        approach_location: What :meth:`can_see` returns once an approach has begun
            (i.e. after the first :meth:`goto`) while not searching — models the
            fresh, closer estimate a robot gets after walking toward a long-range
            sighting. ``None`` -> :meth:`can_see` keeps returning ``static_location``.
        approach_after_polls: Number of approaching :meth:`can_see` polls that
            still return ``static_location`` (the object still far, mid-approach)
            before ``approach_location`` kicks in — models the walk-in taking a
            few steps to reach a reliable close-range look.
    """

    def __init__(self, *, static_location: Optional[XY] = None,
                 search_location: Optional[XY] = None,
                 find_after_polls: int = 0, arrives: bool = True,
                 nav_fails: bool = False, fail_reason: str = "goal unreachable",
                 can_search: bool = True, can_pickup: bool = True,
                 reconfirm_location: Optional[XY] = None,
                 confirm_range: Optional[float] = None,
                 approach_location: Optional[XY] = None,
                 approach_after_polls: int = 0,
                 pose: XY = (0.0, 0.0), room: str = "the area") -> None:
        self.static_location = static_location
        self.search_location = search_location
        self.find_after_polls = find_after_polls
        self._arrives = arrives
        self.nav_fails = nav_fails
        self.fail_reason = fail_reason
        self.can_search = can_search
        self.can_pickup = can_pickup
        self.reconfirm_location = reconfirm_location
        self.confirm_range = confirm_range
        self.approach_location = approach_location
        self.approach_after_polls = approach_after_polls
        self.pose = pose            # this robot's own (exactly known) pose (F3)
        self.room = room            # this robot's current region/room name (F3)
        self._searching = False
        self._poll_count = 0
        self._approaching = False   # latched once a goto (approach) has begun
        self._approach_polls = 0
        self.log: List[Tuple[str, object]] = []

    def can_see(self, query) -> Optional[XY]:
        if self._searching:
            self._poll_count += 1
            if (self.search_location is not None
                    and self._poll_count > self.find_after_polls):
                return self.search_location
            return None
        if self._approaching and self.approach_location is not None:
            self._approach_polls += 1
            if self._approach_polls > self.approach_after_polls:
                return self.approach_location
            return self.static_location   # still far, mid-approach
        return self.static_location

    def confirm_report_range_m(self) -> Optional[float]:
        return self.confirm_range

    def reconfirm_target(self, query) -> Optional[XY]:
        self.log.append(("reconfirm_target", self.reconfirm_location))
        return self.reconfirm_location

    def report_origin(self) -> Tuple[XY, str]:
        return (self.pose, self.room)

    def goto(self, xy: XY) -> None:
        self._approaching = True
        self.log.append(("goto", xy))

    def arrived(self) -> bool:
        return self._arrives

    def failed(self) -> bool:
        return self.nav_fails

    def failure_reason(self) -> str:
        return self.fail_reason

    def start_search(self, query, region: str) -> bool:
        self.log.append(("start_search", region))
        if not self.can_search:
            return False
        self._searching = True
        self._poll_count = 0
        return True

    def abort_search(self) -> None:
        self._searching = False
        self.log.append(("abort_search", None))

    def pickup(self, query) -> bool:
        self.log.append(("pickup", None))
        return self.can_pickup

    def deliver(self, destination_xy: XY) -> None:
        self.log.append(("deliver", destination_xy))

    def calls(self, name: str) -> List[object]:
        """Return the recorded argument of every call to action ``name``."""
        return [arg for n, arg in self.log if n == name]


def pump(clock: FakeClock, protocols: Sequence[RobotProtocol],
         ticks: int) -> None:
    """Step every protocol (in the given order) for ``ticks`` ticks.

    Messages posted during a tick land in inboxes and are drained at the start of
    each protocol's next step, so a message hop costs about one tick — fully
    deterministic for a fixed protocol order.
    """
    for _ in range(ticks):
        for proto in protocols:
            proto.step(clock.now())
        clock.tick()
