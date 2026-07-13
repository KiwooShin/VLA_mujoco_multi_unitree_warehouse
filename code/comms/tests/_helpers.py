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
    """

    def __init__(self, *, static_location: Optional[XY] = None,
                 search_location: Optional[XY] = None,
                 find_after_polls: int = 0, arrives: bool = True) -> None:
        self.static_location = static_location
        self.search_location = search_location
        self.find_after_polls = find_after_polls
        self._arrives = arrives
        self._searching = False
        self._poll_count = 0
        self.log: List[Tuple[str, object]] = []

    def can_see(self, query) -> Optional[XY]:
        if self._searching:
            self._poll_count += 1
            if (self.search_location is not None
                    and self._poll_count > self.find_after_polls):
                return self.search_location
            return None
        return self.static_location

    def goto(self, xy: XY) -> None:
        self.log.append(("goto", xy))

    def arrived(self) -> bool:
        return self._arrives

    def start_search(self, query, region: str) -> None:
        self._searching = True
        self._poll_count = 0
        self.log.append(("start_search", region))

    def abort_search(self) -> None:
        self._searching = False
        self.log.append(("abort_search", None))

    def pickup(self, query) -> None:
        self.log.append(("pickup", None))

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
