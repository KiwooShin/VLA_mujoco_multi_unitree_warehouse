"""fakes.py — A MuJoCo-free fake :class:`SimEngine` for service/route tests.

Scripts a tiny deterministic mission (request -> status -> complete) so the
:class:`FleetService` loop can be driven end-to-end without EGL, GPU, or a real
warehouse build. It reveals bus lines on a fixed step schedule and honours the
``StopSim`` abort contract, mirroring the real engine's public behaviour.
"""

from __future__ import annotations

from typing import Callable, List, Optional, Sequence

from code.apps.fleet_web.commands import validate_command
from code.apps.fleet_web.engine import MsgSnap, SimEngine, StopSim
from code.apps.fleet_web.status import RobotSnap

_CALLSIGNS = ("Alpha", "Bravo", "Charlie", "Delta")


class FakeEngine(SimEngine):
    """A scripted, deterministic sim engine (no MuJoCo)."""

    def __init__(self, *, mission_steps: int = 6) -> None:
        self.mission_steps = mission_steps
        self.reset_count = 0
        self.idle_steps = 0
        self.in_mission = False
        self._owner = ""
        self._target = ""
        self._phase = "STANDING BY"
        self._revealed: List[MsgSnap] = []
        self._schedule: List[tuple] = []
        self._outcome = "complete"
        self._on_pad = False

    @property
    def callsigns(self) -> Sequence[str]:
        return _CALLSIGNS

    def reset(self) -> None:
        self.reset_count += 1
        self.in_mission = False
        self._owner = ""
        self._target = ""
        self._phase = "STANDING BY"
        self._revealed = []
        self._schedule = []
        self._on_pad = False

    def submit(self, text: str) -> None:
        check = validate_command(text, _CALLSIGNS)
        self._target = check.target_desc
        self._owner = "Bravo" if check.is_fleet else check.recipient
        self._revealed = []
        self._on_pad = False
        o = self._owner
        tg = self._target
        self._schedule = [
            (0, MsgSnap("user", o,
                        f"REQUEST_TASK: fetch the {tg} to the delivery pad")),
            (1, MsgSnap(o, "user",
                        f"STATUS_UPDATE: heading over to fetch the {tg}.")),
            (self.mission_steps - 1, MsgSnap(
                o, "user", f"TASK_COMPLETE: Delivered the {tg} to the "
                "delivery pad.")),
        ]

    def run_mission(self, on_step: Callable[[int], None], max_steps: int) -> str:
        self.in_mission = True
        self._phase = "FETCHING"
        steps = min(self.mission_steps, max_steps)
        try:
            for t in range(steps):
                while self._schedule and self._schedule[0][0] == t:
                    self._revealed.append(self._schedule.pop(0)[1])
                on_step(t)
        except StopSim:
            self.in_mission = False
            self._phase = "STANDING BY"
            return "stopped"
        self.in_mission = False
        self._phase = "DONE"
        self._on_pad = self._outcome == "complete"
        return self._outcome

    def idle_step(self) -> None:
        self.idle_steps += 1

    def render_jpeg(self) -> Optional[bytes]:
        return b"\xff\xd8FAKEJPEG\xff\xd9"

    def robots(self) -> List[RobotSnap]:
        out: List[RobotSnap] = []
        for cs in _CALLSIGNS:
            owner = cs == self._owner and self.in_mission
            out.append(RobotSnap(
                name=cs,
                coord_state="OWNER_NAVIGATING" if owner else "IDLE",
                motion="walking" if owner else "idle",
                dist_to_goal=2.5 if owner else None,
                carrying=False, is_owner=owner,
                task_desc=self._target if owner else ""))
        return out

    def bus_snaps(self) -> List[MsgSnap]:
        return list(self._revealed)

    def mission_phase(self) -> str:
        return self._phase

    def task_desc(self) -> str:
        return self._target

    def object_on_pad(self) -> bool:
        return self._on_pad
