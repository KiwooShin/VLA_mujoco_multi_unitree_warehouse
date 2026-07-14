"""runner_adapter.py — Build a generic FrameState from a MissionRunner.

This is the ONE place that knows the current :class:`~code.fleet.mission.MissionRunner`
shape; the composer stays mission-agnostic. Everything is read defensively
(``getattr`` / ``try``) so the parallel fleet/comms work — new performatives,
concurrent owners, extra rings — degrades gracefully rather than crashing.

Import-light (no cv2 / MuJoCo): it only reads attributes off a live runner.
"""

from __future__ import annotations

from typing import List, Optional

from code.apps.demos import style
from code.apps.demos.models import (FrameState, Pad, PlannedPath, Ring,
                                    RobotFrame)

# Protocol RobotState.name -> tile chip phrase.
_STATE_CHIP = {
    "OWNER_QUERYING": "querying peers",
    "OWNER_DELEGATING": "delegating search",
    "OWNER_NAVIGATING": "fetching",
    "OWNER_DELIVERING": "carrying",
    "ASSIST_SEARCHING": "searching",
}
_UNIT_CHIP = {"walking": "moving", "paused": "paused", "arrived": "standing by",
              "idle": "standing by", "fallen": "fell"}


def _chip(mr, cs: str) -> str:
    """One robot's state-chip phrase (carrying / searching X / fetching / ...)."""
    try:
        if mr.carry.carrying(cs):
            return "carrying"
    except Exception:
        pass
    st = getattr(getattr(mr.protocols.get(cs), "state", None), "name", None)
    if st == "ASSIST_SEARCHING":
        region = getattr(mr._search.get(cs), "region", None)
        return f"searching {region}" if region else "searching"
    if st in _STATE_CHIP:
        return _STATE_CHIP[st]
    unit = mr.fleet.units[cs]
    return _UNIT_CHIP.get(getattr(unit.state, "value", ""), "standing by")


def _rings(mr) -> List[Ring]:
    """Target ring(s). Supports multiple if the runner exposes a list."""
    rings: List[Ring] = []
    multi = getattr(mr, "known_targets_xy", None)      # future multi-goal hook
    if callable(multi):
        try:
            for xy in multi() or []:
                if xy is not None:
                    rings.append(Ring(tuple(xy)))
            if rings:
                return rings
        except Exception:
            pass
    try:
        xy = mr.known_target_xy()
        if xy is not None:
            rings.append(Ring((float(xy[0]), float(xy[1]))))
    except Exception:
        pass
    return rings


def _pads(mr) -> List[Pad]:
    """Delivery-pad highlight(s) from the layout zones."""
    pads: List[Pad] = []
    for z in getattr(mr.layout, "zones", []):
        if getattr(z, "name", "") == "delivery":
            pads.append(Pad((float(z.cx), float(z.cy)), float(z.half_x),
                            float(z.half_y)))
    return pads


def _planned_paths(mr) -> List[PlannedPath]:
    """Planned route for any owner currently navigating (accent-coloured)."""
    out: List[PlannedPath] = []
    for cs in mr.callsigns:
        st = getattr(getattr(mr.protocols.get(cs), "state", None), "name", None)
        if st not in ("OWNER_NAVIGATING", "OWNER_DELIVERING"):
            continue
        try:
            pts = list(mr.fleet.units[cs].planned_path or [])
        except Exception:
            pts = []
        if len(pts) >= 2:
            out.append(PlannedPath(pts, color=style.accent_bgr(cs)))
    return out


def _mission_lines(mr) -> List[str]:
    """Order + allocator lines under the panel title."""
    lines: List[str] = []
    task = getattr(mr, "task", None)
    if task is not None:
        try:
            lines.append(f"Order: {task.describe()}")
        except Exception:
            pass
    alloc = getattr(mr, "allocation", None)
    if alloc is not None:
        try:
            lines.append(f"Allocator: {alloc.describe()}")
        except Exception:
            pass
    return lines


def frame_state_from_runner(mr, step: int, *,
                            sim_time: Optional[float] = None) -> FrameState:
    """Assemble a :class:`FrameState` from a live MissionRunner at ``step``."""
    robots = [
        RobotFrame(name=cs, xy=mr.fleet.units[cs].xy, yaw=mr.fleet.units[cs].yaw,
                   trail=list(mr.trails.get(cs, [])), chip=_chip(mr, cs))
        for cs in mr.callsigns
    ]
    return FrameState(
        step=step,
        phase=mr.phase(),
        robots=robots,
        transcript=list(mr.bus.transcript),
        mission_lines=_mission_lines(mr),
        rings=_rings(mr),
        pads=_pads(mr),
        planned_paths=_planned_paths(mr),
        sim_time=sim_time,
    )
