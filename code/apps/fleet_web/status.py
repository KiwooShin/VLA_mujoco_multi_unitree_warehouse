"""status.py — Per-robot status derivation for the UI chips.

Maps the raw coordination/motion state read off a robot (via the public
:class:`code.comms.protocol.RobotProtocol` /
:class:`code.fleet.robot_unit.RobotUnit` accessors) into a friendly chip dict
the front-end renders directly. Pure functions over plain values, so no MuJoCo
is needed to unit-test the labelling.
"""

from __future__ import annotations

import dataclasses
from typing import Dict, Optional

# Per-callsign accent colours (hex) mirroring code.fleet.viz torso accents /
# code.fleet.mission_video overlay colours, for the status dots + transcript.
ACCENT_HEX: Dict[str, str] = {
    "Alpha": "#e63946",    # red
    "Bravo": "#3a7bd5",    # blue
    "Charlie": "#e8c33a",  # yellow
    "Delta": "#9e4fd0",    # purple
    "Echo": "#1accbd",     # teal (F6 six-robot scale-up)
    "Foxtrot": "#f78a1a",  # orange
}
DEFAULT_ACCENT: str = "#9aa0a6"

# Friendly labels for each coordination (protocol) state.
_COORD_LABEL: Dict[str, str] = {
    "IDLE": "Standing by",
    "OWNER_QUERYING": "Asking peers",
    "OWNER_DELEGATING": "Coordinating search",
    "OWNER_NAVIGATING": "Fetching",
    "OWNER_DELIVERING": "Delivering",
    "ASSIST_SEARCHING": "Searching",
}


@dataclasses.dataclass(frozen=True)
class RobotSnap:
    """Raw, UI-agnostic snapshot of one robot for a single frame.

    Attributes:
        name: Callsign.
        coord_state: Coordination state name (``RobotProtocol.state.name``).
        motion: Motion state value (``RobotUnit.state.value``: idle/walking/
            paused/arrived/fallen).
        dist_to_goal: Straight-line metres to the current nav goal, or ``None``.
        carrying: Whether this robot is carrying the requested object.
        is_owner: Whether this robot owns the active task.
        task_desc: The active object phrase (owner only), else ``""``.
    """

    name: str
    coord_state: str
    motion: str
    dist_to_goal: Optional[float]
    carrying: bool
    is_owner: bool
    task_desc: str = ""


def accent(name: str) -> str:
    """Return the accent hex colour for a callsign."""
    return ACCENT_HEX.get(name, DEFAULT_ACCENT)


def state_label(snap: RobotSnap) -> str:
    """Return the friendly primary-state label for a chip."""
    if snap.carrying:
        return "Carrying to pad"
    if snap.coord_state == "IDLE":
        if snap.motion == "fallen":
            return "Fallen"
        if snap.motion == "arrived":
            return "Arrived"
        return "Standing by"
    return _COORD_LABEL.get(snap.coord_state, snap.coord_state.title())


def robot_view(snap: RobotSnap) -> Dict[str, object]:
    """Render one robot chip as a JSON-serializable dict.

    Args:
        snap: The raw per-robot snapshot for this frame.

    Returns:
        ``{name, color, state, task, motion, busy, dist}`` where ``task`` is the
        owner's ``"<object> → delivery pad"`` line (or ``"—"``), ``busy`` marks
        a robot engaged in the mission, and ``dist`` is a short metres string.
    """
    label = state_label(snap)
    if snap.is_owner and snap.task_desc:
        task = f"{snap.task_desc} → delivery pad"
    elif snap.coord_state == "ASSIST_SEARCHING":
        task = "assisting the search"
    else:
        task = "—"
    dist = ""
    if snap.dist_to_goal is not None and snap.motion in ("walking", "paused"):
        dist = f"{snap.dist_to_goal:.1f} m"
    busy = snap.coord_state != "IDLE" or snap.carrying
    return {"name": snap.name, "color": accent(snap.name), "state": label,
            "task": task, "motion": snap.motion, "busy": busy, "dist": dist}
