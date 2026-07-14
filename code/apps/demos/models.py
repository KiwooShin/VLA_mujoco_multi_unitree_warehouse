"""models.py — Generic per-frame state the DemoComposer renders.

These dataclasses are the *only* contract between a mission and the composer:
the composer draws whatever they expose and hardcodes no mission shape. A
production script fills a :class:`FrameState` each control step (directly, or via
:func:`code.apps.demos.runner_adapter.frame_state_from_runner`) and hands it to
:meth:`DemoComposer.compose`.

Kept dependency-free (no cv2 / MuJoCo) so the pure layout/effects/text tests can
construct fake states cheaply.
"""

from __future__ import annotations

import dataclasses
from typing import Any, Optional, Sequence, Tuple

XY = Tuple[float, float]
BGR = Tuple[int, int, int]


@dataclasses.dataclass
class RobotFrame:
    """One robot's per-frame render inputs.

    Attributes:
        name: Callsign (selects the accent colour).
        xy: Pelvis world ``(x, y)`` (m) for the BEV marker + ego render.
        yaw: Pelvis yaw (rad) for the heading tick + ego camera.
        trail: World ``(x, y)`` breadcrumb path (drawn strided, bounded cost).
        chip: Short state phrase for the tile chip (e.g. ``"searching storage A"``,
            ``"carrying"``, ``"paused"``).
    """

    name: str
    xy: XY
    yaw: float
    trail: Sequence[XY] = ()
    chip: str = ""


@dataclasses.dataclass
class Ring:
    """A pulsing target ring on the BEV floor (multiple supported per frame).

    Attributes:
        xy: World ``(x, y)`` (m) to ring.
        color: Optional BGR override; ``None`` uses the default gold accent.
        label: Optional short caption drawn beside the ring.
    """

    xy: XY
    color: Optional[BGR] = None
    label: str = ""


@dataclasses.dataclass
class Pad:
    """A delivery-pad highlight (axis-aligned floor rectangle)."""

    xy: XY
    half_x: float
    half_y: float
    label: str = "DELIVERY"


@dataclasses.dataclass
class PlannedPath:
    """A planned-path polyline (owner route), optionally accent-coloured."""

    pts: Sequence[XY]
    color: Optional[BGR] = None


@dataclasses.dataclass
class FrameState:
    """Everything the composer needs to draw one frame.

    Attributes:
        step: Simulation control step (drives comm-glow decay + the default clock).
        phase: Short HUD phrase for the current mission phase.
        robots: Per-robot render inputs; order fixes the ego-tile order.
        transcript: Message-like objects (``sender``/``recipient``/``performative``/
            ``payload``/``t_step``) — rendered generically, so clarify/user lines
            appear automatically when present.
        mission_lines: One or more task/allocator lines under the panel title.
        rings: Zero or more target rings (list -> supports two concurrent goals).
        pads: Zero or more delivery-pad highlights.
        planned_paths: Zero or more planned-route polylines.
        sim_time: Simulated seconds for the HUD; ``None`` -> ``step * sim_dt``.
    """

    step: int
    phase: str = ""
    robots: Sequence[RobotFrame] = ()
    transcript: Sequence[Any] = ()
    mission_lines: Sequence[str] = ()
    rings: Sequence[Ring] = ()
    pads: Sequence[Pad] = ()
    planned_paths: Sequence[PlannedPath] = ()
    sim_time: Optional[float] = None
