"""mission.py — End-to-end collaborative fetch-to-destination mission runner.

:class:`MissionRunner` is the Phase-4 integration: it owns the co-simulation
:class:`~code.fleet.fleet.Fleet`, the :class:`~code.comms.bus.MessageBus`, one
:class:`~code.comms.protocol.RobotProtocol` per robot wired onto a
:class:`~code.fleet.actions.FleetRobotActions` bridge, the shared mock
:class:`~code.fleet.carry.CarryManager`, and the path-shortest
:mod:`~code.fleet.allocator`.

``submit(text)`` turns a natural-language order ("Alpha, fetch the red cube to the
delivery pad" / "someone bring me the red cube") into an addressed task via
:mod:`code.comms.addressing` + a colour/shape resolver. ``run(max_steps)`` then
drives the closed loop: each control step advances physics, re-poses carried
objects, ticks searches, pumps the allocator, and steps every protocol — until a
``TASK_COMPLETE``/``TASK_FAILED`` reaches the user or the step budget runs out.
The full transcript and per-robot trails are retained for the video.
"""

from __future__ import annotations

import dataclasses
import math
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import mujoco

from code.comms.addressing import parse_addressed_instruction
from code.comms.bus import MessageBus
from code.comms.messages import (ObjectQuery, Performative, TaskKind, TaskSpec)
from code.comms.protocol import RobotProtocol, RobotState
from code.fleet.actions import FleetRobotActions
from code.fleet.allocator import AllocationResult, RobotPose, allocate
from code.fleet.carry import CarryManager
from code.fleet.fleet import Fleet
from code.fleet.search import SEARCH_REGIONS, SearchController
from code.fleet.visibility import VisibilityConfig, is_object_visible
from code.sim.arena_build import COLORS, SHAPES
from code.warehouse.layout import CALLSIGNS, WarehouseLayout, hero_layout

XY = Tuple[float, float]

_COLOR_WORDS: Tuple[str, ...] = tuple(c[0] for c in COLORS)
_SHAPE_WORDS: Tuple[str, ...] = tuple(s[0] for s in SHAPES)


def resolve_query(body: str) -> Optional[ObjectQuery]:
    """Extract a colour/shape :class:`ObjectQuery` from an instruction body.

    Args:
        body: The instruction with the addressee removed (e.g. "fetch the red
            cube to the delivery pad").

    Returns:
        An :class:`ObjectQuery`, or ``None`` if neither a colour nor a shape word
        is present.
    """
    low = f" {body.lower()} "
    color = next((c for c in _COLOR_WORDS if f" {c}" in low), None)
    shape = next((s for s in _SHAPE_WORDS if f" {s}" in low), None)
    if color is None and shape is None:
        return None
    return ObjectQuery(color, shape)


def delivery_xy(layout: WarehouseLayout) -> Tuple[str, XY]:
    """Return the delivery pad's ``(name, (x, y))`` from a layout."""
    for z in layout.zones:
        if z.name == "delivery":
            return ("delivery pad", (float(z.cx), float(z.cy)))
    return ("delivery pad", (0.0, 0.0))


# ---------------------------------------------------------------------------
# Phase labelling (for the video HUD)
# ---------------------------------------------------------------------------
_OWNER_PHASE: Dict[RobotState, str] = {
    RobotState.OWNER_QUERYING: "QUERYING PEERS",
    RobotState.OWNER_DELEGATING: "DELEGATING SEARCH",
    RobotState.OWNER_NAVIGATING: "FETCHING",
    RobotState.OWNER_DELIVERING: "CARRYING TO PAD",
}


@dataclasses.dataclass
class MissionResult:
    """Summary of a finished mission (for evals)."""

    outcome: str                    # "complete" | "failed" | "timeout"
    owner: Optional[str]
    steps: int
    any_fell: bool
    object_on_pad: bool
    task_complete_sent: bool


class MissionRunner:
    """Owns the fleet, comms and coordination for one warehouse mission."""

    def __init__(self, layout: Optional[WarehouseLayout] = None, *,
                 seed: int = 0, objects: Optional[List[dict]] = None,
                 callsigns: Sequence[str] = CALLSIGNS, use_gpu: bool = True,
                 teachers: Optional[Dict[str, object]] = None,
                 reply_deadline_steps: int = 60,
                 search_deadline_steps: int = 6000,
                 vis_cfg: Optional[VisibilityConfig] = None,
                 regions: Sequence[str] = SEARCH_REGIONS,
                 perception_mode: str = "oracle") -> None:
        """Build the fleet, bus, per-robot protocols/bridges and carry manager.

        Args:
            layout: Warehouse layout (defaults to :func:`hero_layout`).
            seed: RNG seed for default object placement (ignored if ``objects``).
            objects: Explicit scenario object placement (color/shape/x/y/size).
            callsigns: Robots to run (order fixes pause + tie-break priority).
            use_gpu: Prefer CUDA for the walk policies.
            teachers: Optional pre-loaded teachers to reuse across missions.
            reply_deadline_steps: Peer visibility-reply timeout (steps).
            search_deadline_steps: Delegated-search timeout (steps).
            vis_cfg: Visibility-oracle geometry.
            regions: Region labels for search delegation + allocation.
            perception_mode: ``"oracle"`` (default; the pure geometric visibility
                oracle drives ``can_see`` — deterministic, for determinism-sensitive
                evals) or ``"groundnet"`` (the real learned detector CONFIRMS each
                oracle-visible sighting and refines the reported location). The
                protocol and message flow are identical in both modes.
        """
        self.layout = layout or hero_layout()
        self.callsigns: List[str] = list(callsigns)
        self._vis = vis_cfg or VisibilityConfig()
        self._regions = tuple(regions)
        self.perception_mode = perception_mode

        self.fleet = Fleet(self.layout, goals={}, callsigns=self.callsigns,
                           use_gpu=use_gpu, teachers=teachers, build_viz=True,
                           seed=seed, objects=objects)
        self.scene_cfg = self.fleet.scene_cfg
        self._dest_name, self._dest_xy = delivery_xy(self.layout)

        # The shared viz model is display-only (never stepped): robots roam and
        # transiently overlap in it, and a carried object is re-posed onto an
        # arm. Its ``mj_forward`` only needs kinematics for rendering + hand
        # poses, so disable contact/constraint solving to keep it from ever
        # tripping ``FactorizeHessian`` on a degenerate overlap.
        if self.fleet.viz is not None:
            self.fleet.viz.model.opt.disableflags |= (
                int(mujoco.mjtDisableBit.mjDSBL_CONTACT)
                | int(mujoco.mjtDisableBit.mjDSBL_CONSTRAINT))

        self.bus = MessageBus(self._now)
        self.carry = CarryManager(self.fleet, self.scene_cfg)
        # GROUND_NET perception (groundnet mode only): shared detector weights +
        # one grounding renderer for the whole fleet, one isolated RobotPerception
        # per robot. Empty in oracle mode (import graph + behaviour unchanged).
        self.perceptions: Dict[str, object] = {}
        self._grounding_renderer = None
        # Accepted detector confirmations, ``(step, DetectionResult)`` (video/eval).
        self.confirmations: List[Tuple[int, object]] = []
        if perception_mode == "groundnet":
            self._build_perceptions()
        self._search: Dict[str, SearchController] = {}
        self.actions: Dict[str, FleetRobotActions] = {}
        self.protocols: Dict[str, RobotProtocol] = {}
        for cs in self.callsigns:
            peers = [c for c in self.callsigns if c != cs]
            search_ctrl = SearchController(self.fleet.units[cs])
            act = FleetRobotActions(cs, self.fleet.units[cs], self.scene_cfg,
                                    search_ctrl, self.carry, vis_cfg=self._vis,
                                    perception=self.perceptions.get(cs))
            self._search[cs] = search_ctrl
            self.actions[cs] = act
            self.protocols[cs] = RobotProtocol(
                cs, self.bus, act, peers, search_regions=self._regions,
                reply_deadline_steps=reply_deadline_steps,
                search_deadline_steps=search_deadline_steps)

        self._t = 0
        self._steps = 0
        self._submitted = 0
        self.task: Optional[TaskSpec] = None
        self.primary_owner: Optional[str] = None
        self.allocation: Optional[AllocationResult] = None
        self.trails: Dict[str, List[XY]] = {c: [] for c in self.callsigns}
        # Fleet requests that arrived with no allocatable robot: retried each
        # drain, queued-once notice sent, failed if still unassigned at budget end.
        self._pending_fleet: List[TaskSpec] = []
        self._fleet_queued_notified = False

    # -- perception (groundnet mode) --------------------------------------
    def _build_perceptions(self) -> None:
        """Build the shared GROUND_NET detector + renderer and per-robot confirmers.

        The detector weights are loaded ONCE and the single model object is
        shared across all four per-robot :class:`RobotPerception` states (only
        their track-hysteresis + heatmap caches are per-robot). Falls back to the
        classical HSV pipeline per robot if the checkpoint is missing.
        """
        from code.fleet.perception_bridge import (GroundingCamRenderer,
                                                   RobotPerception,
                                                   load_shared_detector)
        if self.fleet.viz is None:
            raise ValueError("groundnet perception requires a fleet built with build_viz=True")
        detector = load_shared_detector()
        self._grounding_renderer = GroundingCamRenderer(self.fleet.viz.model)
        for cs in self.callsigns:
            self.perceptions[cs] = RobotPerception(
                cs, self.fleet.viz, detector=detector,
                renderer=self._grounding_renderer)

    def _perception_step(self, t: int) -> None:
        """Groundnet-mode per-step perception: owner-approach confirmation + drain.

        No-op in oracle mode. The ``can_see`` confirmations (single-shot at
        find-time) already ran inside the protocol step; here the navigating
        owner additionally runs the learned detector on the target it is walking
        toward — the natural "keep perceiving the target during approach" signal,
        which is where the grounding cam gets well-framed looks (the visibility
        edge, by contrast, is often at a wide bearing outside its narrower FOV).
        This is telemetry only: it never changes the oracle-gated found location
        or the protocol/message flow.
        """
        if self.perception_mode != "groundnet":
            return
        self._owner_approach_confirm()
        for cs in self.callsigns:
            p = self.perceptions.get(cs)
            if p is None:
                continue
            ev = p.pop_confirmation()
            if ev is not None:
                self.confirmations.append((t, ev))

    def _owner_approach_confirm(self) -> None:
        """Run the detector on the owner's target while it navigates toward it."""
        from code.fleet.perception_bridge import (CONFIRM_RANGE_M,
                                                   GROUNDING_HALF_FOV_DEG)
        owner = self.primary_owner
        if (owner is None or owner not in self.perceptions or self.task is None
                or self.carry.carrying(owner)
                or self.protocols[owner].state is not RobotState.OWNER_NAVIGATING):
            return
        tgt = self.target_xy()
        if tgt is None:
            return
        u = self.fleet.units[owner]
        rx, ry = u.xy
        yaw = u.yaw
        walls = self.scene_cfg.get("walls", [])
        if not is_object_visible((rx, ry), yaw, u.base_height, tgt, walls,
                                 cfg=self._vis):
            return  # occluded / out of oracle FOV -> nothing to confirm
        dx, dy = tgt[0] - rx, tgt[1] - ry
        dist = math.hypot(dx, dy)
        bearing = math.degrees((math.atan2(dy, dx) - yaw + math.pi)
                               % (2.0 * math.pi) - math.pi)
        if dist > CONFIRM_RANGE_M or abs(bearing) > GROUNDING_HALF_FOV_DEG:
            return  # outside the grounding cam's usable range / narrower FOV
        self.perceptions[owner].confirm(self.task.query, (rx, ry, yaw),
                                        oracle_xy=tgt)

    # -- clock ------------------------------------------------------------
    def _now(self) -> int:
        """Current simulation step (bus clock)."""
        return self._t

    # -- submission -------------------------------------------------------
    def _mission_active(self) -> bool:
        """Whether a submitted mission is still in flight (no terminal outcome yet)."""
        return self._submitted > 0 and not self._terminal_seen()

    def submit(self, text: str) -> TaskSpec:
        """Parse and post a natural-language order onto the bus.

        One :class:`MissionRunner` runs exactly one mission at a time. Submitting
        a second order while the first is still in flight would silently clobber
        the active task and owner attribution, so it is rejected.

        Args:
            text: The raw order (addressed to a callsign or the fleet).

        Returns:
            The resolved :class:`TaskSpec`.

        Raises:
            ValueError: If no object colour/shape can be resolved from the order.
            RuntimeError: If a mission is already in progress on this runner.
        """
        if self._mission_active():
            raise RuntimeError(
                "a mission is already in progress; one MissionRunner handles one "
                "mission at a time")
        addr = parse_addressed_instruction(text, self.callsigns)
        query = resolve_query(addr.body)
        if query is None:
            raise ValueError(f"could not resolve an object from {addr.body!r}")
        task = TaskSpec(TaskKind.FETCH, query, self._dest_name, self._dest_xy,
                        requester="user")
        self.task = task
        if addr.is_fleet:
            self.bus.post("user", "fleet", Performative.FLEET_REQUEST, {"task": task})
        else:
            self.primary_owner = addr.recipient
            self.bus.post("user", addr.recipient, Performative.REQUEST_TASK,
                          {"task": task})
        self._submitted += 1
        return task

    # -- allocator --------------------------------------------------------
    def _run_allocator(self) -> None:
        """Drain fleet requests, assign the path-shortest idle robot, or queue.

        Newly drained requests join any still-unassigned ones; each is retried
        every drain. A request with no allocatable robot is kept (not dropped) and
        the user is told once that the order is queued.
        """
        for msg in self.bus.drain(self.bus.allocator_inbox):
            if msg.performative is Performative.FLEET_REQUEST:
                self._pending_fleet.append(msg.payload["task"])
        if not self._pending_fleet:
            return
        still_pending: List[TaskSpec] = []
        for task in self._pending_fleet:
            if not self._try_allocate(task):
                still_pending.append(task)
        self._pending_fleet = still_pending

    def _try_allocate(self, task: TaskSpec) -> bool:
        """Attempt to assign one task to an idle robot; return whether it stuck."""
        poses = {cs: RobotPose(u.xy, u.yaw, u.base_height)
                 for cs, u in self.fleet.units.items()}
        idle = [cs for cs in self.callsigns if self.protocols[cs].is_idle()]
        result = allocate(poses, self.scene_cfg, task.query, idle,
                          vis_cfg=self._vis, regions=self._regions)
        if result.winner is None:
            if not self._fleet_queued_notified:
                self._fleet_queued_notified = True
                self.bus.post("allocator", "user", Performative.STATUS_UPDATE,
                              {"text": "all robots busy - order queued"})
            return False
        self.allocation = result
        self.primary_owner = result.winner
        self.bus.post("allocator", "user", Performative.STATUS_UPDATE,
                      {"text": result.describe()})
        self.bus.post("allocator", result.winner,
                      Performative.REQUEST_TASK, {"task": task})
        return True

    # -- main loop --------------------------------------------------------
    def run(self, max_steps: int,
            on_step: Optional[Callable[["MissionRunner", int], None]] = None
            ) -> MissionResult:
        """Drive the mission to completion, failure or the step budget.

        Args:
            max_steps: Hard cap on control steps.
            on_step: Optional ``on_step(runner, step)`` hook (e.g. video capture)
                invoked after each fully-updated step.

        Returns:
            A :class:`MissionResult`.
        """
        # Per-mission reset of every robot's detector track state (no cross-
        # mission leak — the singleton bug the baseline survey flagged).
        for p in self.perceptions.values():
            p.reset()
        for t in range(max_steps):
            self._t = t
            self._run_allocator()
            self.fleet.step_all()
            self.carry.update()
            for cs in self.callsigns:
                self._search[cs].tick()
            for cs in self.callsigns:
                self.protocols[cs].step(t)
            self._perception_step(t)
            for cs in self.callsigns:
                self.trails[cs].append(self.fleet.units[cs].xy)
            self._steps = t + 1
            if on_step is not None:
                on_step(self, t)
            if self._is_done():
                break
        # Budget exhausted with a fleet order that never found a free robot:
        # fail it explicitly so the user gets a terminal answer, not silence.
        if self._pending_fleet and not self._terminal_seen():
            self.bus.post("allocator", "user", Performative.TASK_FAILED,
                          {"reason": "no robot became available to take the order"})
            self._pending_fleet = []
        return self._result()

    def _is_done(self) -> bool:
        """True once a terminal result has reached the user and all robots idle."""
        if self._submitted == 0 or not self._terminal_seen():
            return False
        if self._pending_fleet:
            return False
        if self.bus.pending(self.bus.allocator_inbox) > 0:
            return False
        return all(p.is_idle() for p in self.protocols.values())

    def _terminal_seen(self) -> bool:
        """Whether a TASK_COMPLETE/TASK_FAILED has been posted to the user."""
        return any(m.performative in (Performative.TASK_COMPLETE,
                                      Performative.TASK_FAILED)
                   for m in self.bus.transcript)

    def _result(self) -> MissionResult:
        """Assemble the mission summary from the transcript + world state."""
        outcome = "timeout"
        complete = False
        for m in self.bus.transcript:
            if m.performative is Performative.TASK_COMPLETE:
                outcome, complete = "complete", True
            elif m.performative is Performative.TASK_FAILED:
                outcome = "failed"
        return MissionResult(
            outcome=outcome, owner=self.primary_owner, steps=self._steps,
            any_fell=self.fleet.any_fell, object_on_pad=self.object_on_pad(),
            task_complete_sent=complete)

    # -- introspection (video / evals) -----------------------------------
    def object_on_pad(self, tol: float = 1.0) -> bool:
        """Whether the requested object now rests within the delivery pad."""
        if self.task is None:
            return False
        for obj in self.scene_cfg["objects"]:
            if self.task.query.matches(obj):
                if (abs(float(obj["x"]) - self._dest_xy[0]) <= tol
                        and abs(float(obj["y"]) - self._dest_xy[1]) <= tol):
                    return True
        return False

    def target_xy(self) -> Optional[XY]:
        """Current world (x, y) of the requested object (tracks carry)."""
        if self.task is None:
            return None
        for obj in self.scene_cfg["objects"]:
            if self.task.query.matches(obj):
                return (float(obj["x"]), float(obj["y"]))
        return None

    def phase(self) -> str:
        """A short HUD phrase describing what the fleet is doing right now."""
        for cs in self.callsigns:
            if self.carry.carrying(cs):
                return "CARRYING TO PAD"
        if self.primary_owner is not None:
            st = self.protocols[self.primary_owner].state
            if st in _OWNER_PHASE:
                return _OWNER_PHASE[st]
        searching = [self._search[cs].region for cs in self.callsigns
                     if self._search[cs].active and self._search[cs].region]
        if searching:
            return f"SEARCHING {'/'.join(sorted(set(searching)))}"
        if self._terminal_seen():
            return "DONE"
        return "STANDING BY"

    @property
    def transcript_lines(self) -> List[str]:
        """The full comms transcript as demo caption lines."""
        return self.bus.transcript_lines()

    def close(self) -> None:
        """Release the fleet's viz renderers (and the grounding renderer)."""
        if self._grounding_renderer is not None:
            self._grounding_renderer.close()
            self._grounding_renderer = None
        self.fleet.close()
