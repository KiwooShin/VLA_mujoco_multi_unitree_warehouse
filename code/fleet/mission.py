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
from code.comms.messages import (ObjectQuery, Performative, TaskKind, TaskSpec,
                                 reconstruct_location)
from code.comms.protocol import RobotProtocol, RobotState
from code.fleet.actions import FleetRobotActions
from code.fleet.allocator import (AllocationResult, RobotPose, allocate,
                                   planned_path_length)
from code.fleet.carry import CarryManager
from code.fleet.fleet import Fleet
from code.fleet.search import (SearchController, free_centroid,
                               search_regions_for_layout)
from code.fleet.visibility import VisibilityConfig, is_object_visible
from code.sim.arena_build import COLORS, SHAPES
from code.warehouse.layout import CALLSIGNS, WarehouseLayout, hero_layout

XY = Tuple[float, float]

_COLOR_WORDS: Tuple[str, ...] = tuple(c[0] for c in COLORS)
_SHAPE_WORDS: Tuple[str, ...] = tuple(s[0] for s in SHAPES)

# F4 — generic object reference ("the object" / "an object" / "something ...").
# A bare noun ("object"/"item") is always generic; a pronoun ("something"/
# "anything") only counts as an object when a fetch verb is present, so "do
# something useful" stays unresolvable.
_GENERIC_NOUNS: Tuple[str, ...] = ("object", "item")
_GENERIC_PRONOUNS: Tuple[str, ...] = ("something", "anything")
_FETCH_VERBS: Tuple[str, ...] = (
    "bring", "fetch", "get", "grab", "carry", "deliver", "take", "retrieve")


def _is_generic_reference(low: str) -> bool:
    """Whether a colour/shape-free body names the object generically (F4)."""
    if any(f" {n} " in low for n in _GENERIC_NOUNS):
        return True
    if any(f" {p} " in low for p in _GENERIC_PRONOUNS):
        return any(f" {v} " in low for v in _FETCH_VERBS)
    return False


def resolve_query(body: str) -> Optional[ObjectQuery]:
    """Extract a colour/shape :class:`ObjectQuery` from an instruction body.

    A colour and/or shape word yields a specific query. When neither is present
    a *generic* reference — "the object", "an object", or "something" alongside a
    fetch verb (F4) — yields the wildcard :class:`ObjectQuery` (matches any
    object; the first one a robot finds wins). Anything else is unresolvable.

    Args:
        body: The instruction with the addressee removed (e.g. "fetch the red
            cube to the delivery pad", or "bring the object to the destination").

    Returns:
        An :class:`ObjectQuery` (specific or generic), or ``None`` if no object
        can be resolved.
    """
    low = f" {body.lower()} "
    color = next((c for c in _COLOR_WORDS if f" {c}" in low), None)
    shape = next((s for s in _SHAPE_WORDS if f" {s}" in low), None)
    if color is None and shape is None:
        if _is_generic_reference(low):
            return ObjectQuery(None, None)  # generic: match any object
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

    outcome: str                    # "complete" | "failed" | "timeout" | "stopped"
    owner: Optional[str]
    steps: int
    any_fell: bool
    object_on_pad: bool
    task_complete_sent: bool
    mean_vla_infer_ms: float = 0.0  # F5 locomotion step cost (0.0 in teacher mode)


class MissionRunner:
    """Owns the fleet, comms and coordination for one warehouse mission."""

    def __init__(self, layout: Optional[WarehouseLayout] = None, *,
                 seed: int = 0, objects: Optional[List[dict]] = None,
                 callsigns: Sequence[str] = CALLSIGNS, use_gpu: bool = True,
                 teachers: Optional[Dict[str, object]] = None,
                 reply_deadline_steps: int = 60,
                 search_deadline_steps: int = 6000,
                 vis_cfg: Optional[VisibilityConfig] = None,
                 regions: Optional[Sequence[str]] = None,
                 perception_mode: str = "oracle",
                 locomotion: str = "teacher",
                 vla_ckpt: Optional[str] = None,
                 vla_device: Optional[str] = None) -> None:
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
            regions: Region labels for search delegation + allocation. When
                ``None`` (default) they follow the layout (F6): the room names on
                a multi-room layout (spawn room excluded — the fleet covers it
                from its bays), the north/middle/south thirds on the hero hall.
            perception_mode: ``"oracle"`` (default; the pure geometric visibility
                oracle drives ``can_see`` — deterministic, for determinism-sensitive
                evals) or ``"groundnet"`` (the real learned detector CONFIRMS each
                oracle-visible sighting and refines the reported location). The
                protocol and message flow are identical in both modes.
            locomotion: ``"teacher"`` (default; WBC walk policy) or ``"vla"``
                (F5: the trained GroundedNav policy, one model shared by the whole
                fleet). Unchanged mission logic in both modes.
            vla_ckpt: GroundedNav checkpoint (``locomotion="vla"``; None -> F5
                default).
            vla_device: Torch device for the shared VLA policy (None -> auto).
        """
        self.layout = layout or hero_layout()
        self.callsigns: List[str] = list(callsigns)
        self._vis = vis_cfg or VisibilityConfig()
        # F6: search regions follow the layout unless the caller pins them.
        self._regions = (tuple(regions) if regions is not None
                         else search_regions_for_layout(self.layout))
        self._rooms = tuple(self.layout.rooms)
        self.perception_mode = perception_mode
        self.locomotion = locomotion

        self.fleet = Fleet(self.layout, goals={}, callsigns=self.callsigns,
                           use_gpu=use_gpu, teachers=teachers, build_viz=True,
                           seed=seed, objects=objects, locomotion=locomotion,
                           vla_ckpt=vla_ckpt, vla_device=vla_device)
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
        # Rooms-mode search delegation assigns each peer its nearest unsearched
        # room by A* distance; hero thirds keep the in-order default (None).
        assigner = self._room_region_assigner if self._rooms else None
        self._search: Dict[str, SearchController] = {}
        self.actions: Dict[str, FleetRobotActions] = {}
        self.protocols: Dict[str, RobotProtocol] = {}
        for cs in self.callsigns:
            peers = [c for c in self.callsigns if c != cs]
            search_ctrl = SearchController(self.fleet.units[cs], rooms=self._rooms)
            act = FleetRobotActions(cs, self.fleet.units[cs], self.scene_cfg,
                                    search_ctrl, self.carry, vis_cfg=self._vis,
                                    perception=self.perceptions.get(cs),
                                    layout=self.layout)
            self._search[cs] = search_ctrl
            self.actions[cs] = act
            self.protocols[cs] = RobotProtocol(
                cs, self.bus, act, peers, search_regions=self._regions,
                reply_deadline_steps=reply_deadline_steps,
                search_deadline_steps=search_deadline_steps,
                region_assigner=assigner)

        self._t = 0
        self._steps = 0
        self._submitted = 0
        self._mission_base = 0          # transcript index the current mission began at
        self._cancelled = False
        self.task: Optional[TaskSpec] = None
        self.primary_owner: Optional[str] = None
        self.allocation: Optional[AllocationResult] = None
        self.trails: Dict[str, List[XY]] = {c: [] for c in self.callsigns}
        # F2: the deferred target symbol. ``_reported_target`` is the object's
        # first reported/seen position (drawn only once known); ``_target_index``
        # locks onto the specific object once it is picked up (then tracked live).
        self._reported_target: Optional[XY] = None
        self._target_index: Optional[int] = None
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
        """Run the detector on the owner's target while it navigates toward it.

        The accepted confirmation (a well-framed, close-range look) additionally
        STEERS the owner's fetch goal onto the detector's fresh estimate via
        :meth:`~code.comms.protocol.RobotProtocol.refine_nav_goal` — the D-14
        fix: a stale/off found-location can no longer strand the owner outside
        pickup range, because the approach itself continuously refines the goal.
        """
        from code.fleet.perception_bridge import (CONFIRM_RANGE_M,
                                                   GROUNDING_HALF_FOV_DEG,
                                                   REFINE_MIN_CONF)
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
        det = self.perceptions[owner].confirm(self.task.query, (rx, ry, yaw),
                                              oracle_xy=tgt)
        # D-14: a confident, well-framed close-range confirmation refines the goal.
        if det is not None and det.confidence >= REFINE_MIN_CONF:
            self.protocols[owner].refine_nav_goal(det.world_xy, self._t,
                                                  robot_xy=(rx, ry))

    # -- clock ------------------------------------------------------------
    def _now(self) -> int:
        """Current simulation step (bus clock)."""
        return self._t

    # -- lifecycle / submission -------------------------------------------
    def _mission_active(self) -> bool:
        """Whether a submitted mission is still in flight (no terminal outcome yet)."""
        return self._submitted > 0 and not self._terminal_seen()

    def reset_mission(self) -> None:
        """Clear per-mission state so a fresh order can run on this runner.

        Reuses the built fleet, walk teachers, message bus and per-robot
        protocols — no MuJoCo rebuild. The physical world is *continuous* across
        missions: robots remain where they stopped and a delivered object stays
        on the pad. Only the task / owner / allocation bookkeeping, the deferred
        target tracking and this runner's trails are cleared; the bus transcript
        is retained as one continuous log, with ``_mission_base`` marking where
        the next mission's messages begin (so a prior mission's ``TASK_COMPLETE``
        can never satisfy the next mission's done-check).

        Safe to call once the previous mission reached a terminal outcome; it is
        also invoked automatically by :meth:`submit` when reused.
        """
        self.task = None
        self.primary_owner = None
        self.allocation = None
        self._pending_fleet = []
        self._fleet_queued_notified = False
        self._reported_target = None
        self._target_index = None
        self._cancelled = False
        self.trails = {c: [] for c in self.callsigns}
        self._mission_base = len(self.bus.transcript)
        self._submitted = 0
        for p in self.perceptions.values():
            p.reset()

    def submit(self, text: str) -> TaskSpec:
        """Parse and post a natural-language order onto the bus.

        A :class:`MissionRunner` runs one mission at a time, but the same runner
        can run successive missions: submitting after the previous mission has
        finished auto-resets the per-mission state (see :meth:`reset_mission`).
        Submitting while a mission is still in flight is rejected (it would
        clobber the active task and owner attribution).

        Args:
            text: The raw order (addressed to a callsign or the fleet).

        Returns:
            The resolved :class:`TaskSpec`.

        Raises:
            ValueError: If no object can be resolved from the order.
            RuntimeError: If a mission is already in progress on this runner.
        """
        if self._mission_active():
            raise RuntimeError(
                "a mission is already in progress; call reset_mission() or wait "
                "for the current mission to finish before submitting another")
        if self._submitted > 0:
            self.reset_mission()  # a previous mission finished — start clean
        self._mission_base = len(self.bus.transcript)
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
                          vis_cfg=self._vis, regions=self._regions,
                          rooms=self._rooms)
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

    # -- room-aware search delegation (F6) --------------------------------
    def _room_region_assigner(self, peers: Sequence[str],
                              regions: Sequence[str]) -> List[Tuple[str, str]]:
        """Assign each searcher its nearest unsearched room by A* distance.

        Greedy global nearest: repeatedly pair the peer/room with the shortest
        planned A* path (ties broken by callsign then room name for determinism),
        until peers or rooms run out. Handed to each owner's protocol as its
        region assigner on the multi-room layout.
        """
        remaining_peers = list(peers)
        remaining_regions = list(regions)
        centroids = {r: free_centroid(self.scene_cfg, r, rooms=self._rooms)
                     for r in remaining_regions}
        result: List[Tuple[str, str]] = []
        while remaining_peers and remaining_regions:
            best: Optional[Tuple[float, str, str]] = None
            for peer in remaining_peers:
                pxy = self.fleet.units[peer].xy
                for region in remaining_regions:
                    cost = planned_path_length(self.scene_cfg, pxy,
                                               centroids[region])
                    key = (cost, peer, region)
                    if best is None or key < best:
                        best = key
            _, peer, region = best
            result.append((peer, region))
            remaining_peers.remove(peer)
            remaining_regions.remove(region)
        return result

    # -- main loop --------------------------------------------------------
    def run(self, max_steps: int,
            on_step: Optional[Callable[["MissionRunner", int], None]] = None
            ) -> MissionResult:
        """Drive the mission to completion, failure or the step budget.

        Args:
            max_steps: Hard cap on control steps.
            on_step: Optional ``on_step(runner, step)`` hook (e.g. video capture)
                invoked after each fully-updated step. Returning ``False`` cancels
                the run promptly (a clean public stop contract for a live host);
                the result outcome is then ``"stopped"``.

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
            self._update_target_knowledge()
            for cs in self.callsigns:
                self.trails[cs].append(self.fleet.units[cs].xy)
            self._steps = t + 1
            if on_step is not None and on_step(self, t) is False:
                self._cancelled = True
                break
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

    def _mission_transcript(self):
        """The current mission's slice of the (continuous) bus transcript."""
        return self.bus.transcript[self._mission_base:]

    def _terminal_seen(self) -> bool:
        """Whether THIS mission posted a TASK_COMPLETE/TASK_FAILED to the user."""
        return any(m.performative in (Performative.TASK_COMPLETE,
                                      Performative.TASK_FAILED)
                   for m in self._mission_transcript())

    def _result(self) -> MissionResult:
        """Assemble the mission summary from the transcript + world state."""
        outcome = "stopped" if self._cancelled else "timeout"
        complete = False
        for m in self._mission_transcript():
            if m.performative is Performative.TASK_COMPLETE:
                outcome, complete = "complete", True
            elif m.performative is Performative.TASK_FAILED:
                outcome = "failed" if not self._cancelled else outcome
        return MissionResult(
            outcome=outcome, owner=self.primary_owner, steps=self._steps,
            any_fell=self.fleet.any_fell, object_on_pad=self.object_on_pad(),
            task_complete_sent=complete,
            mean_vla_infer_ms=self.fleet.mean_vla_infer_ms())

    # -- deferred target symbol (F2) --------------------------------------
    def _update_target_knowledge(self) -> None:
        """Advance the deferred-target state one step (call after protocols step).

        Locks ``_target_index`` onto the specific object as soon as any robot
        picks one up (thereafter the ring tracks it live — in-hand, then on the
        pad), and records ``_reported_target`` the first time the object is
        actually located (a peer/searcher report or the owner's own sighting),
        so the ring is drawn only from that moment, at the reported position.
        """
        if self._target_index is None:
            for cs in self.callsigns:
                idx = self.carry.carried_index(cs)
                if idx is None:
                    idx = self.carry.released.get(cs)
                if idx is not None:
                    self._target_index = idx
                    break
        if self._reported_target is None:
            self._reported_target = self._first_sighting_location()

    def _first_sighting_location(self) -> Optional[XY]:
        """The object's first reported / first-seen position this mission, or None."""
        for m in self._mission_transcript():
            if m.performative is Performative.REPORT_FOUND:
                return reconstruct_location(m.payload)
            if (m.performative is Performative.REPORT_VISIBILITY
                    and m.payload.get("visible")):
                return reconstruct_location(m.payload)
        owner = self.primary_owner
        if owner is not None:
            lt = self.protocols[owner].located_target
            if lt is not None:
                return (float(lt[0]), float(lt[1]))
        return None

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
        """Current world (x, y) of the requested object (tracks carry).

        Ground-truth position of the (first) matching object — used by the
        perception confirmer. The video/BEV should instead call
        :meth:`known_target_xy`, which honours the F2 deferral.
        """
        if self.task is None:
            return None
        if self._target_index is not None:
            obj = self.scene_cfg["objects"][self._target_index]
            return (float(obj["x"]), float(obj["y"]))
        for obj in self.scene_cfg["objects"]:
            if self.task.query.matches(obj):
                return (float(obj["x"]), float(obj["y"]))
        return None

    def known_target_xy(self) -> Optional[XY]:
        """Where to draw the target ring — or ``None`` until the object is known (F2).

        No symbol is drawn until a robot has actually located the object; from
        then it sits at the *reported* position, and once a robot picks the
        object up the ring tracks that specific object live (in-hand, then on the
        pad). This is the deferred, honest target marker for the video + BEV.
        """
        if self._target_index is not None:
            obj = self.scene_cfg["objects"][self._target_index]
            return (float(obj["x"]), float(obj["y"]))
        return self._reported_target

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
