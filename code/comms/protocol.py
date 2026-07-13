"""protocol.py — Per-robot coordination state machine for the fleet comms layer.

Each robot runs one :class:`RobotProtocol` instance (no globals). The protocol
is pumped once per simulation step via :meth:`RobotProtocol.step`; on each step it
drains its inbox, reacts to messages, and advances any time- or perception-driven
state, issuing physical actions through the narrow :class:`RobotActions` interface
and posting new messages onto the shared :class:`~code.comms.bus.MessageBus`.

The encoded protocol (docs/multi_plan.md sec 1):

* **Task owner** (received ``REQUEST_TASK``): check own view -> else query peers
  one-by-one -> if a peer sees it, go there -> else ``COMMAND_SEARCH`` idle peers
  partitioned by region -> await ``REPORT_FOUND`` -> call off the other searchers
  -> navigate / pick up / deliver -> milestone ``STATUS_UPDATE``\\ s and a final
  ``TASK_COMPLETE`` to the requester (or ``TASK_FAILED`` if the search is
  exhausted).
* **Peer** answering ``QUERY_VISIBILITY``: reply with its own view (reflex).
* **Searcher** commanded via ``COMMAND_SEARCH``: ``ACCEPT`` if idle (else
  ``REJECT`` busy); on finding the object ``REPORT_FOUND`` to the commander
  **only**, then stop.

Need-to-know is enforced *structurally*: only an owner (a protocol holding a task)
can message the requester, and searchers report a find solely to their commander —
helpers never address ``"user"``.
"""

from __future__ import annotations

import abc
import dataclasses
import enum
import math
from typing import Callable, Dict, List, Optional, Sequence, Set, Tuple

from code.comms.bus import MessageBus
from code.comms.messages import (Message, ObjectQuery, Performative, TaskSpec,
                                 reconstruct_location, relative_report_payload)

XY = Tuple[float, float]

# A region-assignment strategy: given the idle peers available to search (in
# reserve order) and the region labels to cover, return the ``(peer, region)``
# pairs to command. The default (``None``) zips peers to regions in order; the
# fleet layer injects an A*-nearest-room assigner for the multi-room layout.
RegionAssigner = Callable[[Sequence[str], Sequence[str]], List[Tuple[str, str]]]

# How many times a missed mock pickup is re-attempted (re-approach) before the
# owner declares the fetch failed. One retry is enough for a transient miss.
MAX_PICKUP_RETRIES: int = 1

# Close-range goal refinement (D-14 delivery outlier). While the owner walks to a
# found location, the fleet's continuous approach-confirmation stream feeds fresh,
# well-framed detector estimates to :meth:`RobotProtocol.refine_nav_goal`, which
# steers the fetch goal onto the better estimate so a stale/off report can't
# strand the owner outside pickup range. Bounded so it can never thrash: at most
# one refinement every ``GOAL_REFINE_INTERVAL`` steps, only a plausible
# *same-object* nudge (moves more than ``GOAL_REFINE_MIN_DELTA_M`` but no farther
# than ``GOAL_REFINE_MAX_DELTA_M`` from the current goal), and only while the
# owner is still farther than ``GOAL_REFINE_MIN_RANGE_M`` (the pickup radius) from
# the goal. In oracle mode the fleet never calls this (the report is exact), so
# oracle behaviour is byte-identical.
GOAL_REFINE_INTERVAL: int = 50          # K: min steps between refinements
GOAL_REFINE_MAX_DELTA_M: float = 2.5    # same-object sanity (reject far jumps)
GOAL_REFINE_MIN_DELTA_M: float = 0.3    # ignore sub-threshold nudges
GOAL_REFINE_MIN_RANGE_M: float = 0.6    # only refine while > pickup radius away
# Wider same-object refinement gate applied when the fetch goal came from an
# *approximate* (long-range) report — see CONFIRM-THEN-REPORT below. A report the
# reporter itself flagged ``approx`` was made without a close-range confirm, so
# the owner's approach may need to correct it by more than a normal same-object
# nudge; the gate is widened (but still bounded) so a genuine close-range fix
# during the approach can still land.
GOAL_REFINE_MAX_DELTA_APPROX_M: float = 4.0

# CONFIRM-THEN-REPORT (gen_eval rooms seed-6 long-range mis-report). A learned
# detector's world-xy error grows with range: the confirmer's bearing gate admits
# a detection up to ~22 deg off, whose world error scales as range*sin(22 deg)
# (~0.375*range), so a sighting first made at 6 m can be ~2 m off even though it
# passed the gate. Reporting that raw long-range estimate strands the owner
# outside pickup range (the approach never re-frames the true object). So in
# groundnet mode a FIRST sighting beyond the detector's reliable-report range
# (:meth:`RobotActions.confirm_report_range_m`, ``None`` in oracle mode -> the
# discipline is off and behaviour is byte-identical) is NOT reported yet: the
# searcher/owner walks toward the sighting, re-confirms at close range, THEN
# reports/commits the refined estimate. Bounded to one approach leg — reaching a
# standoff without a close confirm, or losing sight of the object for
# ``CONFIRM_MAX_NO_SIGHT_STEPS``, falls back to the long-range estimate flagged
# ``approx`` (which the owner then refines against with the wider gate above).
CONFIRM_STANDOFF_M: float = 3.0          # plan this far SHORT of the estimate
CONFIRM_MAX_NO_SIGHT_STEPS: int = 40     # lost-sight grace before approx fallback

# Default region labels handed to searchers, partitioning the hall. These MUST
# be regions that :func:`code.fleet.search.region_bounds` understands, otherwise
# a default-constructed protocol crashes the moment it delegates a search.
DEFAULT_REGIONS: Tuple[str, ...] = ("north", "middle", "south")


# ---------------------------------------------------------------------------
# States
# ---------------------------------------------------------------------------
class RobotState(enum.Enum):
    """The coordination state of one robot."""

    IDLE = "IDLE"
    OWNER_QUERYING = "OWNER_QUERYING"        # queried a peer, awaiting its reply
    OWNER_DELEGATING = "OWNER_DELEGATING"    # commanded searchers, awaiting a find
    OWNER_NAVIGATING = "OWNER_NAVIGATING"    # walking to the located object
    OWNER_DELIVERING = "OWNER_DELIVERING"    # carrying the object to the destination
    ASSIST_SEARCHING = "ASSIST_SEARCHING"    # searching a region for a commander


# ---------------------------------------------------------------------------
# RobotActions — the world interface the fleet layer implements
# ---------------------------------------------------------------------------
class RobotActions(abc.ABC):
    """Narrow interface between the protocol and the physical/simulated robot.

    The protocol never touches MuJoCo, the planner, or perception directly — it
    only calls these methods. The Phase-4 fleet layer supplies a concrete
    implementation backed by A* navigation, the heatmap detector and mock pickup;
    unit tests supply scripted fakes.

    Navigation (:meth:`goto`, :meth:`deliver`) is asynchronous: it kicks off
    motion and returns immediately; the protocol polls :meth:`arrived` on later
    steps to learn when the most recently issued target has been reached, or
    :meth:`failed` to learn that it can no longer make progress (the robot fell
    or the goal is unreachable) so a stuck task fails fast instead of burning the
    whole step budget.
    """

    @abc.abstractmethod
    def can_see(self, query: ObjectQuery) -> Optional[XY]:
        """Return the object's world ``(x, y)`` if currently visible, else ``None``."""

    def reconfirm_target(self, query: ObjectQuery) -> Optional[XY]:
        """Force a fresh close-range perception fix on the target, or ``None``.

        Called before a pickup re-approach so the retry heads for the object's
        current best estimate rather than a stale reported point (D-14). The base
        implementation returns ``None`` (no fresh fix — the caller keeps its
        current goal), which is exactly the behaviour with a perfect oracle; the
        fleet bridge overrides it to re-run the learned detector in groundnet
        mode only, leaving the oracle path byte-identical."""
        return None

    def confirm_report_range_m(self) -> Optional[float]:
        """Range beyond which a FIRST sighting must be close-confirmed (CONFIRM-THEN-REPORT).

        When a searcher (or an owner on its own first sighting) sees the object
        farther than this, the raw detector estimate is unreliable enough that
        reporting/committing it immediately can strand the fetcher outside pickup
        range; the protocol instead walks toward the sighting and re-confirms at
        close range before reporting. The base implementation returns ``None`` —
        the discipline is disabled and a sighting is reported the instant it is
        seen, which is exactly (and byte-identically) the perfect-oracle path. The
        fleet bridge overrides it to return the detector's measured reliable range
        only in groundnet mode."""
        return None

    def report_origin(self) -> Tuple[XY, str]:
        """Return the reporter's own ``(x, y)`` pose and its region/room name.

        Used to build the F3 relative-position report payload: each robot knows
        its own pose exactly, so it reports the object's offset relative to
        itself plus the room it is standing in. The base implementation returns
        ``((0.0, 0.0), "the area")`` so scripted test fakes keep working; the
        fleet bridge overrides it with the robot's true pose and current room.
        """
        return ((0.0, 0.0), "the area")

    @abc.abstractmethod
    def goto(self, xy: XY) -> None:
        """Begin navigating to a world position (asynchronous)."""

    @abc.abstractmethod
    def arrived(self) -> bool:
        """Return whether the most recently issued nav target has been reached."""

    def failed(self) -> bool:
        """Return whether the robot can no longer make navigation progress.

        True when the robot has fallen or its most recent goto/deliver goal is
        unreachable (and cannot be re-planned). The base implementation returns
        ``False`` so scripted test fakes and any always-succeeding backend stay
        valid without overriding it.
        """
        return False

    def failure_reason(self) -> str:
        """Return a short human reason for the current :meth:`failed` state."""
        return "navigation failed"

    @abc.abstractmethod
    def start_search(self, query: ObjectQuery, region: str) -> bool:
        """Begin searching a named region for the object (asynchronous).

        Returns:
            True if a coverable patrol was started; False if the region has no
            reachable patrol (the caller should decline the assignment).
        """

    @abc.abstractmethod
    def abort_search(self) -> None:
        """Stop any in-progress search immediately."""

    @abc.abstractmethod
    def pickup(self, query: ObjectQuery) -> bool:
        """Pick up the object (mock grasp; robot must be within reach).

        Returns:
            True if the object is now held; False if the grasp missed (out of
            reach) so the caller can re-approach or fail the task.
        """

    @abc.abstractmethod
    def deliver(self, destination_xy: XY) -> None:
        """Begin navigating to a destination while carrying the object."""


# ---------------------------------------------------------------------------
# Internal bookkeeping
# ---------------------------------------------------------------------------
@dataclasses.dataclass
class _AssistCtx:
    """State of an accepted search assignment (helper role)."""

    commander: str
    query: ObjectQuery
    region: str
    started_t: int
    # CONFIRM-THEN-REPORT approach sub-state (groundnet long-range first sighting).
    # While ``confirming`` the searcher is walking toward a standoff to re-confirm
    # the sighting at close range before it sends REPORT_FOUND; ``best_xy`` holds
    # the freshest/closest estimate seen so far (the approx fallback), and the two
    # timestamps bound the single approach leg.
    confirming: bool = False
    best_xy: Optional[XY] = None
    approach_started_t: int = 0
    last_seen_t: int = 0


# ---------------------------------------------------------------------------
# RobotProtocol
# ---------------------------------------------------------------------------
class RobotProtocol:
    """The coordination state machine for a single robot.

    Args:
        callsign: This robot's name (e.g. ``"Alpha"``).
        bus: The shared message bus.
        actions: The world interface implementation.
        peers: Callsigns of the other robots this robot may query / command.
        search_regions: Region labels to partition a delegated search across
            peers (one region per commanded peer, in order).
        reply_deadline_steps: Steps to wait for a peer's ``REPORT_VISIBILITY``
            before treating it as "not visible" and moving on.
        search_deadline_steps: Steps to wait for a ``REPORT_FOUND`` before
            declaring the task failed (search exhausted).
    """

    def __init__(self, callsign: str, bus: MessageBus, actions: RobotActions,
                 peers: Sequence[str], *,
                 search_regions: Sequence[str] = DEFAULT_REGIONS,
                 reply_deadline_steps: int = 50,
                 search_deadline_steps: int = 2000,
                 region_assigner: Optional[RegionAssigner] = None) -> None:
        self.callsign = callsign
        self._bus = bus
        self._actions = actions
        self._peers: Tuple[str, ...] = tuple(peers)
        self._regions: Tuple[str, ...] = tuple(search_regions)
        self._reply_deadline = int(reply_deadline_steps)
        self._search_deadline_steps = int(search_deadline_steps)
        self._region_assigner = region_assigner

        self._state = RobotState.IDLE
        self.last_result: Optional[str] = None  # "complete" | "failed" | None

        # Owner-role bookkeeping (valid only while owning a task).
        self._task: Optional[TaskSpec] = None
        self._query_queue: List[str] = []
        self._awaiting_peer: Optional[str] = None
        self._query_deadline = 0
        self._pending_accept: Set[str] = set()
        self._searchers: Set[str] = set()
        self._reserve: List[str] = []
        self._peer_region: Dict[str, str] = {}
        self._search_deadline = 0
        self._nav_issued_t = 0
        self._nav_target: Optional[XY] = None  # object location for pickup re-approach
        self._nav_target_approx = False  # goal came from an approx (long-range) report -> wider refine gate
        self._pickup_retries = 0
        self._last_refine_t = 0  # last close-range goal refinement (D-14)
        # Owner own-sighting CONFIRM-THEN-REPORT approach: while ``_own_confirm``
        # the owner is walking to a standoff to re-confirm its OWN long-range
        # first sighting before it commits the fetch goal (state stays
        # OWNER_NAVIGATING — "fetching" — but no located target is published yet).
        self._own_confirm = False
        self._own_confirm_best: Optional[XY] = None
        self._own_confirm_started_t = 0
        self._own_confirm_last_seen_t = 0

        # Helper-role bookkeeping.
        self._assist: Optional[_AssistCtx] = None

    # -- public API --------------------------------------------------------
    @property
    def state(self) -> RobotState:
        """The robot's current coordination state."""
        return self._state

    def is_idle(self) -> bool:
        """Return whether the robot is free to accept new work."""
        return self._state is RobotState.IDLE

    @property
    def located_target(self) -> Optional[XY]:
        """The object position this owner has located, or ``None`` (F2).

        Set once the owner begins navigating to a located object — whether from
        its own first sighting, a peer's visibility report or a searcher's find —
        and held through delivery; ``None`` while still querying / delegating or
        when idle. Lets the mission layer defer the target ring until the object
        is actually known and draw it at the reported position.
        """
        if self._state in (RobotState.OWNER_NAVIGATING,
                            RobotState.OWNER_DELIVERING):
            return self._nav_target
        return None

    def refine_nav_goal(self, new_xy: XY, t: int, *, robot_xy: XY) -> bool:
        """Steer the in-flight fetch goal onto a fresher close-range estimate (D-14).

        Only acts while this owner is walking to a found object (``OWNER_NAVIGATING``
        — never mid-delivery), and only for a bounded, plausible *same-object*
        refinement (see the ``GOAL_REFINE_*`` constants): rate-limited to once per
        ``GOAL_REFINE_INTERVAL`` steps, the new estimate must move the goal by more
        than ``GOAL_REFINE_MIN_DELTA_M`` but no farther than ``GOAL_REFINE_MAX_DELTA_M``,
        and the owner must still be farther than the pickup radius
        (``GOAL_REFINE_MIN_RANGE_M``) from the current goal. When accepted it updates
        the goal and re-plans (a fresh :meth:`RobotActions.goto`).

        Args:
            new_xy: The detector's fresh world ``(x, y)`` estimate of the object.
            t: The current simulation step (for the rate limit).
            robot_xy: The owner's current pelvis ``(x, y)`` (for the range gate).

        Returns:
            True iff the goal was refined and a fresh navigation issued.
        """
        if self._state is not RobotState.OWNER_NAVIGATING or self._nav_target is None:
            return False
        if t - self._last_refine_t < GOAL_REFINE_INTERVAL:
            return False
        gx, gy = self._nav_target
        nx, ny = float(new_xy[0]), float(new_xy[1])
        delta = math.hypot(nx - gx, ny - gy)
        max_delta = (GOAL_REFINE_MAX_DELTA_APPROX_M if self._nav_target_approx
                     else GOAL_REFINE_MAX_DELTA_M)
        if not (GOAL_REFINE_MIN_DELTA_M < delta <= max_delta):
            return False
        if math.hypot(gx - float(robot_xy[0]), gy - float(robot_xy[1])) <= GOAL_REFINE_MIN_RANGE_M:
            return False  # already within pickup range of the goal — don't nudge
        self._nav_target = (nx, ny)
        self._last_refine_t = t
        self._nav_issued_t = t
        self._actions.goto(self._nav_target)
        return True

    def step(self, t_step: int) -> None:
        """Advance the state machine by one simulation step.

        Drains the inbox and reacts to every queued message, then advances any
        deadline- or perception-driven state (query timeouts, search polling,
        navigation arrival).

        Args:
            t_step: The current simulation step.
        """
        for msg in self._bus.drain(self.callsign):
            self._handle(msg, t_step)
        self._tick_state(t_step)

    # -- message dispatch --------------------------------------------------
    def _handle(self, msg: Message, t: int) -> None:
        perf = msg.performative
        if perf is Performative.REQUEST_TASK:
            self._on_request_task(msg, t)
        elif perf is Performative.QUERY_VISIBILITY:
            self._on_query_visibility(msg, t)
        elif perf is Performative.REPORT_VISIBILITY:
            self._on_report_visibility(msg, t)
        elif perf is Performative.COMMAND_SEARCH:
            self._on_command_search(msg, t)
        elif perf is Performative.ACCEPT:
            self._on_accept(msg, t)
        elif perf is Performative.REJECT:
            self._on_reject(msg, t)
        elif perf is Performative.REPORT_FOUND:
            self._on_report_found(msg, t)
        # STATUS_UPDATE / TASK_COMPLETE / TASK_FAILED / FLEET_REQUEST are not
        # acted on by a robot protocol (they terminate at user/allocator).

    # -- owner role: receiving a task -------------------------------------
    def _on_request_task(self, msg: Message, t: int) -> None:
        if not self.is_idle():
            return  # busy: silently decline (allocator picks an idle robot)
        self._task = msg.payload["task"]
        self.last_result = None
        loc = self._actions.can_see(self._task.query)
        if loc is not None:
            self._begin_owner_leg(loc, t)
        else:
            self._query_queue = list(self._peers)
            self._start_next_query(t)

    def _begin_owner_leg(self, loc: XY, t: int) -> None:
        """Start the owner's fetch from its OWN first sighting (CONFIRM-THEN-REPORT).

        A close (or oracle-mode) sighting is committed immediately as the fetch
        goal; a groundnet long-range sighting is instead confirmed by walking to a
        standoff first (:meth:`_enter_owner_confirm`), so an unreliable long-range
        estimate never becomes the committed goal.
        """
        reliable = self._actions.confirm_report_range_m()
        if reliable is not None and self._sighting_range(loc) > reliable:
            self._enter_owner_confirm(loc, t)
        else:
            self._begin_navigation(loc, t)

    def _start_next_query(self, t: int) -> None:
        """Query the next peer, or delegate a search if all peers are exhausted."""
        if not self._query_queue:
            self._begin_delegation(t)
            return
        peer = self._query_queue.pop(0)
        self._awaiting_peer = peer
        self._query_deadline = t + self._reply_deadline
        self._state = RobotState.OWNER_QUERYING
        self._bus.post(self.callsign, peer, Performative.QUERY_VISIBILITY,
                       {"query": self._task.query})

    def _on_report_visibility(self, msg: Message, t: int) -> None:
        if self._state is not RobotState.OWNER_QUERYING:
            return
        if msg.sender != self._awaiting_peer:
            return  # stale / out-of-turn reply
        if msg.payload.get("visible"):
            self._query_queue = []
            self._awaiting_peer = None
            # F3: reconstruct the absolute object position from the peer's
            # relative report (reporter_pose + rel_offset).
            self._begin_navigation(reconstruct_location(msg.payload), t)
        else:
            self._start_next_query(t)

    # -- owner role: delegating a search ----------------------------------
    def _begin_delegation(self, t: int) -> None:
        self._state = RobotState.OWNER_DELEGATING
        self._pending_accept = set()
        self._searchers = set()
        self._peer_region = {}
        self._reserve = list(self._peers)
        self._search_deadline = t + self._search_deadline_steps
        for peer, region in self._plan_assignment(self._reserve, self._regions):
            self._reserve.remove(peer)
            self._command_search(peer, region)
        if not self._pending_accept:
            self._fail_owner("no robots available to search", t)

    def _plan_assignment(self, peers: Sequence[str],
                         regions: Sequence[str]) -> List[Tuple[str, str]]:
        """Return the ``(peer, region)`` search assignments for delegation.

        With the default (``region_assigner is None``) each region is handed to
        the next reserve peer in order — the historical behaviour (more regions
        than peers leaves the surplus uncovered). The fleet layer injects an
        A*-nearest-room assigner so, on the multi-room layout, each searcher
        takes the nearest unsearched room.
        """
        if self._region_assigner is not None:
            return list(self._region_assigner(peers, regions))
        return [(peer, region) for peer, region in zip(peers, regions)]

    def _command_search(self, peer: str, region: str) -> None:
        self._peer_region[peer] = region
        self._pending_accept.add(peer)
        self._bus.post(self.callsign, peer, Performative.COMMAND_SEARCH,
                       {"query": self._task.query, "region": region, "cancel": False})

    def _command_cancel(self, peer: str) -> None:
        self._bus.post(self.callsign, peer, Performative.COMMAND_SEARCH,
                       {"query": self._task.query,
                        "region": self._peer_region.get(peer, ""), "cancel": True})

    def _on_accept(self, msg: Message, t: int) -> None:
        if self._state is not RobotState.OWNER_DELEGATING:
            return
        if msg.sender in self._pending_accept:
            self._pending_accept.discard(msg.sender)
            self._searchers.add(msg.sender)

    def _on_reject(self, msg: Message, t: int) -> None:
        """Handle a peer declining/abandoning a search command.

        Fires both for a peer that rejects the initial ``COMMAND_SEARCH`` (busy
        or an uncoverable region) and for a searcher that had already ACCEPTed
        but can no longer continue (e.g. it fell). In either case the peer is
        dropped and its region is re-planned onto a reserve peer if one is free.
        """
        if self._state is not RobotState.OWNER_DELEGATING:
            return
        if msg.sender not in self._pending_accept and msg.sender not in self._searchers:
            return  # stale / not one of ours
        self._pending_accept.discard(msg.sender)
        self._searchers.discard(msg.sender)
        region = self._peer_region.get(msg.sender)
        # Re-plan: hand the rejected peer's region to a reserve peer, if any.
        if region is not None and self._reserve:
            self._command_search(self._reserve.pop(0), region)
        if not self._pending_accept and not self._searchers:
            self._fail_owner("all peers busy; nobody can search", t)

    def _on_report_found(self, msg: Message, t: int) -> None:
        if self._state is not RobotState.OWNER_DELEGATING:
            return
        reporter = msg.sender
        self._cancel_active_searchers(exclude=reporter)
        # F3: reconstruct the absolute object position from the searcher's
        # relative report (reporter_pose + rel_offset). A report the searcher
        # flagged ``approx`` (it could not close-confirm) gets the wider
        # refinement gate so the owner's approach can correct it.
        approx = bool(msg.payload.get("approx", False))
        self._begin_navigation(reconstruct_location(msg.payload), t, approx=approx)

    def _cancel_active_searchers(self, exclude: Optional[str] = None) -> None:
        """Call off every active/pending searcher (in fixed peer order for determinism)."""
        active = self._searchers | self._pending_accept
        for peer in self._peers:  # deterministic order, unlike set iteration
            if peer in active and peer != exclude:
                self._command_cancel(peer)

    # -- owner role: fetch & deliver --------------------------------------
    def _begin_navigation(self, location: XY, t: int, *,
                          approx: bool = False) -> None:
        self._state = RobotState.OWNER_NAVIGATING
        self._own_confirm = False  # committing to a real goal ends any confirm leg
        self._own_confirm_best = None
        self._nav_issued_t = t
        self._nav_target = (float(location[0]), float(location[1]))
        self._nav_target_approx = bool(approx)
        self._pickup_retries = 0
        self._last_refine_t = t
        self._actions.goto(location)
        self._notify_user(
            Performative.STATUS_UPDATE,
            text=(f"Found the {self._task.query.describe()} at "
                  f"({location[0]:.1f}, {location[1]:.1f}); heading over to fetch it."))

    def _advance_navigating(self, t: int) -> None:
        if self._own_confirm:
            self._advance_owner_confirm(t)
            return
        if self._actions.failed():  # fell / unreachable object -> fail fast
            self._fail_owner(self._actions.failure_reason(), t)
            return
        if not (t > self._nav_issued_t and self._actions.arrived()):
            return
        if self._actions.pickup(self._task.query):
            self._begin_delivering(t)
        elif self._pickup_retries < MAX_PICKUP_RETRIES:
            # Grasp missed: re-approach the object once before giving up. Never
            # transition to delivery (let alone completion) without a real hold.
            # D-14: force a fresh close-range perception fix first, so the retry
            # heads for the object's current best estimate rather than the stale
            # reported point (returns None -> unchanged goal in oracle mode).
            self._pickup_retries += 1
            fresh = self._actions.reconfirm_target(self._task.query)
            if fresh is not None and self._nav_target is not None:
                fx, fy = float(fresh[0]), float(fresh[1])
                max_delta = (GOAL_REFINE_MAX_DELTA_APPROX_M if self._nav_target_approx
                             else GOAL_REFINE_MAX_DELTA_M)
                if math.hypot(fx - self._nav_target[0],
                              fy - self._nav_target[1]) <= max_delta:
                    self._nav_target = (fx, fy)
            self._nav_issued_t = t
            self._actions.goto(self._nav_target)
        else:
            self._fail_owner("could not pick up the object", t)

    # -- owner role: confirm-then-commit own long-range sighting -----------
    def _enter_owner_confirm(self, loc: XY, t: int) -> None:
        """Walk to a standoff to re-confirm the owner's OWN long-range sighting.

        The owner enters ``OWNER_NAVIGATING`` (it IS heading toward the object)
        but with ``_own_confirm`` set and no committed ``_nav_target`` yet, so no
        "found it" milestone is announced and no target ring is published until
        the close-range confirm lands. It plans to a standoff ~``CONFIRM_STANDOFF_M``
        short of the estimate (never onto the unreliable point).
        """
        self._state = RobotState.OWNER_NAVIGATING
        self._own_confirm = True
        self._own_confirm_best = (float(loc[0]), float(loc[1]))
        self._own_confirm_started_t = t
        self._own_confirm_last_seen_t = t
        self._nav_target = None
        self._nav_target_approx = False
        self._nav_issued_t = t
        self._actions.goto(self._standoff_toward(loc, CONFIRM_STANDOFF_M))

    def _advance_owner_confirm(self, t: int) -> None:
        """Advance the owner's confirm approach; commit the goal once resolved."""
        if self._actions.failed():
            # Can't complete the approach -> commit to the best estimate we have,
            # flagged approx so the fetch approach refines it with the wider gate.
            self._commit_owner_confirm(self._own_confirm_best, t, approx=True)
            return
        loc = self._actions.can_see(self._task.query)
        if loc is not None:
            self._own_confirm_best = (float(loc[0]), float(loc[1]))
            self._own_confirm_last_seen_t = t
            reliable = self._actions.confirm_report_range_m()
            if reliable is None or self._sighting_range(loc) <= reliable:
                # Close-range confirm: commit the refined estimate and fetch it.
                self._commit_owner_confirm(loc, t, approx=False)
                return
        if t > self._own_confirm_started_t and self._actions.arrived():
            self._commit_owner_confirm(self._own_confirm_best, t, approx=True)
            return
        if t - self._own_confirm_last_seen_t >= CONFIRM_MAX_NO_SIGHT_STEPS:
            self._commit_owner_confirm(self._own_confirm_best, t, approx=True)

    def _commit_owner_confirm(self, loc: Optional[XY], t: int, *,
                              approx: bool) -> None:
        """Finish the owner's confirm leg by committing ``loc`` as the fetch goal."""
        target = loc if loc is not None else self._nav_target
        self._own_confirm = False
        if target is None:  # nothing ever sighted (defensive) -> fail the fetch
            self._fail_owner("lost sight of the object before confirming", t)
            return
        self._begin_navigation(target, t, approx=approx)

    def _begin_delivering(self, t: int) -> None:
        """Report the pickup and start carrying the object to the destination."""
        self._notify_user(
            Performative.STATUS_UPDATE,
            text=(f"Picked up the {self._task.query.describe()}; "
                  f"delivering to the {self._task.destination_name}."))
        self._state = RobotState.OWNER_DELIVERING
        self._nav_issued_t = t
        self._actions.deliver(self._task.destination_xy)

    def _advance_delivering(self, t: int) -> None:
        if self._actions.failed():  # fell mid-carry (object dropped) -> fail fast
            self._fail_owner(self._actions.failure_reason(), t)
            return
        if t > self._nav_issued_t and self._actions.arrived():
            self._notify_user(
                Performative.TASK_COMPLETE,
                text=(f"Delivered the {self._task.query.describe()} to the "
                      f"{self._task.destination_name}."))
            self._finish_owner("complete")

    def _fail_owner(self, reason: str, t: int) -> None:
        self._cancel_active_searchers()
        self._notify_user(Performative.TASK_FAILED, reason=reason)
        self._finish_owner("failed")

    def _finish_owner(self, result: str) -> None:
        self.last_result = result
        self._task = None
        self._query_queue = []
        self._awaiting_peer = None
        self._pending_accept = set()
        self._searchers = set()
        self._reserve = []
        self._peer_region = {}
        self._nav_target = None
        self._nav_target_approx = False
        self._own_confirm = False
        self._own_confirm_best = None
        self._pickup_retries = 0
        self._state = RobotState.IDLE

    def _notify_user(self, performative: Performative, **payload: object) -> None:
        """Send a milestone/outcome to the task requester (owner-only path).

        Structurally enforces need-to-know: only a protocol that currently holds
        a task can reach this method, and it always addresses that task's
        requester — helpers (no task) can never message the user.
        """
        assert self._task is not None, "only a task owner may report to the requester"
        self._bus.post(self.callsign, self._task.requester, performative, payload)

    # -- helper role: answering queries & searching -----------------------
    def _on_query_visibility(self, msg: Message, t: int) -> None:
        query = msg.payload["query"]
        loc = self._actions.can_see(query)
        if loc is not None:
            # F3: report the object's position relative to my own known pose.
            payload = self._relative_report({"query": query, "visible": True}, loc)
        else:
            payload = {"query": query, "visible": False}
        self._bus.post(self.callsign, msg.sender, Performative.REPORT_VISIBILITY,
                       payload, in_reply_to=msg.msg_id)

    def _relative_report(self, base: dict, obj_xy: XY) -> dict:
        """Build an F3 relative-position report payload for ``obj_xy``.

        Asks the world bridge for my own exact pose + current room and encodes
        the object as an offset from me (the receiver reconstructs the absolute
        position with :func:`code.comms.messages.reconstruct_location`).
        """
        (rx, ry), room = self._actions.report_origin()
        return relative_report_payload((rx, ry), room, obj_xy, extra=base)

    def _on_command_search(self, msg: Message, t: int) -> None:
        if msg.payload.get("cancel"):
            if (self._state is RobotState.ASSIST_SEARCHING and self._assist
                    and msg.sender == self._assist.commander):
                self._actions.abort_search()
                self._end_assist()
            return
        if self.is_idle() and self._enter_assist(
                msg.sender, msg.payload["query"], msg.payload["region"], t):
            self._bus.post(self.callsign, msg.sender, Performative.ACCEPT,
                           {"region": msg.payload["region"]}, in_reply_to=msg.msg_id)
        else:
            reason = "busy" if not self.is_idle() else "region not coverable"
            self._bus.post(self.callsign, msg.sender, Performative.REJECT,
                           {"reason": reason}, in_reply_to=msg.msg_id)

    def _enter_assist(self, commander: str, query: ObjectQuery, region: str,
                      t: int) -> bool:
        """Enter the searching state for a commander (also a fleet/test entry point).

        Returns:
            True if a coverable patrol was started (state -> ``ASSIST_SEARCHING``);
            False if the region has no reachable patrol, in which case the robot
            stays idle so the caller can REJECT the command.
        """
        if not self._actions.start_search(query, region):
            return False
        self._assist = _AssistCtx(commander, query, region, started_t=t)
        self._state = RobotState.ASSIST_SEARCHING
        return True

    def _advance_searching(self, t: int) -> None:
        assist = self._assist
        if assist is None:
            return
        if assist.confirming:
            self._advance_searcher_confirm(t)
            return
        if self._actions.failed():
            # We fell mid-search: tell the commander so it can reassign the region
            # (or fail gracefully). A REJECT is the schema-legal peer->owner signal
            # for "I can no longer cover this"; the owner's _on_reject re-plans it.
            self._actions.abort_search()
            self._bus.post(self.callsign, assist.commander, Performative.REJECT,
                           {"reason": "searcher fell"})
            self._end_assist()
            return
        if t <= assist.started_t:
            return  # give the search at least one step before the first look
        loc = self._actions.can_see(assist.query)
        if loc is not None:
            reliable = self._actions.confirm_report_range_m()
            if reliable is not None and self._sighting_range(loc) > reliable:
                # CONFIRM-THEN-REPORT: a groundnet long-range first sighting is
                # unreliable — walk toward it and re-confirm before reporting.
                self._enter_searcher_confirm(assist, loc, t)
                return
            # Close (or oracle-mode) sighting -> report the find now. Need-to-know:
            # the find goes to the commander ONLY, then we stop.
            self._post_report_found(assist.commander, assist.query, loc,
                                    approx=False)
            self._actions.abort_search()
            self._end_assist()

    def _enter_searcher_confirm(self, assist: _AssistCtx, loc: XY, t: int) -> None:
        """Begin the searcher's one approach leg to close-confirm a long-range find.

        Stays in ``ASSIST_SEARCHING`` (a flag on the assist context, not a new
        performative — the transcript stays sensible, the find just arrives a few
        seconds later). Stops the region patrol and plans to a standoff
        ~``CONFIRM_STANDOFF_M`` short of the estimate.
        """
        assist.confirming = True
        assist.best_xy = (float(loc[0]), float(loc[1]))
        assist.approach_started_t = t
        assist.last_seen_t = t
        self._actions.abort_search()
        self._actions.goto(self._standoff_toward(loc, CONFIRM_STANDOFF_M))

    def _advance_searcher_confirm(self, t: int) -> None:
        """Advance the searcher's confirm approach; report once resolved."""
        assist = self._assist
        if assist is None:
            return
        if self._actions.failed():
            # Can't complete the approach -> hand off the best long-range estimate
            # flagged approx (we DID sight it; a wider owner gate recovers it).
            self._finish_searcher_confirm(assist, assist.best_xy, approx=True)
            return
        loc = self._actions.can_see(assist.query)
        if loc is not None:
            assist.best_xy = (float(loc[0]), float(loc[1]))
            assist.last_seen_t = t
            reliable = self._actions.confirm_report_range_m()
            if reliable is None or self._sighting_range(loc) <= reliable:
                self._finish_searcher_confirm(assist, loc, approx=False)
                return
        if t > assist.approach_started_t and self._actions.arrived():
            # Reached the standoff without a close confirm -> report approx.
            self._finish_searcher_confirm(assist, assist.best_xy, approx=True)
            return
        if t - assist.last_seen_t >= CONFIRM_MAX_NO_SIGHT_STEPS:
            # Lost sight of the object during the approach -> report approx.
            self._finish_searcher_confirm(assist, assist.best_xy, approx=True)

    def _finish_searcher_confirm(self, assist: _AssistCtx, loc: Optional[XY],
                                 approx: bool) -> None:
        """Send the (possibly approx) REPORT_FOUND and end the assist."""
        target = loc if loc is not None else assist.best_xy
        if target is not None:
            self._post_report_found(assist.commander, assist.query, target,
                                    approx=approx)
        self._actions.abort_search()
        self._end_assist()

    def _post_report_found(self, commander: str, query: ObjectQuery, loc: XY,
                           *, approx: bool) -> None:
        """Post a REPORT_FOUND to the commander (adds ``approx`` only when true).

        A confirmed / oracle-mode report carries the exact historical payload (no
        ``approx`` key) so those messages stay byte-identical; only a long-range
        fallback tags ``approx=True`` for the owner's wider refinement gate.
        """
        base: Dict[str, object] = {"object": query}
        if approx:
            base["approx"] = True
        self._bus.post(self.callsign, commander, Performative.REPORT_FOUND,
                       self._relative_report(base, loc))

    def _sighting_range(self, loc: XY) -> float:
        """Straight-line range (m) from my own known pose to a sighting estimate."""
        (rx, ry), _room = self._actions.report_origin()
        return math.hypot(float(loc[0]) - rx, float(loc[1]) - ry)

    def _standoff_toward(self, loc: XY, standoff_m: float) -> XY:
        """A point ``standoff_m`` short of ``loc`` along my heading toward it.

        Keeps the approach pointed at the sighting (so it stays framed) while
        never planning onto the unreliable long-range estimate itself.
        """
        (rx, ry), _room = self._actions.report_origin()
        dx, dy = float(loc[0]) - rx, float(loc[1]) - ry
        d = math.hypot(dx, dy)
        if d <= standoff_m:
            return (rx, ry)
        k = (d - standoff_m) / d
        return (rx + dx * k, ry + dy * k)

    def _end_assist(self) -> None:
        self._assist = None
        self._state = RobotState.IDLE

    # -- per-step time/perception advance ---------------------------------
    def _tick_state(self, t: int) -> None:
        state = self._state
        if state is RobotState.OWNER_QUERYING:
            if t >= self._query_deadline:  # peer never answered -> not visible
                self._start_next_query(t)
        elif state is RobotState.OWNER_DELEGATING:
            if t >= self._search_deadline:
                self._fail_owner("search exhausted", t)
            elif not self._pending_accept and not self._searchers:
                self._fail_owner("all peers busy; nobody can search", t)
        elif state is RobotState.OWNER_NAVIGATING:
            self._advance_navigating(t)
        elif state is RobotState.OWNER_DELIVERING:
            self._advance_delivering(t)
        elif state is RobotState.ASSIST_SEARCHING:
            self._advance_searching(t)
