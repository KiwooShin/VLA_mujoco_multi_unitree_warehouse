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
from typing import Dict, List, Optional, Sequence, Set, Tuple

from code.comms.bus import MessageBus
from code.comms.messages import Message, ObjectQuery, Performative, TaskSpec

XY = Tuple[float, float]

# How many times a missed mock pickup is re-attempted (re-approach) before the
# owner declares the fetch failed. One retry is enough for a transient miss.
MAX_PICKUP_RETRIES: int = 1

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
                 search_deadline_steps: int = 2000) -> None:
        self.callsign = callsign
        self._bus = bus
        self._actions = actions
        self._peers: Tuple[str, ...] = tuple(peers)
        self._regions: Tuple[str, ...] = tuple(search_regions)
        self._reply_deadline = int(reply_deadline_steps)
        self._search_deadline_steps = int(search_deadline_steps)

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
        self._pickup_retries = 0

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
            self._begin_navigation(loc, t)
        else:
            self._query_queue = list(self._peers)
            self._start_next_query(t)

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
            self._begin_navigation(msg.payload["location"], t)
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
        for region in self._regions:
            if not self._reserve:
                break  # more regions than peers: leave the rest uncovered
            self._command_search(self._reserve.pop(0), region)
        if not self._pending_accept:
            self._fail_owner("no robots available to search", t)

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
        self._begin_navigation(msg.payload["location"], t)

    def _cancel_active_searchers(self, exclude: Optional[str] = None) -> None:
        """Call off every active/pending searcher (in fixed peer order for determinism)."""
        active = self._searchers | self._pending_accept
        for peer in self._peers:  # deterministic order, unlike set iteration
            if peer in active and peer != exclude:
                self._command_cancel(peer)

    # -- owner role: fetch & deliver --------------------------------------
    def _begin_navigation(self, location: XY, t: int) -> None:
        self._state = RobotState.OWNER_NAVIGATING
        self._nav_issued_t = t
        self._nav_target = (float(location[0]), float(location[1]))
        self._pickup_retries = 0
        self._actions.goto(location)
        self._notify_user(
            Performative.STATUS_UPDATE,
            text=(f"Found the {self._task.query.describe()} at "
                  f"({location[0]:.1f}, {location[1]:.1f}); heading over to fetch it."))

    def _advance_navigating(self, t: int) -> None:
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
            self._pickup_retries += 1
            self._nav_issued_t = t
            self._actions.goto(self._nav_target)
        else:
            self._fail_owner("could not pick up the object", t)

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
        payload = {"query": query, "visible": loc is not None}
        if loc is not None:
            payload["location"] = loc
        self._bus.post(self.callsign, msg.sender, Performative.REPORT_VISIBILITY,
                       payload, in_reply_to=msg.msg_id)

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
            # Need-to-know: the find goes to the commander ONLY, then we stop.
            self._bus.post(self.callsign, assist.commander, Performative.REPORT_FOUND,
                           {"object": assist.query, "location": loc})
            self._actions.abort_search()
            self._end_assist()

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
