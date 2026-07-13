"""messages.py — Typed, addressed message vocabulary for the fleet comms layer.

Pure-Python, no MuJoCo/GPU: every type here is a frozen dataclass or enum so
messages are hashable-by-value, deterministic and trivially unit-testable
(docs/multi_plan.md sec 1). The message set encodes the canonical warehouse
protocol — a user asks a robot to fetch an object; the robot checks its own
view, queries peers, delegates a search, receives the location, then fetches
and delivers while reporting milestones back to the *requester only*
(need-to-know sharing).

Public API
----------
Performative         — enum of speech-act types carried by every Message.
TaskKind             — enum of task verbs (FETCH for now, extensible).
ObjectQuery          — a (colour, shape) referent with matching/description.
TaskSpec             — a fully specified task (kind + object + destination).
Message              — one addressed, validated message (bus-assigned id/step).
"""

from __future__ import annotations

import dataclasses
import enum
from types import MappingProxyType
from typing import Any, Mapping, Optional, Tuple


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------
class Performative(enum.Enum):
    """Speech-act type of a :class:`Message` (FIPA-style performatives)."""

    REQUEST_TASK = "REQUEST_TASK"          # user/allocator -> robot: own this task
    QUERY_VISIBILITY = "QUERY_VISIBILITY"  # owner -> peer: can you see O?
    REPORT_VISIBILITY = "REPORT_VISIBILITY"  # peer -> owner: (not) visible [+ xy]
    COMMAND_SEARCH = "COMMAND_SEARCH"      # owner -> peer: search region (or cancel)
    ACCEPT = "ACCEPT"                      # peer -> owner: accept a command
    REJECT = "REJECT"                      # peer -> owner: reject (reason)
    REPORT_FOUND = "REPORT_FOUND"          # searcher -> owner ONLY: object + location
    STATUS_UPDATE = "STATUS_UPDATE"        # owner -> requester: milestone text
    TASK_COMPLETE = "TASK_COMPLETE"        # owner -> requester: task done
    TASK_FAILED = "TASK_FAILED"            # owner -> requester: task failed (reason)
    FLEET_REQUEST = "FLEET_REQUEST"        # user -> fleet: any robot bring O


class TaskKind(enum.Enum):
    """Kind of task a :class:`TaskSpec` describes (extensible)."""

    FETCH = "FETCH"


_TASK_VERB: Mapping[TaskKind, str] = MappingProxyType({TaskKind.FETCH: "fetch"})


# ---------------------------------------------------------------------------
# ObjectQuery
# ---------------------------------------------------------------------------
@dataclasses.dataclass(frozen=True)
class ObjectQuery:
    """A colour/shape referent for an object, either field optionally a wildcard.

    Attributes:
        color_name: Colour word (e.g. ``"red"``) or ``None`` to match any colour.
        shape_name: Shape word (e.g. ``"cube"``) or ``None`` to match any shape.
    """

    color_name: Optional[str] = None
    shape_name: Optional[str] = None

    def matches(self, obj: Mapping[str, Any]) -> bool:
        """Return whether a scene object dict satisfies this query.

        Args:
            obj: Object dict with ``color_name`` / ``shape_name`` keys (the
                baseline scene object schema, ``code.sim.scene``).

        Returns:
            True iff every non-wildcard field equals the object's field.
        """
        if self.color_name is not None and obj.get("color_name") != self.color_name:
            return False
        if self.shape_name is not None and obj.get("shape_name") != self.shape_name:
            return False
        return True

    def describe(self) -> str:
        """Return a short human phrase, e.g. ``"red cube"`` / ``"cube"`` / ``"object"``."""
        shape = self.shape_name or "object"
        return f"{self.color_name} {shape}" if self.color_name else shape

    @property
    def is_generic(self) -> bool:
        """Whether this query is the wildcard "any object" referent (F4).

        A generic query (both fields ``None``) matches every scene object, so a
        fleet-addressed "bring the object to the destination" resolves to the
        first object a robot actually finds.
        """
        return self.color_name is None and self.shape_name is None


# ---------------------------------------------------------------------------
# TaskSpec
# ---------------------------------------------------------------------------
@dataclasses.dataclass(frozen=True)
class TaskSpec:
    """A fully specified fleet task: fetch an object to a destination.

    Attributes:
        kind: Task verb (:class:`TaskKind`); only ``FETCH`` for now.
        query: The object to act on.
        destination_name: Human name of the destination (e.g. ``"delivery pad"``).
        destination_xy: World ``(x, y)`` of the destination in metres.
        requester: Who asked — ``"user"`` or a robot callsign. Milestones and
            completion reports go here and *only* here (need-to-know).
    """

    kind: TaskKind
    query: ObjectQuery
    destination_name: str
    destination_xy: Tuple[float, float]
    requester: str = "user"

    def describe(self) -> str:
        """Return e.g. ``"fetch the red cube to the delivery pad"``."""
        verb = _TASK_VERB.get(self.kind, self.kind.value.lower())
        return f"{verb} the {self.query.describe()} to the {self.destination_name}"


# ---------------------------------------------------------------------------
# Message
# ---------------------------------------------------------------------------
# Required payload keys per performative. Presence (not type) is validated in
# Message.__post_init__ so malformed protocol traffic fails loudly at post time.
_REQUIRED_PAYLOAD: Mapping[Performative, Tuple[str, ...]] = MappingProxyType({
    Performative.REQUEST_TASK: ("task",),
    Performative.QUERY_VISIBILITY: ("query",),
    Performative.REPORT_VISIBILITY: ("query", "visible"),
    Performative.COMMAND_SEARCH: ("query", "region"),
    Performative.ACCEPT: (),
    Performative.REJECT: ("reason",),
    # F3: position reports are relative to the reporter (each robot knows its own
    # pose exactly). The receiver reconstructs the absolute object position with
    # ``reconstruct_location`` (reporter_pose + rel_offset); no absolute is sent.
    Performative.REPORT_FOUND: ("object", "reporter_pose", "rel_offset"),
    Performative.STATUS_UPDATE: ("text",),
    Performative.TASK_COMPLETE: ("text",),
    Performative.TASK_FAILED: ("reason",),
    Performative.FLEET_REQUEST: ("task",),
})


@dataclasses.dataclass(frozen=True)
class Message:
    """One addressed, validated message on the bus.

    ``msg_id`` and ``t_step`` are assigned by :class:`~code.comms.bus.MessageBus`
    at post time (do not set them by hand outside tests). The ``payload`` is
    frozen to a read-only mapping in ``__post_init__`` so a delivered message can
    never be mutated by a recipient.

    Attributes:
        msg_id: Monotonic bus-assigned id (unique per bus).
        t_step: Simulation step at which the message was posted.
        sender: ``"user"`` or a robot callsign.
        recipient: A robot callsign, ``"user"``, or ``"fleet"`` (allocator).
        performative: The speech act (:class:`Performative`).
        payload: Immutable mapping of performative-specific fields.
        in_reply_to: ``msg_id`` this message answers, or ``None``.

    Raises:
        TypeError: If ``performative`` is not a :class:`Performative`.
        ValueError: If a required payload key for the performative is missing.
    """

    msg_id: int
    t_step: int
    sender: str
    recipient: str
    performative: Performative
    payload: Mapping[str, Any] = dataclasses.field(default_factory=dict)
    in_reply_to: Optional[int] = None

    def __post_init__(self) -> None:
        if not isinstance(self.performative, Performative):
            raise TypeError(
                f"performative must be a Performative, got {self.performative!r}"
            )
        if not self.sender:
            raise ValueError("message sender must be a non-empty string")
        if not self.recipient:
            raise ValueError("message recipient must be a non-empty string")
        payload = dict(self.payload or {})
        missing = [k for k in _REQUIRED_PAYLOAD[self.performative] if k not in payload]
        if missing:
            raise ValueError(
                f"{self.performative.name} payload missing required key(s): "
                f"{', '.join(missing)}"
            )
        # Freeze the payload so recipients cannot mutate delivered messages.
        object.__setattr__(self, "payload", MappingProxyType(payload))


# ---------------------------------------------------------------------------
# F3 — relative position reports
# ---------------------------------------------------------------------------
def relative_report_payload(
    reporter_pose: Tuple[float, float], room: str,
    obj_xy: Tuple[float, float], extra: Optional[Mapping[str, Any]] = None,
) -> dict:
    """Build a relative-position report payload (F3).

    A reporting robot knows its own pose exactly, so it reports where the object
    is *relative to itself* rather than an absolute world position: the receiver
    reconstructs the absolute position with :func:`reconstruct_location`.

    Args:
        reporter_pose: The reporter's own world ``(x, y)`` (m).
        room: The reporter's current named region / room.
        obj_xy: The object's world ``(x, y)`` (m), as the reporter perceives it.
        extra: Extra payload fields to merge in (e.g. ``{"query": q}``).

    Returns:
        A payload dict with ``reporter_pose``, ``room`` and ``rel_offset`` set
        (plus any ``extra`` fields). No absolute location is included.
    """
    rp = (float(reporter_pose[0]), float(reporter_pose[1]))
    out: dict = dict(extra or {})
    out["reporter_pose"] = rp
    out["room"] = str(room)
    out["rel_offset"] = (float(obj_xy[0]) - rp[0], float(obj_xy[1]) - rp[1])
    return out


def reconstruct_location(payload: Mapping[str, Any]) -> Tuple[float, float]:
    """Reconstruct an object's absolute ``(x, y)`` from a relative report (F3).

    ``absolute = reporter_pose + rel_offset`` — the protocol always consumes this
    reconstruction (never a transmitted absolute), so a robot that misplaces its
    own pose is the only way a report can be wrong.

    Args:
        payload: A :data:`Performative.REPORT_VISIBILITY` (visible) or
            :data:`Performative.REPORT_FOUND` payload.

    Returns:
        The reconstructed absolute world ``(x, y)`` (m).
    """
    rp = payload["reporter_pose"]
    ro = payload["rel_offset"]
    return (float(rp[0]) + float(ro[0]), float(rp[1]) + float(ro[1]))
