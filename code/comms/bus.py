"""bus.py — Synchronous, deterministic message bus for the fleet comms layer.

The bus assigns each posted :class:`~code.comms.messages.Message` a monotonic id
and stamps it with the current simulation step (read from a caller-supplied
clock), routes it into per-recipient FIFO inboxes, and keeps the full ordered
transcript for demo overlays and logging. There are no threads: the co-simulation
loop pumps the bus at step boundaries by calling :meth:`MessageBus.drain` for each
robot, so behaviour is fully reproducible.

``transcript_lines`` renders crisp, non-technical captions ("t=1200 Alpha->Bravo
QUERY_VISIBILITY: can you see the red cube?") intended to be burned onto demo
video frames.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

from code.comms.messages import Message, Performative

# Recipient label that routes to the fleet allocator inbox (Phase 4 hook).
FLEET: str = "fleet"


# ---------------------------------------------------------------------------
# Human-readable formatting (per-performative caption phrases)
# ---------------------------------------------------------------------------
def _xy(loc: Any) -> str:
    """Format an ``(x, y)`` location tuple as ``"(1.5, 0.0)"`` (or ``"?"``)."""
    try:
        x, y = loc
        return f"({float(x):.1f}, {float(y):.1f})"
    except (TypeError, ValueError):
        return "?"


def _describe(obj: Any) -> str:
    """Return ``obj.describe()`` if available, else ``str(obj)``."""
    describe = getattr(obj, "describe", None)
    return describe() if callable(describe) else str(obj)


def _report_sentence(sender: str, payload: Mapping[str, Any]) -> str:
    """Render the F3 fixed relative-position report sentence.

    The exact, user-mandated wording (docs/final_demo_spec.md F3): the reporter
    names itself and its room, states its own exact position, and gives the
    object's offset *relative to itself* (dx, dy) — the receiver reconstructs
    the absolute position from ``reporter_pose + rel_offset``.
    """
    rp = payload.get("reporter_pose", (0.0, 0.0))
    ro = payload.get("rel_offset", (0.0, 0.0))
    room = payload.get("room", "the area")
    name = sender or "?"
    return (f"I am robot {name}, currently in {room} at position "
            f"({float(rp[0]):.1f}, {float(rp[1]):.1f}). The object is located "
            f"{float(ro[0]):.1f} m and {float(ro[1]):.1f} m away from me.")


def _phrase(perf: Performative, payload: Mapping[str, Any],
            sender: str = "") -> str:
    """Return the human caption phrase for one message's performative + payload.

    Args:
        perf: The message performative.
        payload: The message payload.
        sender: The message sender's name, used to render the F3 relative
            position-report sentence in the reporter's own voice.
    """
    if perf is Performative.REQUEST_TASK:
        return _describe(payload.get("task"))
    if perf is Performative.QUERY_VISIBILITY:
        return f"can you see the {_describe(payload.get('query'))}?"
    if perf is Performative.REPORT_VISIBILITY:
        what = _describe(payload.get("query"))
        if payload.get("visible"):
            return _report_sentence(sender, payload)  # F3 relative report
        return f"no, I can't see the {what}"
    if perf is Performative.COMMAND_SEARCH:
        what = _describe(payload.get("query"))
        if payload.get("cancel"):
            return f"stand down — the {what} has been found"
        return f"search the {payload.get('region', '?')} area for the {what}"
    if perf is Performative.ACCEPT:
        region = payload.get("region")
        return f"on it — searching {region}" if region else "on it"
    if perf is Performative.REJECT:
        return f"can't — {payload.get('reason', 'unavailable')}"
    if perf is Performative.REPORT_FOUND:
        return _report_sentence(sender, payload)  # F3 relative report
    if perf in (Performative.STATUS_UPDATE, Performative.TASK_COMPLETE):
        return str(payload.get("text", ""))
    if perf is Performative.TASK_FAILED:
        return f"couldn't finish — {payload.get('reason', 'unknown reason')}"
    if perf is Performative.FLEET_REQUEST:
        return f"any robot: {_describe(payload.get('task'))}"
    if perf is Performative.CLARIFY:
        return str(payload.get("question", "which one do you mean?"))
    if perf is Performative.USER_REPLY:
        text = payload.get("text")
        return str(text) if text else f"I mean the {_describe(payload.get('query'))}"
    return ""


def format_line(msg: Message) -> str:
    """Render one message as a demo caption line.

    Args:
        msg: The message to render.

    Returns:
        A string such as
        ``"t=1200 Alpha->Bravo QUERY_VISIBILITY: can you see the red cube?"``.
    """
    return (f"t={msg.t_step} {msg.sender}->{msg.recipient} "
            f"{msg.performative.name}: "
            f"{_phrase(msg.performative, msg.payload, msg.sender)}")


# ---------------------------------------------------------------------------
# MessageBus
# ---------------------------------------------------------------------------
class MessageBus:
    """A synchronous, deterministic, per-recipient FIFO message bus.

    Messages addressed to ``"fleet"`` are delivered to the configurable
    allocator inbox (the Phase-4 task-allocator hook); every other recipient
    gets its own FIFO inbox keyed by callsign / ``"user"``.
    """

    def __init__(self, clock: Callable[[], int],
                 allocator_inbox: str = "allocator") -> None:
        """Initialise the bus.

        Args:
            clock: Zero-argument callable returning the current sim step; used to
                stamp ``t_step`` on every posted message.
            allocator_inbox: Inbox name that ``"fleet"``-addressed messages route
                to (drained by the Phase-4 allocator).
        """
        self._clock = clock
        self._allocator_inbox = allocator_inbox
        self._next_id = 0
        self._inboxes: Dict[str, List[Message]] = {}
        self._transcript: List[Message] = []

    @property
    def allocator_inbox(self) -> str:
        """Name of the inbox that ``"fleet"``-addressed messages route to."""
        return self._allocator_inbox

    def _key(self, recipient: str) -> str:
        """Map a recipient label to its inbox key (``"fleet"`` -> allocator)."""
        return self._allocator_inbox if recipient == FLEET else recipient

    def post(self, sender: str, recipient: str, performative: Performative,
             payload: Optional[Mapping[str, Any]] = None,
             in_reply_to: Optional[int] = None) -> Message:
        """Build, record and route a message.

        Args:
            sender: ``"user"`` or a robot callsign.
            recipient: A robot callsign, ``"user"``, or ``"fleet"``.
            performative: The speech act.
            payload: Performative-specific fields (validated by :class:`Message`).
            in_reply_to: ``msg_id`` this message answers, or ``None``.

        Returns:
            The posted :class:`Message`, with ``msg_id`` and ``t_step`` assigned.

        Raises:
            ValueError: If the payload is missing a required key.
        """
        msg = Message(
            msg_id=self._next_id,
            t_step=int(self._clock()),
            sender=sender,
            recipient=recipient,
            performative=performative,
            payload=dict(payload or {}),
            in_reply_to=in_reply_to,
        )
        self._next_id += 1
        self._transcript.append(msg)
        self._inboxes.setdefault(self._key(recipient), []).append(msg)
        return msg

    def drain(self, recipient: str) -> List[Message]:
        """Remove and return all queued messages for a recipient, in FIFO order.

        Args:
            recipient: A robot callsign, ``"user"``, ``"fleet"``, or the
                allocator inbox name.

        Returns:
            The queued messages (possibly empty); the inbox is left empty.
        """
        key = self._key(recipient)
        msgs = self._inboxes.get(key, [])
        self._inboxes[key] = []
        return msgs

    def pending(self, recipient: str) -> int:
        """Return how many messages are currently queued for ``recipient``."""
        return len(self._inboxes.get(self._key(recipient), []))

    @property
    def transcript(self) -> Tuple[Message, ...]:
        """The full ordered message history (read-only snapshot)."""
        return tuple(self._transcript)

    def transcript_lines(self, last_n: Optional[int] = None) -> List[str]:
        """Render the transcript (or its last ``last_n`` messages) as captions.

        Args:
            last_n: If given, only the most recent ``last_n`` messages are
                rendered; otherwise the whole transcript is.

        Returns:
            One caption string per message, oldest first (see :func:`format_line`).
        """
        msgs: Sequence[Message] = self._transcript
        if last_n is not None:
            msgs = msgs[-last_n:]
        return [format_line(m) for m in msgs]
