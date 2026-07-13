"""commands.py — Pure command validation for typed fleet orders.

Turns a raw typed instruction into an accept/reject decision *before* it ever
reaches the sim thread, so the user gets an immediate, friendly response:

* unknown callsign ("Zulu, fetch ...") -> a hint listing the real robots,
* unparseable object ("Alpha, do a barrel roll") -> a hint to name a colour
  and/or shape,
* otherwise an :class:`CommandCheck` describing who it is for and what object.

Only :mod:`code.comms.addressing` (pure, no MuJoCo) is imported, so this module
is trivially unit-testable. The colour/shape vocabulary mirrors
``code.sim.arena_build.COLORS`` / ``SHAPES`` (kept as a literal here to avoid
pulling the MuJoCo-heavy ``code.sim`` package into the validation path).
"""

from __future__ import annotations

import dataclasses
import re
from typing import List, Optional, Sequence

from code.comms.addressing import parse_addressed_instruction

# Mirrors code.sim.arena_build.COLORS / SHAPES (names only).
COLOR_WORDS: tuple[str, ...] = (
    "red", "yellow", "blue", "green", "orange", "purple", "cyan")
SHAPE_WORDS: tuple[str, ...] = ("ball", "cube", "cylinder", "cone")

# Words that legitimately open an order without being a callsign (greetings +
# fleet/broadcast words), used to suppress the unknown-callsign heuristic.
_NON_CALLSIGN_LEAD = {
    "hey", "hi", "yo", "ok", "okay", "hello", "please",
    "someone", "somebody", "anyone", "anybody", "everyone", "everybody",
    "whoever", "robots", "team", "guys", "all",
}
_LEADING_NAME_RE = re.compile(r"^\s*([A-Za-z]{2,})\s*,")


@dataclasses.dataclass(frozen=True)
class CommandCheck:
    """The outcome of validating a typed order.

    Attributes:
        ok: Whether the order is accepted.
        error: A friendly, user-facing reason when ``ok`` is False, else ``""``.
        is_fleet: Whether the order routes to the allocator (fleet-addressed).
        recipient: Canonical callsign, or ``"fleet"`` (only meaningful if ``ok``).
        recipient_label: Human label for messages ("Alpha" / "the fleet").
        target_desc: Human object phrase, e.g. ``"red cube"`` (only if ``ok``).
    """

    ok: bool
    error: str = ""
    is_fleet: bool = False
    recipient: str = ""
    recipient_label: str = ""
    target_desc: str = ""


def _resolve_object(body: str) -> Optional[str]:
    """Return a ``"<colour> <shape>"`` phrase from ``body``, or ``None``.

    Mirrors :func:`code.fleet.mission.resolve_query` but returns a display
    phrase instead of a query object (this module stays MuJoCo-free). Either
    field may be absent; ``None`` means neither a colour nor a shape was named.
    """
    low = f" {body.lower()} "
    color = next((c for c in COLOR_WORDS if f" {c}" in low), None)
    shape = next((s for s in SHAPE_WORDS if f" {s}" in low), None)
    if color is None and shape is None:
        return None
    if color and shape:
        return f"{color} {shape}"
    return color or shape or None


def validate_command(text: str, callsigns: Sequence[str]) -> CommandCheck:
    """Validate a raw typed order against the fleet's callsigns and vocabulary.

    Args:
        text: The raw instruction as typed by the user.
        callsigns: Known robot callsigns (e.g. ``("Alpha", "Bravo", ...)``).

    Returns:
        A :class:`CommandCheck`; ``ok`` is False (with a friendly ``error``) for
        an empty order, an unknown named callsign, or an order that names no
        recognizable object.
    """
    text = (text or "").strip()
    known = {c.lower() for c in callsigns}
    if not text:
        return CommandCheck(
            False, "Type a command, e.g. “Alpha, fetch the red cube to the "
            "delivery pad”.")

    # Unknown-callsign heuristic: a leading "<Name>," that is neither a known
    # robot nor a greeting/fleet word is almost certainly a mistyped callsign
    # (the addressing parser would otherwise silently route it to the fleet).
    m = _LEADING_NAME_RE.match(text)
    if m:
        lead = m.group(1).lower()
        if lead not in known and lead not in _NON_CALLSIGN_LEAD:
            roster = ", ".join(callsigns)
            return CommandCheck(
                False, f"I don't have a robot called “{m.group(1)}”. "
                f"Try {roster} — or say “someone”.")

    addr = parse_addressed_instruction(text, callsigns)
    target = _resolve_object(addr.body)
    if target is None:
        return CommandCheck(
            False, "I couldn't tell which object you mean. Name a colour and/or "
            "shape, e.g. “the red cube” or “the blue ball”.")

    label = "the fleet" if addr.is_fleet else addr.recipient
    return CommandCheck(True, "", addr.is_fleet, addr.recipient, label, target)


def example_commands() -> List[dict]:
    """Return the example-command buttons shown in the UI (seed-0 scene)."""
    return [
        {"label": "Alpha → red cube",
         "text": "Alpha, fetch the red cube to the delivery pad"},
        {"label": "someone → blue ball",
         "text": "someone bring me the blue ball"},
        {"label": "Charlie → green cylinder",
         "text": "Charlie, fetch the green cylinder to the delivery pad"},
    ]
