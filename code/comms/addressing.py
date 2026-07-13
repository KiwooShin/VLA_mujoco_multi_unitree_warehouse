"""addressing.py — Split the *addressee* off a natural-language instruction.

The warehouse accepts spoken/typed orders like ``"Alpha, fetch the red cube to
the delivery pad"`` or ``"someone bring me the yellow cone"``. This module peels
off *who* the order is for and returns the remaining *body* verbatim; the body is
still resolved into a concrete task later by the existing per-robot resolvers
(``code.apps.fancy.live.resolve_live_instruction`` /
``code.apps.repl.planner.Planner``) against that robot's own knowledge — this
module only handles addressing (docs/multi_plan.md sec 3, ``code/fleet`` upstream).

A named, known callsign routes to that robot. A fleet/broadcast word ("someone",
"anyone", "everyone", "robots") — or an imperative with no addressee at all —
routes to ``"fleet"`` (the Phase-4 allocator). Matching is case-insensitive and
tolerant of greetings ("hey"/"ok") and punctuation; the body preserves the
original casing so downstream resolvers see the user's exact words.
"""

from __future__ import annotations

import dataclasses
import re
from typing import Optional, Sequence

# Recipient label meaning "route to the allocator".
FLEET: str = "fleet"

# Leading conversational filler stripped before/after finding the addressee.
_GREETING = r"(?:hey|hi|yo|ok|okay|hello|please)"
_GREETING_RE = re.compile(rf"^\s*(?:{_GREETING})\b[\s,]*", re.IGNORECASE)
_LEAD_PLEASE_RE = re.compile(r"^\s*please\b[\s,]*", re.IGNORECASE)

# Words that address the whole fleet / an unspecified robot -> allocator.
_FLEET_WORDS = (
    "someone", "somebody", "anyone", "anybody", "everyone", "everybody",
    "whoever", "all robots", "robots", "team", "guys",
)
_FLEET_RE = re.compile(
    r"\b(" + "|".join(re.escape(w) for w in _FLEET_WORDS) + r")\b", re.IGNORECASE)


@dataclasses.dataclass(frozen=True)
class AddressedInstruction:
    """The addressee split off an instruction.

    Attributes:
        recipient: A canonical callsign, or ``"fleet"`` for allocator routing.
        body: The remaining instruction (original casing), addressee removed.
        is_fleet: Whether ``recipient`` is the fleet/allocator.
        matched_callsign: The callsign found (canonical), or ``None`` if fleet.
    """

    recipient: str
    body: str
    is_fleet: bool
    matched_callsign: Optional[str]


def _tidy(text: str) -> str:
    """Strip surrounding whitespace/punctuation and a leading greeting/"please"."""
    text = _GREETING_RE.sub("", text, count=1)
    text = _LEAD_PLEASE_RE.sub("", text)
    text = re.sub(r"^[\s,;:!.\-]+", "", text)
    text = re.sub(r"[\s,;:]+$", "", text)
    return re.sub(r"\s{2,}", " ", text).strip()


def parse_addressed_instruction(
        text: str, callsigns: Sequence[str]) -> AddressedInstruction:
    """Extract the target robot (or fleet) from a natural-language instruction.

    Args:
        text: The raw instruction, e.g. ``"hey bravo find the blue ball"``.
        callsigns: Known robot callsigns (e.g. ``code.warehouse.layout.CALLSIGNS``).

    Returns:
        An :class:`AddressedInstruction`. When several callsigns appear, the
        first (left-most) wins. When no callsign and no fleet word appear, an
        imperative body is routed to ``"fleet"``.
    """
    canon = {c.lower(): c for c in callsigns}
    stripped = _GREETING_RE.sub("", text or "", count=1)

    # 1) An explicitly named, known callsign wins (left-most if several).
    best: Optional[re.Match] = None
    for lower in canon:
        m = re.search(rf"\b{re.escape(lower)}\b", stripped, re.IGNORECASE)
        if m and (best is None or m.start() < best.start()):
            best = m
    if best is not None:
        recipient = canon[best.group(0).lower()]
        body = _tidy(stripped[:best.start()] + " " + stripped[best.end():])
        return AddressedInstruction(recipient, body, False, recipient)

    # 2) A fleet/broadcast word routes to the allocator.
    m = _FLEET_RE.search(stripped)
    if m is not None:
        body = _tidy(stripped[:m.start()] + " " + stripped[m.end():])
        return AddressedInstruction(FLEET, body, True, None)

    # 3) No addressee: an imperative body is a fleet request by default.
    return AddressedInstruction(FLEET, _tidy(stripped), True, None)
