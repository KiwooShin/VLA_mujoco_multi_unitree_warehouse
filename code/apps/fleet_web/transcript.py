"""transcript.py — A persistent, incrementally-fetchable comms transcript.

The UI streams the fleet's chatter into a scrolling panel and polls
``/state?after=<id>`` for *new* lines only. Because the sim thread rebuilds a
fresh :class:`~code.fleet.mission.MissionRunner` (and therefore a fresh, empty
message bus) per mission, this log accumulates lines *across* missions under one
monotonic id space so the panel keeps its history.

The log is pure Python (no MuJoCo/comms imports) so it unit-tests in isolation;
callers hand it already-formatted ``(sender, recipient, text)`` triples.
"""

from __future__ import annotations

import dataclasses
from typing import Dict, List


# Coarse sender categories that drive per-line colouring in the UI.
def kind_for(sender: str) -> str:
    """Map a message sender to a UI style class."""
    low = sender.lower()
    if low in ("you", "user"):
        return "user"
    if low == "allocator":
        return "allocator"
    if low == "system":
        return "system"
    return "robot"


@dataclasses.dataclass(frozen=True)
class Entry:
    """One transcript line with a stable, monotonic id.

    Attributes:
        id: Monotonic id (unique for the life of the log; ``> 0``).
        sender: Who spoke ("you" / a callsign / "allocator" / "system").
        recipient: Addressee ("Alpha" / "user" / "the fleet" / "").
        text: The rendered line body.
        kind: UI style class (see :func:`kind_for`).
    """

    id: int
    sender: str
    recipient: str
    text: str
    kind: str

    def as_dict(self) -> Dict[str, object]:
        """Return a JSON-serializable view of this entry."""
        return {"id": self.id, "sender": self.sender,
                "recipient": self.recipient, "text": self.text,
                "kind": self.kind}


class TranscriptLog:
    """An append-only comms log with since-id incremental reads.

    Not internally locked; the owning :class:`FleetService` serializes access.
    """

    def __init__(self) -> None:
        self._entries: List[Entry] = []
        self._next_id: int = 1

    def append(self, sender: str, recipient: str, text: str,
               kind: str = "") -> Entry:
        """Append one line and return the created :class:`Entry`."""
        entry = Entry(self._next_id, sender, recipient, text,
                      kind or kind_for(sender))
        self._entries.append(entry)
        self._next_id += 1
        return entry

    def since(self, after: int) -> List[Entry]:
        """Return every entry whose id is strictly greater than ``after``."""
        if after <= 0:
            return list(self._entries)
        return [e for e in self._entries if e.id > after]

    def dicts_since(self, after: int) -> List[Dict[str, object]]:
        """Return :meth:`since` entries as JSON-serializable dicts."""
        return [e.as_dict() for e in self.since(after)]

    @property
    def last_id(self) -> int:
        """The id of the most recent entry (0 if the log is empty)."""
        return self._entries[-1].id if self._entries else 0

    def __len__(self) -> int:
        return len(self._entries)
