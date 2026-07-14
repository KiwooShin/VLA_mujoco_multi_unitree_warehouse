"""code.comms — Pure-Python fleet communication layer (Phase 3).

User<->robot and robot<->robot messaging for the multi-robot warehouse:
typed addressed messages, a synchronous deterministic bus with a demo-caption
transcript, per-robot coordination protocols (query-visibility, delegate-search,
report-found, need-to-know reporting), and a natural-language addressing splitter.

No MuJoCo / GPU dependency — fully unit-testable (docs/multi_plan.md sec 1, 4).
"""

from __future__ import annotations

from code.comms.addressing import (
    AddressedInstruction,
    parse_addressed_instruction,
)
from code.comms.bus import FLEET, MessageBus, format_line
from code.comms.messages import (
    Message,
    ObjectQuery,
    Performative,
    TaskKind,
    TaskSpec,
    clarify_options,
    clarify_question,
    reconstruct_location,
    refine_query,
    relative_report_payload,
)
from code.comms.protocol import (
    DEFAULT_REGIONS,
    RobotActions,
    RobotProtocol,
    RobotState,
)

__all__ = [
    "AddressedInstruction",
    "parse_addressed_instruction",
    "MessageBus",
    "format_line",
    "FLEET",
    "Message",
    "ObjectQuery",
    "Performative",
    "TaskKind",
    "TaskSpec",
    "clarify_options",
    "clarify_question",
    "reconstruct_location",
    "refine_query",
    "relative_report_payload",
    "RobotActions",
    "RobotProtocol",
    "RobotState",
    "DEFAULT_REGIONS",
]
