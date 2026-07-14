"""text.py — ASCII sanitising, word-wrap and generic transcript flattening.

cv2's Hershey fonts render non-ASCII as ``?``, so every drawn string passes
through :func:`ascii_sanitize` first. :func:`flatten_transcript` turns arbitrary
message-like objects into coloured, wrapped render lines WITHOUT knowing the
mission shape: known performatives reuse the project's own caption phrasing
(:func:`code.comms.bus._phrase`) and anything else (e.g. a new CLARIFY flow the
fleet agent adds) falls back to common payload text keys, so clarify/user lines
appear automatically.
"""

from __future__ import annotations

from typing import Any, Dict, List, Sequence, Tuple

from code.apps.demos import style

# Typographic -> ASCII (superset of code/fleet/mission_video.py's map).
_ASCII_MAP: Dict[str, str] = {
    "—": "-", "–": "-",      # em / en dash
    "‘": "'", "’": "'",      # curly single quotes
    "“": '"', "”": '"',      # curly double quotes
    "→": "->", "…": "...",   # arrow, ellipsis
    "•": "-",                       # bullet
}

# Payload keys tried (in order) when a performative has no known caption phrase —
# covers clarify questions/replies and any free-text status a new flow adds.
_TEXT_KEYS: Tuple[str, ...] = (
    "text", "question", "prompt", "reply", "answer", "message", "body", "reason")


def ascii_sanitize(text: str) -> str:
    """Map known typographic glyphs to ASCII and drop any other non-ASCII."""
    for src, dst in _ASCII_MAP.items():
        text = text.replace(src, dst)
    return text.encode("ascii", errors="ignore").decode("ascii")


def wrap(text: str, width_chars: int) -> List[str]:
    """Greedy word-wrap to ``width_chars`` per line (long words hard-split)."""
    out: List[str] = []
    cur = ""
    for word in text.split():
        while len(word) > width_chars:               # hard-split over-long tokens
            if cur:
                out.append(cur)
                cur = ""
            out.append(word[:width_chars])
            word = word[width_chars:]
        if cur and len(cur) + 1 + len(word) > width_chars:
            out.append(cur)
            cur = word
        else:
            cur = f"{cur} {word}".strip()
    if cur:
        out.append(cur)
    return out or [""]


def message_phrase(msg: Any) -> str:
    """A human phrase for one message, robust to unknown performatives."""
    try:
        from code.comms.bus import _phrase
        phrase = _phrase(msg.performative, msg.payload, getattr(msg, "sender", ""))
    except Exception:
        phrase = ""
    if phrase:
        return phrase
    payload = getattr(msg, "payload", {}) or {}
    for key in _TEXT_KEYS:
        val = payload.get(key)
        if isinstance(val, str) and val:
            return val
    for val in payload.values():                     # last resort: any string
        if isinstance(val, str) and val:
            return val
    return ""


def _perf_name(msg: Any) -> str:
    perf = getattr(msg, "performative", None)
    return getattr(perf, "name", str(perf)) if perf is not None else ""


def flatten_transcript(transcript: Sequence[Any], width_chars: int,
                       ) -> List[Tuple[str, style.BGR, str]]:
    """Flatten messages into ``(kind, color, text)`` render lines, oldest first.

    ``kind`` is ``"head"`` (``sender -> recipient  PERFORMATIVE``, in the sender's
    colour) or ``"body"`` (a wrapped phrase line, dimmed). One head line plus zero
    or more body lines per message.
    """
    out: List[Tuple[str, style.BGR, str]] = []
    for msg in transcript:
        sender = getattr(msg, "sender", "?")
        recipient = getattr(msg, "recipient", "?")
        color = style.sender_bgr(sender)
        perf = _perf_name(msg)
        head = ascii_sanitize(f"{sender} -> {recipient}")
        if perf:
            head = f"{head}   {perf}"
        out.append(("head", color, head))
        phrase = message_phrase(msg)
        if phrase:
            body_col = style.dim(color, 0.72)
            for line in wrap(ascii_sanitize(phrase), width_chars):
                out.append(("body", body_col, line))
    return out


def tail_lines(lines: Sequence[Tuple[str, style.BGR, str]], max_lines: int,
               ) -> List[Tuple[str, style.BGR, str]]:
    """Keep the last ``max_lines`` render lines (auto-scroll to newest)."""
    if max_lines <= 0:
        return []
    return list(lines[-max_lines:])
