"""effects.py — Pure time-based effects: comm-glow decay and title-card fade.

No cv2 / MuJoCo. Both functions are the testable core of two composer polish
features:

* :func:`glow_levels` — a robot's ego tile border brightens when it sends OR
  receives a message and decays linearly to 0 over ``decay_steps`` (~2 s of
  video at the 50 Hz control loop), giving the "who is talking now" glow.
* :func:`title_card_alpha` — the opening full-frame title card holds solid then
  cross-fades to the live view.
"""

from __future__ import annotations

from typing import Any, Dict, Sequence


def glow_levels(transcript: Sequence[Any], step: int, names: Sequence[str],
                decay_steps: int) -> Dict[str, float]:
    """Return ``{callsign: freshness in [0, 1]}`` for the comm-glow borders.

    A robot's level is ``1 - age/decay_steps`` for the most recent message it
    sent or received (``age = step - msg.t_step``), or 0.0 if none is within the
    window. Every requested name is present in the result (idle -> 0.0), so the
    strip border is always drawable.

    Args:
        transcript: Message-like objects with ``sender``/``recipient``/``t_step``.
        step: Current simulation step.
        names: Callsigns to score.
        decay_steps: Steps over which a fresh message fades to nothing (> 0).
    """
    if decay_steps <= 0:
        raise ValueError(f"decay_steps must be > 0, got {decay_steps}")
    level = {n: 0.0 for n in names}
    wanted = set(names)
    for msg in transcript:
        age = step - int(msg.t_step)
        if age < 0 or age > decay_steps:
            continue
        fresh = 1.0 - age / decay_steps
        for who in (msg.sender, msg.recipient):
            if who in wanted and fresh > level[who]:
                level[who] = fresh
    return level


def title_card_alpha(sim_time: float, hold_secs: float,
                     fade_secs: float) -> float:
    """Opacity of the opening title card at ``sim_time`` (1 = card, 0 = live).

    Solid ``1.0`` until ``hold_secs - fade_secs``, then a linear cross-fade down
    to ``0.0`` at ``hold_secs``, and ``0.0`` after. A non-positive ``hold_secs``
    disables the card entirely.

    Args:
        sim_time: Elapsed simulated seconds.
        hold_secs: Total on-screen duration of the card (incl. the fade).
        fade_secs: Duration of the trailing cross-fade (clamped to ``hold_secs``).
    """
    if hold_secs <= 0.0:
        return 0.0
    fade = max(0.0, min(fade_secs, hold_secs))
    if sim_time >= hold_secs:
        return 0.0
    solid_until = hold_secs - fade
    if sim_time <= solid_until:
        return 1.0
    if fade <= 0.0:
        return 0.0
    return max(0.0, min(1.0, (hold_secs - sim_time) / fade))
