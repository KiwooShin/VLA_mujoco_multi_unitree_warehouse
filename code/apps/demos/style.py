"""style.py — Shared visual constants for the Demo Set v2 composer.

Pure data only (no cv2 / MuJoCo), so the layout / text / effects unit tests can
import the palette without pulling a rendering backend. Colours are BGR (cv2's
channel order). The per-callsign accents mirror
:data:`code.fleet.viz.ACCENT_RGBA` (kept as literals here so this module stays
import-light, exactly as ``code/fleet/mission_video.py`` hardcodes its own
sender palette).
"""

from __future__ import annotations

from typing import Dict, Tuple

BGR = Tuple[int, int, int]

# ---------------------------------------------------------------------------
# Theme (dark, recruiter-facing polish)
# ---------------------------------------------------------------------------
CANVAS_BG: BGR = (18, 18, 20)      # whole-frame backdrop
BEV_FRAME_BG: BGR = (12, 12, 14)   # letterbox behind the fitted BEV map
PANEL_BG: BGR = (26, 26, 30)       # comms panel
PANEL_TITLE_BG: BGR = (38, 38, 46)  # title bar within the panel
STRIP_BG: BGR = (14, 14, 16)       # ego strip backdrop
TILE_BG: BGR = (22, 22, 26)        # per-robot tile backdrop
CHIP_BG: BGR = (34, 34, 40)        # state-chip pill
DIVIDER: BGR = (70, 70, 82)        # hairline separators

TEXT_PRIMARY: BGR = (238, 238, 240)
TEXT_MUTED: BGR = (168, 168, 176)
TEXT_DIM: BGR = (120, 120, 128)
SHADOW: BGR = (0, 0, 0)

RING_GOLD: BGR = (60, 205, 255)    # default target-ring accent
PAD_GREEN: BGR = (120, 214, 150)   # delivery-pad highlight

# Per-callsign accent (BGR) — b,g,r derived from code.fleet.viz.ACCENT_RGBA.
ACCENT_BGR: Dict[str, BGR] = {
    "Alpha": (30, 30, 224),
    "Bravo": (229, 86, 40),
    "Charlie": (25, 204, 239),
    "Delta": (209, 61, 158),
    "Echo": (188, 204, 25),
    "Foxtrot": (25, 137, 247),
}
DEFAULT_ACCENT: BGR = (140, 216, 78)

# Non-robot transcript senders.
SENDER_BGR: Dict[str, BGR] = {
    "user": (236, 236, 238),
    "allocator": (150, 230, 150),
    "fleet": (210, 190, 120),
}

# ---------------------------------------------------------------------------
# Geometry / typography
# ---------------------------------------------------------------------------
CANVAS_W: int = 1600
CANVAS_H: int = 900
PANEL_W: int = 540
STRIP_H: int = 240
OUTER_MARGIN: int = 16

# Comms panel internals.
PANEL_PAD: int = 18
PANEL_TITLE_H: int = 66
LINE_H: int = 19          # transcript body line height (px)
WRAP_CHARS: int = 46      # transcript word-wrap width (chars)

# Ego strip internals.
TILE_GAP: int = 10
TILE_MARGIN: int = 12
TILE_NAME_H: int = 20
TILE_CHIP_H: int = 24
EGO_ASPECT: float = 320.0 / 240.0   # native FleetViz ego aspect (4:3)

# BEV render size (<= the 960x720 FleetViz offscreen buffer). 1.6 aspect keeps
# letterbox bars off the ~1060x660 BEV region.
BEV_RENDER_W: int = 960
BEV_RENDER_H: int = 600


def accent_bgr(name: str) -> BGR:
    """Return a callsign's accent colour (BGR), or the fleet default."""
    return ACCENT_BGR.get(name, DEFAULT_ACCENT)


def sender_bgr(sender: str) -> BGR:
    """Return the transcript colour for any sender (robot accent or role)."""
    if sender in ACCENT_BGR:
        return ACCENT_BGR[sender]
    return SENDER_BGR.get(sender, TEXT_MUTED)


def dim(color: BGR, factor: float) -> BGR:
    """Scale a BGR colour's brightness by ``factor`` (clamped to [0, 255])."""
    return tuple(int(max(0, min(255, c * factor))) for c in color)  # type: ignore[return-value]
