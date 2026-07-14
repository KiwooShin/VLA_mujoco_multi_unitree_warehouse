"""layout.py — Pure canvas geometry for the 1600x900 demo composer.

Splits the frame into three non-overlapping regions (BEV map, comms panel, ego
strip), sizes the per-robot ego tiles for any robot count (4 and 6 both fit and
the tile shrinks to stay in-strip), and computes the aspect-preserving fit of the
rendered BEV into its region. No cv2 / MuJoCo — trivially unit-testable.
"""

from __future__ import annotations

import dataclasses
from typing import Dict, List, Tuple

from code.apps.demos import style


@dataclasses.dataclass(frozen=True)
class Rect:
    """An axis-aligned pixel rectangle ``(x, y, w, h)`` (top-left origin)."""

    x: int
    y: int
    w: int
    h: int

    @property
    def x1(self) -> int:
        return self.x + self.w

    @property
    def y1(self) -> int:
        return self.y + self.h

    def contains_rect(self, other: "Rect") -> bool:
        """Whether ``other`` lies fully inside this rectangle."""
        return (other.x >= self.x and other.y >= self.y
                and other.x1 <= self.x1 and other.y1 <= self.y1)


def overlaps(a: Rect, b: Rect) -> bool:
    """Whether two rectangles share any interior area."""
    return not (a.x1 <= b.x or b.x1 <= a.x or a.y1 <= b.y or b.y1 <= a.y)


def regions(canvas_w: int = style.CANVAS_W, canvas_h: int = style.CANVAS_H,
            panel_w: int = style.PANEL_W, strip_h: int = style.STRIP_H,
            ) -> Dict[str, Rect]:
    """Return the ``bev`` / ``panel`` / ``strip`` regions (non-overlapping).

    The comms panel is the full-height right column; the BEV map and ego strip
    stack in the remaining left column. Together they tile the canvas exactly.
    """
    left_w = canvas_w - panel_w
    panel = Rect(left_w, 0, panel_w, canvas_h)
    strip = Rect(0, canvas_h - strip_h, left_w, strip_h)
    bev = Rect(0, 0, left_w, canvas_h - strip_h)
    return {"bev": bev, "panel": panel, "strip": strip}


def tile_rects(strip: Rect, n: int, *, gap: int = style.TILE_GAP,
               margin: int = style.TILE_MARGIN) -> List[Rect]:
    """Lay ``n`` equal ego tiles across the strip (works for 4 and 6).

    Tiles share the strip's inner height; width divides the inner width evenly
    after gaps, so more robots yield narrower tiles — every robot stays visible.

    Raises:
        ValueError: If ``n`` < 1 or the strip is too small for ``n`` tiles.
    """
    if n < 1:
        raise ValueError(f"need at least one tile, got {n}")
    inner_x = strip.x + margin
    inner_y = strip.y + margin
    inner_w = strip.w - 2 * margin
    inner_h = strip.h - 2 * margin
    tile_w = (inner_w - (n - 1) * gap) // n
    if tile_w <= 0 or inner_h <= 0:
        raise ValueError(f"strip {strip} too small for {n} tiles")
    out: List[Rect] = []
    x = inner_x
    for _ in range(n):
        out.append(Rect(x, inner_y, tile_w, inner_h))
        x += tile_w + gap
    return out


def ego_area(tile: Rect, *, name_h: int = style.TILE_NAME_H,
             chip_h: int = style.TILE_CHIP_H, aspect: float = style.EGO_ASPECT,
             ) -> Rect:
    """The live-ego sub-rectangle inside a tile (name above, chip below).

    Height is bounded by the space left after the name row and chip pill, and by
    the tile width at the ego aspect, so the ego image never overflows the tile.
    """
    avail_h = tile.h - name_h - chip_h
    ego_h = min(avail_h, int(round(tile.w / aspect)))
    ego_h = max(1, ego_h)
    ego_w = min(tile.w, int(round(ego_h * aspect)))
    ex = tile.x + (tile.w - ego_w) // 2
    ey = tile.y + name_h
    return Rect(ex, ey, ego_w, ego_h)


def contain_fit(src_w: int, src_h: int, box_w: int, box_h: int,
                ) -> Tuple[int, int, int, int]:
    """Aspect-preserving fit of ``src`` into ``box``; centre with letterbox.

    Returns ``(off_x, off_y, w, h)``: the placement of the scaled image inside a
    box-sized canvas (offsets are the letterbox margins).
    """
    if src_w <= 0 or src_h <= 0:
        raise ValueError("source size must be positive")
    scale = min(box_w / src_w, box_h / src_h)
    w = max(1, int(round(src_w * scale)))
    h = max(1, int(round(src_h * scale)))
    off_x = (box_w - w) // 2
    off_y = (box_h - h) // 2
    return off_x, off_y, w, h
