"""draw.py — cv2 drawing primitives for the DemoComposer.

All rendering-backend (cv2) code lives here so :mod:`composer` stays orchestration
and the pure helpers (layout / effects / text / style) stay import-light. Every
public function draws in-place onto a BGR uint8 array; text is ASCII-sanitised and
anti-aliased with a soft shadow for legibility over the render.
"""

from __future__ import annotations

import math
from typing import Optional, Sequence

import cv2
import numpy as np

from code.apps.demos import style, text
from code.apps.demos.layout import Rect
from code.apps.demos.models import FrameState, Pad, PlannedPath, Ring, RobotFrame
from code.apps.warehouse_demo import bev as bevmod

_FONT = cv2.FONT_HERSHEY_SIMPLEX
_TRAIL_MAX_PTS = 300


# ---------------------------------------------------------------------------
# Text + shape helpers
# ---------------------------------------------------------------------------
def put_text(img: np.ndarray, s: str, org, scale: float, color: style.BGR,
             *, thick: int = 1, shadow: int = 3, center: bool = False) -> None:
    """Draw ASCII-sanitised, shadowed, anti-aliased text (optionally centred)."""
    s = text.ascii_sanitize(s)
    x, y = org
    if center:
        (tw, _), _ = cv2.getTextSize(s, _FONT, scale, thick)
        x = int(x - tw / 2)
    if shadow:
        cv2.putText(img, s, (x, y), _FONT, scale, style.SHADOW, thick + shadow,
                    cv2.LINE_AA)
    cv2.putText(img, s, (x, y), _FONT, scale, color, thick, cv2.LINE_AA)


def fill_rect(img: np.ndarray, r: Rect, color: style.BGR) -> None:
    """Fill a rectangle region (clipped to the image)."""
    cv2.rectangle(img, (r.x, r.y), (r.x1 - 1, r.y1 - 1), color, -1)


# ---------------------------------------------------------------------------
# BEV overlays (drawn in BEV render pixel space, then the whole map is fitted)
# ---------------------------------------------------------------------------
def draw_pad(img: np.ndarray, cam, pad: Pad, step: int) -> None:
    """Highlight a delivery pad: translucent fill + pulsing outline + label."""
    cx, cy = pad.xy
    corners = [(cx - pad.half_x, cy - pad.half_y), (cx + pad.half_x, cy - pad.half_y),
               (cx + pad.half_x, cy + pad.half_y), (cx - pad.half_x, cy + pad.half_y)]
    poly = np.array([cam.project_xy(c, z=0.02) for c in corners], dtype=np.int32)
    overlay = img.copy()
    cv2.fillPoly(overlay, [poly], style.dim(style.PAD_GREEN, 0.5))
    cv2.addWeighted(overlay, 0.28, img, 0.72, 0.0, img)
    pulse = 0.6 + 0.4 * abs(math.sin(step * 0.06))
    cv2.polylines(img, [poly], True, style.dim(style.PAD_GREEN, pulse), 2,
                  cv2.LINE_AA)
    if pad.label:
        u, v = cam.project_xy(pad.xy, z=0.02)
        put_text(img, pad.label, (u - 34, v + 4), 0.42, style.PAD_GREEN, shadow=2)


def draw_planned_path(img: np.ndarray, cam, path: PlannedPath) -> None:
    """Draw a planned-route polyline with subtle waypoint dots."""
    pts = list(path.pts)
    if len(pts) < 2:
        return
    color = path.color or style.RING_GOLD
    bevmod.draw_polyline(img, cam, pts, style.dim(color, 0.85), thickness=2, z=0.05)
    for p in pts[::2]:
        cv2.circle(img, cam.project_xy(p, z=0.05), 2, color, -1, cv2.LINE_AA)


def draw_ring(img: np.ndarray, cam, ring: Ring, step: int) -> None:
    """Draw a pulsing target ring (white halo + coloured ring + optional label)."""
    color = ring.color or style.RING_GOLD
    r = 12 + int(4 * abs(math.sin(step * 0.15)))
    c = cam.project_xy(ring.xy, z=0.3)
    cv2.circle(img, c, r + 3, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.circle(img, c, r, color, 2, cv2.LINE_AA)
    cv2.drawMarker(img, c, color, cv2.MARKER_CROSS, 10, 1, cv2.LINE_AA)
    if ring.label:
        put_text(img, ring.label, (c[0] + r + 6, c[1] + 4), 0.4, color, shadow=2)


def _trail_points(trail: Sequence) -> list:
    """Bounded, order-preserving stride of a trail (constant per-frame cost)."""
    trail = list(trail)
    if len(trail) < 2:
        return trail
    stride = max(1, len(trail) // _TRAIL_MAX_PTS)
    pts = trail[::stride]
    if pts[-1] != trail[-1]:
        pts.append(trail[-1])
    return pts


def draw_bev_overlays(img: np.ndarray, cam, state: FrameState) -> None:
    """Draw pads, planned paths, trails, robots+labels and rings (in-place)."""
    for pad in state.pads:
        draw_pad(img, cam, pad, state.step)
    for path in state.planned_paths:
        draw_planned_path(img, cam, path)
    for rf in state.robots:
        color = style.accent_bgr(rf.name)
        pts = _trail_points(rf.trail)
        if len(pts) >= 2:
            bevmod.draw_polyline(img, cam, pts, color, thickness=2, z=0.02)
    for rf in state.robots:
        color = style.accent_bgr(rf.name)
        bevmod.draw_robot(img, cam, rf.xy, rf.yaw, color=color)
        u, v = cam.project_xy(rf.xy, z=1.75)
        put_text(img, rf.name, (u - 20, v), 0.5, color, thick=1, shadow=3)
    for ring in state.rings:
        draw_ring(img, cam, ring, state.step)


# ---------------------------------------------------------------------------
# HUD (top-left of the BEV region)
# ---------------------------------------------------------------------------
def draw_hud(canvas: np.ndarray, bev: Rect, title: str, phase: str,
             sim_time: float) -> None:
    """Draw the demo title / phase / sim-time HUD over the BEV top-left."""
    x = bev.x + 20
    put_text(canvas, title, (x, bev.y + 34), 0.72, style.TEXT_PRIMARY, thick=2)
    put_text(canvas, f"PHASE   {phase}", (x, bev.y + 60), 0.5, style.RING_GOLD)
    put_text(canvas, f"TIME    {sim_time:5.1f} s", (x, bev.y + 82), 0.5,
             style.TEXT_MUTED)


# ---------------------------------------------------------------------------
# Comms panel (right column)
# ---------------------------------------------------------------------------
def draw_panel(canvas: np.ndarray, rect: Rect, *, title: str, project: str,
               mission_lines: Sequence[str], transcript: Sequence) -> None:
    """Draw the full-height comms panel: title bar, mission lines, transcript."""
    fill_rect(canvas, rect, style.PANEL_BG)
    fill_rect(canvas, Rect(rect.x, rect.y, rect.w, style.PANEL_TITLE_H),
              style.PANEL_TITLE_BG)
    px = rect.x + style.PANEL_PAD
    put_text(canvas, title, (px, rect.y + 30), 0.68, style.TEXT_PRIMARY, thick=2)
    put_text(canvas, project, (px, rect.y + 52), 0.44, style.TEXT_DIM, shadow=2)

    y = rect.y + style.PANEL_TITLE_H + 24
    wrap_chars = max(10, int((rect.w - 2 * style.PANEL_PAD) / 8.4))
    for line in mission_lines:
        for wl in text.wrap(text.ascii_sanitize(line), wrap_chars):
            put_text(canvas, wl, (px, y), 0.46, style.TEXT_MUTED, shadow=2)
            y += 22
    y += 6
    cv2.line(canvas, (px, y), (rect.x1 - style.PANEL_PAD, y), style.DIVIDER, 1,
             cv2.LINE_AA)
    y += 22
    put_text(canvas, "COMMS", (px, y), 0.5, style.TEXT_PRIMARY, thick=1)
    y += 14
    cv2.line(canvas, (px, y), (rect.x1 - style.PANEL_PAD, y), style.DIVIDER, 1,
             cv2.LINE_AA)
    _draw_transcript(canvas, rect, top=y + 18, transcript=transcript)


def _draw_transcript(canvas: np.ndarray, rect: Rect, *, top: int,
                     transcript: Sequence) -> None:
    """Render the sender-coloured, word-wrapped, auto-scrolled transcript."""
    px = rect.x + style.PANEL_PAD
    bottom = rect.y1 - style.PANEL_PAD
    max_lines = max(0, (bottom - top) // style.LINE_H)
    wrap_chars = max(10, int((rect.w - 2 * style.PANEL_PAD) / 8.0))
    lines = text.flatten_transcript(transcript, wrap_chars)
    y = top
    for kind, color, s in text.tail_lines(lines, max_lines):
        if kind == "head":
            put_text(canvas, s, (px, y), 0.42, color, thick=1, shadow=2)
        else:
            put_text(canvas, s, (px + 12, y), 0.4, color, thick=1, shadow=2)
        y += style.LINE_H


# ---------------------------------------------------------------------------
# Ego strip (bottom)
# ---------------------------------------------------------------------------
def draw_tile(canvas: np.ndarray, tile: Rect, ego: Rect, *, name: str,
              ego_bgr: np.ndarray, chip: str, glow: float) -> None:
    """Draw one robot tile: name, live ego, comm-glow border and state chip."""
    accent = style.accent_bgr(name)
    fill_rect(canvas, tile, style.TILE_BG)
    put_text(canvas, name, (tile.x + 4, tile.y + 15), 0.46, accent, thick=1,
             shadow=2)
    resized = cv2.resize(ego_bgr, (ego.w, ego.h), interpolation=cv2.INTER_AREA)
    canvas[ego.y:ego.y1, ego.x:ego.x1] = resized
    # Comm-glow border: a subtle always-on accent frame that thickens/brightens
    # with freshness (glow in [0, 1]).
    base = style.dim(accent, 0.34)
    cv2.rectangle(canvas, (ego.x, ego.y), (ego.x1 - 1, ego.y1 - 1), base, 1,
                  cv2.LINE_AA)
    if glow > 0.02:
        for k in range(3, 0, -1):
            a = glow * (k / 3.0)
            col = style.dim(accent, 0.35 + 0.65 * a)
            pad = k
            cv2.rectangle(canvas, (ego.x - pad, ego.y - pad),
                          (ego.x1 - 1 + pad, ego.y1 - 1 + pad), col, 1, cv2.LINE_AA)
    chip_r = Rect(tile.x, tile.y1 - style.TILE_CHIP_H + 2, tile.w,
                  style.TILE_CHIP_H - 2)
    fill_rect(canvas, chip_r, style.CHIP_BG)
    chip_txt = text.ascii_sanitize(chip)
    max_chars = max(3, int(tile.w / 7.0))
    if len(chip_txt) > max_chars:
        chip_txt = chip_txt[:max_chars - 1] + "."
    put_text(canvas, chip_txt, (chip_r.x + 6, chip_r.y + 15), 0.4,
             style.TEXT_PRIMARY, shadow=2)


# ---------------------------------------------------------------------------
# Title card
# ---------------------------------------------------------------------------
def make_title_card(shape, *, title: str, description: str, project: str,
                    ) -> np.ndarray:
    """Render a full-frame opening title card (dark, centred)."""
    h, w = shape[:2]
    card = np.full((h, w, 3), style.CANVAS_BG, dtype=np.uint8)
    cx = w // 2
    put_text(card, project.upper(), (cx, int(h * 0.34)), 0.7, style.RING_GOLD,
             thick=1, center=True)
    put_text(card, title, (cx, int(h * 0.46)), 1.7, style.TEXT_PRIMARY, thick=3,
             center=True)
    cv2.line(card, (cx - 200, int(h * 0.50)), (cx + 200, int(h * 0.50)),
             style.DIVIDER, 1, cv2.LINE_AA)
    put_text(card, description, (cx, int(h * 0.57)), 0.7, style.TEXT_MUTED,
             thick=1, center=True)
    return card


def blend_over(canvas: np.ndarray, card: np.ndarray, alpha: float) -> None:
    """Alpha-blend the title card over the live canvas in-place (1 = full card)."""
    if alpha <= 0.0:
        return
    if alpha >= 1.0:
        canvas[:] = card
        return
    cv2.addWeighted(card, alpha, canvas, 1.0 - alpha, 0.0, canvas)
