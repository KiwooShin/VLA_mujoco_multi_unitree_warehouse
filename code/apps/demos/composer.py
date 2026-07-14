"""composer.py — DemoComposer: one polished 1600x900 frame per call.

Layout (regions are non-overlapping; see :mod:`code.apps.demos.layout`)::

    +---------------------------------+----------------+
    |  BEV map (fit hall) + HUD       |  COMMS panel   |
    |  trails / rings / paths / pad   |  title bar     |
    |                                 |  mission lines |
    +---------------------------------+  transcript    |
    |  ALWAYS-ON ego strip (n tiles)  |  (auto-scroll) |
    +---------------------------------+----------------+

The composer is deliberately mission-agnostic: it renders whatever a
:class:`~code.apps.demos.models.FrameState` exposes (any number of target rings,
planned paths, pads, robots and transcript lines, including new clarify/user
flows), so it works unchanged for the 4- and 6-robot demos.

Rendering runs on the GPU only if the NVIDIA EGL ICD was pinned before MuJoCo's
first context — the package ``__init__`` calls ``force_nvidia_egl()`` for that,
but a caller that builds a fleet first should call it at its own entrypoint top.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import cv2
import mujoco
import numpy as np

from code.apps.demos import draw, effects, style
from code.apps.demos.layout import Rect, contain_fit, ego_area, regions, tile_rects
from code.apps.demos.models import FrameState
from code.apps.warehouse_demo import bev as bevmod

# ~2 s of video at the 50 Hz control loop -> comm-glow fully decays.
GLOW_DECAY_STEPS: int = 100
_SIM_DT: float = 0.02


class DemoComposer:
    """Renders one demo frame per :meth:`compose` call from a FrameState + viz."""

    def __init__(self, viz, hall_x: float, hall_y: float, *, title: str,
                 description: str, project: str = "G1 Warehouse Fleet",
                 fps: int = 30, sim_dt: float = _SIM_DT,
                 title_card_secs: float = 2.0, title_fade_secs: float = 0.6,
                 glow_decay_steps: int = GLOW_DECAY_STEPS,
                 canvas: Tuple[int, int] = (style.CANVAS_W, style.CANVAS_H)) -> None:
        """Build the composer and its dedicated BEV renderer.

        Args:
            viz: A :class:`~code.fleet.viz.FleetViz` (its ``model``/``data`` are
                read for the BEV render and ``render_ego`` for the tiles).
            hall_x, hall_y: Hall extents (m) for the fit-BEV camera.
            title: Demo title (HUD + panel + title card).
            description: One-line demo description (title card).
            project: Project name (panel subtitle + title card).
            fps: Target playback fps (metadata for the recorder).
            sim_dt: Seconds per control step (for the sim-time clock).
            title_card_secs: Opening full-frame card duration (0 disables it).
            title_fade_secs: Trailing cross-fade duration of the card.
            glow_decay_steps: Comm-glow decay window (control steps).
            canvas: Output ``(width, height)``.
        """
        self.viz = viz
        self.title = title
        self.description = description
        self.project = project
        self.fps = fps
        self.sim_dt = sim_dt
        self.title_card_secs = title_card_secs
        self.title_fade_secs = title_fade_secs
        self.glow_decay_steps = glow_decay_steps
        self.canvas_w, self.canvas_h = canvas

        self.regions: Dict[str, Rect] = regions(self.canvas_w, self.canvas_h)
        self._fovy = float(viz.model.vis.global_.fovy)
        self.bev_cam = bevmod.fit_bev_camera(
            hall_x, hall_y, width=style.BEV_RENDER_W, height=style.BEV_RENDER_H,
            fovy_deg=self._fovy)
        self._bev_mjv = self.bev_cam.to_mjv_camera()
        self._bev_rend: Optional[mujoco.Renderer] = None
        self._card: Optional[np.ndarray] = None

    # -- introspection ----------------------------------------------------
    def tiles(self, n: int) -> List[Rect]:
        """Ego-tile rectangles for ``n`` robots (see :func:`layout.tile_rects`)."""
        return tile_rects(self.regions["strip"], n)

    # -- rendering --------------------------------------------------------
    def _render_bev(self, data) -> np.ndarray:
        """Render the whole-hall BEV as a BGR uint8 frame at the render size."""
        if self._bev_rend is None:
            self._bev_rend = mujoco.Renderer(
                self.viz.model, style.BEV_RENDER_H, style.BEV_RENDER_W)
        self._bev_rend.update_scene(data, self._bev_mjv)
        rgb = self._bev_rend.render()
        return np.ascontiguousarray(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))

    def _paste_bev(self, canvas: np.ndarray, bev_bgr: np.ndarray) -> None:
        """Fit the drawn BEV into its region (centred, dark letterbox frame)."""
        region = self.regions["bev"]
        draw.fill_rect(canvas, region, style.BEV_FRAME_BG)
        m = style.OUTER_MARGIN
        box_w, box_h = region.w - 2 * m, region.h - 2 * m
        ox, oy, w, h = contain_fit(bev_bgr.shape[1], bev_bgr.shape[0], box_w, box_h)
        interp = cv2.INTER_AREA if w <= bev_bgr.shape[1] else cv2.INTER_LINEAR
        fitted = cv2.resize(bev_bgr, (w, h), interpolation=interp)
        x0, y0 = region.x + m + ox, region.y + m + oy
        canvas[y0:y0 + h, x0:x0 + w] = fitted

    def _draw_strip(self, canvas: np.ndarray, state: FrameState,
                    glow: Dict[str, float]) -> None:
        """Draw the always-on ego strip: one live tile per robot."""
        strip = self.regions["strip"]
        draw.fill_rect(canvas, strip, style.STRIP_BG)
        robots = list(state.robots)
        if not robots:
            return
        rects = tile_rects(strip, len(robots))
        for rf, tile in zip(robots, rects):
            ego = ego_area(tile)
            try:
                ego_rgb = self.viz.render_ego(rf.name, rf.yaw)
                ego_bgr = cv2.cvtColor(ego_rgb, cv2.COLOR_RGB2BGR)
            except Exception:
                ego_bgr = np.zeros((ego.h, ego.w, 3), dtype=np.uint8)
            draw.draw_tile(canvas, tile, ego, name=rf.name, ego_bgr=ego_bgr,
                           chip=rf.chip, glow=glow.get(rf.name, 0.0))

    def compose(self, state: FrameState) -> np.ndarray:
        """Render ``state`` into a fresh BGR uint8 ``(H, W, 3)`` canvas."""
        canvas = np.full((self.canvas_h, self.canvas_w, 3), style.CANVAS_BG,
                         dtype=np.uint8)
        sim_time = state.sim_time if state.sim_time is not None \
            else state.step * self.sim_dt

        # Comms panel (right).
        draw.draw_panel(canvas, self.regions["panel"], title=self.title,
                        project=self.project, mission_lines=state.mission_lines,
                        transcript=state.transcript)

        # BEV map (centre-left) + overlays.
        bev_bgr = self._render_bev(self.viz.data)
        draw.draw_bev_overlays(bev_bgr, self.bev_cam, state)
        self._paste_bev(canvas, bev_bgr)
        draw.draw_hud(canvas, self.regions["bev"], self.title, state.phase,
                      sim_time)

        # Ego strip (bottom).
        names = [rf.name for rf in state.robots]
        glow = effects.glow_levels(state.transcript, state.step, names,
                                   self.glow_decay_steps)
        self._draw_strip(canvas, state, glow)

        # Opening title card cross-fade.
        alpha = effects.title_card_alpha(sim_time, self.title_card_secs,
                                         self.title_fade_secs)
        if alpha > 0.0:
            if self._card is None:
                self._card = draw.make_title_card(
                    canvas.shape, title=self.title, description=self.description,
                    project=self.project)
            draw.blend_over(canvas, self._card, alpha)
        return canvas

    def close(self) -> None:
        """Release the BEV renderer (the ego renderer belongs to ``viz``)."""
        if self._bev_rend is not None:
            self._bev_rend.close()
            self._bev_rend = None
