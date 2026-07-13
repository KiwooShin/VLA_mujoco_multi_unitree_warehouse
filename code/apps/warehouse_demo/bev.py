"""bev.py — Fixed wide bird's-eye-view camera + path-overlay math for Phase 1c.

This module owns the *static* warehouse fly-over camera used by the navigation
demo videos: a fixed ``azimuth``/``elevation``/``distance``/``lookat`` framing
that fits the WHOLE 16x12 m hall in one shot (deliberately NOT the single-robot
follow-cam of ``code/apps/fancy/constants.py``, which is tuned to track one
robot at ~17 m / -43.5 deg and only shows a ~14 m patch).

Two concerns live here, both pure enough to unit-test:

* :class:`BevCamera` — the framing plus a MuJoCo-free-camera pinhole projection
  ``world (x, y, z) -> pixel (u, v)`` that matches how ``mujoco.Renderer``
  rasterizes a ``mjCAMERA_FREE`` view (verified empirically against real
  renders). :func:`fit_bev_camera` derives the ``distance`` that frames a hall.
* Overlay drawing (``draw_path``/``draw_polyline``/``draw_marker``/``draw_robot``)
  and :func:`render_bev` which drives an existing
  :class:`code.sim.arena_render.ArenaRenderer`.

Frame convention matches the rest of the project: hall-centered world (x right,
y up in top-down), ``grid``/image rows grow downward.
"""

from __future__ import annotations

import dataclasses
import math
from typing import List, Optional, Sequence, Tuple

import numpy as np

Point = Tuple[float, float]

# MuJoCo's default free-camera vertical field of view (deg). The warehouse model
# never overrides ``model.vis.global_.fovy``; callers should still pass the
# model's value explicitly so projection stays locked to the actual render.
DEFAULT_FOVY_DEG: float = 45.0


@dataclasses.dataclass(frozen=True)
class BevCamera:
    """A fixed MuJoCo free-camera framing plus its pinhole projection.

    Attributes:
        lookat: World point the camera aims at (x, y, z), meters.
        distance: Eye-to-``lookat`` distance, meters.
        azimuth_deg: Azimuth about +z, degrees (MuJoCo convention).
        elevation_deg: Elevation from the horizontal, degrees (negative looks
            down).
        fovy_deg: Vertical field of view, degrees.
        width: Render width in pixels.
        height: Render height in pixels.
    """

    lookat: Tuple[float, float, float]
    distance: float
    azimuth_deg: float
    elevation_deg: float
    fovy_deg: float = DEFAULT_FOVY_DEG
    width: int = 640
    height: int = 480

    def _basis(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Return (eye, forward, right, up) world-frame unit axes + eye pos."""
        az = math.radians(self.azimuth_deg)
        el = math.radians(self.elevation_deg)
        forward = np.array(
            [math.cos(el) * math.cos(az),
             math.cos(el) * math.sin(az),
             math.sin(el)],
            dtype=np.float64,
        )
        eye = np.asarray(self.lookat, dtype=np.float64) - forward * self.distance
        world_up = np.array([0.0, 0.0, 1.0])
        right = np.cross(forward, world_up)
        right /= np.linalg.norm(right)
        up = np.cross(right, forward)
        return eye, forward, right, up

    def project(self, xyz: Sequence[float]) -> Tuple[float, float, float]:
        """Project a world point to pixel coordinates.

        Uses the same pinhole model MuJoCo applies to a ``mjCAMERA_FREE`` view:
        vertical FOV, square pixels, image rows growing downward.

        Args:
            xyz: World point (x, y, z) in meters.

        Returns:
            (u, v, depth): pixel column, pixel row, and camera-frame depth
            (meters, positive in front of the camera).
        """
        eye, forward, right, up = self._basis()
        d = np.asarray(xyz, dtype=np.float64) - eye
        xc = float(np.dot(d, right))
        yc = float(np.dot(d, up))
        zc = float(np.dot(d, forward))
        f = (self.height / 2.0) / math.tan(math.radians(self.fovy_deg) / 2.0)
        if abs(zc) < 1e-9:
            zc = 1e-9 if zc >= 0 else -1e-9
        u = self.width / 2.0 + f * (xc / zc)
        v = self.height / 2.0 - f * (yc / zc)
        return u, v, zc

    def project_xy(self, xy: Point, z: float = 0.0) -> Tuple[int, int]:
        """Project a ground point (x, y, z) to integer pixel (col, row)."""
        u, v, _ = self.project((xy[0], xy[1], z))
        return int(round(u)), int(round(v))

    def to_mjv_camera(self):  # type: ignore[no-untyped-def]
        """Build a configured ``mujoco.MjvCamera`` matching this framing."""
        import mujoco

        cam = mujoco.MjvCamera()
        cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        cam.lookat[:] = list(self.lookat)
        cam.distance = float(self.distance)
        cam.azimuth = float(self.azimuth_deg)
        cam.elevation = float(self.elevation_deg)
        return cam


def fit_bev_camera(
    hall_x: float,
    hall_y: float,
    *,
    width: int = 640,
    height: int = 480,
    fovy_deg: float = DEFAULT_FOVY_DEG,
    azimuth_deg: float = 90.0,
    elevation_deg: float = -78.0,
    margin: float = 1.30,
    lookat_z: float = 0.0,
) -> BevCamera:
    """Derive a static BEV camera that frames the whole hall.

    The ``distance`` is the top-down fit for the larger of the hall's two
    half-extents (accounting for the image aspect ratio), scaled by ``margin``
    to absorb the perimeter walls and the foreshortening introduced by the
    slight downward tilt (``elevation_deg`` < -90 would be exactly top-down).
    Verified empirically: at 16x12 m / 640x480 / fovy 45 deg this yields
    ``distance`` ~= 18.8 m, which keeps all four walls in frame.

    Args:
        hall_x: Full hall extent along x (m).
        hall_y: Full hall extent along y (m).
        width: Render width (px).
        height: Render height (px).
        fovy_deg: Vertical field of view (deg).
        azimuth_deg: Camera azimuth (deg); 90 gives a north-up, east-right map.
        elevation_deg: Camera elevation (deg); negative looks down.
        margin: Multiplicative safety margin on the fitted distance (> 1).
        lookat_z: z of the look-at point (m).

    Returns:
        A :class:`BevCamera` centered on the hall.

    Raises:
        ValueError: If ``margin`` <= 0 or either hall extent is non-positive.
    """
    if margin <= 0.0:
        raise ValueError(f"margin must be > 0, got {margin}")
    if hall_x <= 0.0 or hall_y <= 0.0:
        raise ValueError(f"hall extents must be > 0, got ({hall_x}, {hall_y})")
    aspect = width / height
    half_tan = math.tan(math.radians(fovy_deg) / 2.0)
    need_y = (hall_y / 2.0) / half_tan
    need_x = (hall_x / 2.0) / (half_tan * aspect)
    distance = margin * max(need_x, need_y)
    return BevCamera(
        lookat=(0.0, 0.0, lookat_z),
        distance=distance,
        azimuth_deg=azimuth_deg,
        elevation_deg=elevation_deg,
        fovy_deg=fovy_deg,
        width=width,
        height=height,
    )


# ---------------------------------------------------------------------------
# Overlay drawing (cv2). Kept import-local so the pure math above stays usable
# without OpenCV installed.
# ---------------------------------------------------------------------------
def draw_polyline(
    bgr: np.ndarray,
    cam: BevCamera,
    pts_xy: Sequence[Point],
    color: Tuple[int, int, int],
    *,
    thickness: int = 2,
    z: float = 0.03,
) -> None:
    """Draw a projected world polyline onto a BGR frame in-place."""
    import cv2

    if len(pts_xy) < 2:
        return
    px = [cam.project_xy(p, z=z) for p in pts_xy]
    for a, b in zip(px, px[1:]):
        cv2.line(bgr, a, b, color, thickness, lineType=cv2.LINE_AA)


def draw_path(
    bgr: np.ndarray,
    cam: BevCamera,
    path_xy: Sequence[Point],
    *,
    color: Tuple[int, int, int] = (0, 200, 255),
    thickness: int = 2,
    node_radius: int = 4,
    z: float = 0.03,
) -> None:
    """Draw the planned path polyline plus a dot at each waypoint (in-place)."""
    import cv2

    draw_polyline(bgr, cam, path_xy, color, thickness=thickness, z=z)
    for p in path_xy:
        cv2.circle(bgr, cam.project_xy(p, z=z), node_radius, color, -1,
                   lineType=cv2.LINE_AA)


def draw_marker(
    bgr: np.ndarray,
    cam: BevCamera,
    xy: Point,
    *,
    color: Tuple[int, int, int],
    radius: int = 7,
    z: float = 0.05,
    filled: bool = False,
) -> None:
    """Draw a circular marker at a projected world point (in-place)."""
    import cv2

    cv2.circle(bgr, cam.project_xy(xy, z=z), radius, color,
               -1 if filled else 2, lineType=cv2.LINE_AA)


def draw_robot(
    bgr: np.ndarray,
    cam: BevCamera,
    xy: Point,
    yaw: float,
    *,
    color: Tuple[int, int, int] = (255, 80, 80),
    radius: int = 8,
    heading_len: float = 0.6,
) -> None:
    """Draw the robot as a filled dot with a heading tick (in-place)."""
    import cv2

    c = cam.project_xy(xy, z=0.35)
    cv2.circle(bgr, c, radius, color, -1, lineType=cv2.LINE_AA)
    tip = (xy[0] + heading_len * math.cos(yaw), xy[1] + heading_len * math.sin(yaw))
    cv2.line(bgr, c, cam.project_xy(tip, z=0.35), (255, 255, 255), 2,
             lineType=cv2.LINE_AA)


def render_bev(renderer, data, cam: BevCamera) -> np.ndarray:
    """Render the BEV RGB frame and return it as a BGR uint8 array.

    Args:
        renderer: A live :class:`code.sim.arena_render.ArenaRenderer`.
        data: The ``mujoco.MjData`` to render.
        cam: The static BEV framing.

    Returns:
        A (H, W, 3) uint8 BGR frame (ready for cv2 overlay/writing).
    """
    import cv2

    mjv_cam = getattr(cam, "_mjv_cache", None)
    if mjv_cam is None:
        mjv_cam = cam.to_mjv_camera()
        object.__setattr__(cam, "_mjv_cache", mjv_cam)
    rgb = renderer.render_tp(data, mjv_cam)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def put_hud(bgr: np.ndarray, lines: Sequence[str],
            *, org: Tuple[int, int] = (12, 26),
            color: Tuple[int, int, int] = (255, 255, 255)) -> None:
    """Draw a small stacked text HUD in the top-left corner (in-place)."""
    import cv2

    x, y0 = org
    for i, line in enumerate(lines):
        y = y0 + i * 24
        cv2.putText(bgr, line, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.62,
                    (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(bgr, line, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.62,
                    color, 1, cv2.LINE_AA)


def paste_pip(bgr: np.ndarray, pip_rgb: np.ndarray,
              *, scale: float = 1.0, pad: int = 10) -> None:
    """Paste a picture-in-picture RGB frame into the bottom-right (in-place)."""
    import cv2

    pip = cv2.cvtColor(pip_rgb, cv2.COLOR_RGB2BGR)
    if scale != 1.0:
        pip = cv2.resize(pip, (int(pip.shape[1] * scale), int(pip.shape[0] * scale)))
    ph, pw = pip.shape[:2]
    H, W = bgr.shape[:2]
    y0, x0 = H - ph - pad, W - pw - pad
    if y0 < 0 or x0 < 0:
        return
    cv2.rectangle(bgr, (x0 - 2, y0 - 2), (x0 + pw + 1, y0 + ph + 1),
                  (255, 255, 255), 1)
    bgr[y0:y0 + ph, x0:x0 + pw] = pip
