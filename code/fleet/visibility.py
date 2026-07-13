"""visibility.py — Deterministic geometric visibility oracle for the fleet.

The Phase-4 coordination protocol asks each robot "can you see object O?"
(:meth:`~code.comms.protocol.RobotActions.can_see`). Rather than run the heatmap
detector on a rendered ego frame every step, the fleet answers that question with
a **deterministic geometric oracle**: an object is visible to a robot iff it is

1. within the head camera's horizontal field of view around the robot's yaw
   (derived from the ego-camera constants in :mod:`code.sim.arena_build`),
2. within :data:`MAX_RANGE_M` metres, and
3. in unobstructed line of sight — the 2-D segment from the robot's head to the
   object must not cross any wall/shelf/partition footprint tall enough to
   occlude the sightline.

This oracle stands in for the perception detector and exposes exactly the same
interface (an ``(x, y)`` or ``None``); the real GROUND_NET ego render remains a
Phase-5 overlay option. Because the test is a closed-form function of the shared
``scene_cfg`` wall list and the robots' (exactly known) poses, it is fully
reproducible and unit-testable against hand-built occlusion cases.

Frame convention matches the rest of the project: hall-centered world (x right,
y up in a top-down view); yaw is measured from +x toward +y (rad).
"""

from __future__ import annotations

import dataclasses
import math
from typing import Dict, Optional, Sequence, Tuple

from code.sim.arena_build import CAM_HEAD_Z, EGO_FOVY, EGO_H, EGO_W

XY = Tuple[float, float]

# Maximum detection range of the head camera stand-in (m). Beyond this an object
# is treated as too far to localise reliably, matching the detector's usable band.
MAX_RANGE_M: float = 6.0

# A wall occludes the sightline only if its top rises above this height (m); every
# warehouse wall/shelf/partition is >= 1.8 m so all of them block, but the gate
# keeps the oracle honest for future short props that should not occlude.
_MIN_OCCLUDER_H: float = 0.25

# Nominal object centre height when only an (x, y) is known (m). Objects rest on
# the floor at ~size/2; a small value keeps the sightline low so any real wall
# occludes it.
_DEFAULT_OBJ_Z: float = 0.12

# Nominal object planar half-extent (m) when the object's size is unknown. The
# visibility oracle samples the object's extent (centre + 4 cardinal edge points
# at this radius) so a *partially* visible object — an edge peeking past a wall —
# is labelled visible rather than hidden (a single centre-point segment mislabels
# it, the "through-wall FP" the perception-eval decomposition traced to genuine
# partial visibility). Warehouse objects are ~0.2-0.26 m across, so ~0.12 m is a
# faithful default half-extent; callers that know the object size pass it through.
_DEFAULT_OBJ_RADIUS: float = 0.12


def head_half_fov_rad(fovy_deg: float = EGO_FOVY, w: int = EGO_W,
                      h: int = EGO_H) -> float:
    """Return half the head camera's horizontal field of view (rad).

    The ego camera is specified by a vertical FOV (:data:`code.sim.arena_build.
    EGO_FOVY`) and a pixel aspect ratio; the horizontal FOV follows from the
    standard pinhole relation ``fovx = 2*atan(tan(fovy/2) * w/h)``.

    Args:
        fovy_deg: Vertical field of view (deg).
        w: Ego render width (px).
        h: Ego render height (px).

    Returns:
        Half the horizontal field of view in radians (the yaw tolerance).
    """
    fovy = math.radians(fovy_deg)
    fovx = 2.0 * math.atan(math.tan(fovy / 2.0) * float(w) / float(h))
    return fovx / 2.0


def wrap_angle(a: float) -> float:
    """Wrap an angle to ``(-pi, pi]``."""
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def _segment_intersects_aabb(x0: float, y0: float, x1: float, y1: float,
                             hx: float, hy: float) -> bool:
    """Liang-Barsky test: does segment ``(x0,y0)->(x1,y1)`` meet box +-hx/+-hy?

    The box is axis-aligned and centred at the origin with half-extents
    ``hx``/``hy``. Returns True if any portion of the segment lies inside or on
    the box.
    """
    dx = x1 - x0
    dy = y1 - y0
    t0, t1 = 0.0, 1.0
    for p, q in ((-dx, x0 + hx), (dx, hx - x0), (-dy, y0 + hy), (dy, hy - y0)):
        if abs(p) < 1e-12:
            if q < 0.0:
                return False  # segment parallel to and outside this slab
            continue
        r = q / p
        if p < 0.0:
            if r > t1:
                return False
            if r > t0:
                t0 = r
        else:
            if r < t0:
                return False
            if r < t1:
                t1 = r
    return t0 <= t1


def _segment_intersects_wall(p0: XY, p1: XY, wall: Dict[str, float]) -> bool:
    """True if the 2-D segment crosses a (possibly yawed) wall footprint."""
    cx = float(wall["cx"])
    cy = float(wall["cy"])
    yaw = float(wall.get("yaw", 0.0))
    c, s = math.cos(yaw), math.sin(yaw)
    # Transform both endpoints into the wall's local (axis-aligned) frame.
    lx0 = (p0[0] - cx) * c + (p0[1] - cy) * s
    ly0 = -(p0[0] - cx) * s + (p0[1] - cy) * c
    lx1 = (p1[0] - cx) * c + (p1[1] - cy) * s
    ly1 = -(p1[0] - cx) * s + (p1[1] - cy) * c
    return _segment_intersects_aabb(lx0, ly0, lx1, ly1,
                                    float(wall["half_x"]), float(wall["half_y"]))


def _segment_los_clear(head_xy: XY, obj_xy: XY,
                       walls: Sequence[Dict[str, float]], *,
                       head_z: float, obj_z: float) -> bool:
    """Whether no occluding wall lies on the single head->point 2-D segment.

    A wall blocks the sightline when its footprint intersects the 2-D
    head->point segment AND its top rises above the lower of the two endpoint
    heights (so a wall shorter than both endpoints — e.g. a low crate — never
    occludes). Objects and robots are never occluders (walls only).
    """
    sightline_floor = min(head_z, obj_z)
    for wall in walls:
        if float(wall.get("height", 2.5)) <= max(_MIN_OCCLUDER_H, sightline_floor):
            continue
        if _segment_intersects_wall(head_xy, obj_xy, wall):
            return False
    return True


def line_of_sight_clear(head_xy: XY, obj_xy: XY,
                        walls: Sequence[Dict[str, float]], *,
                        head_z: float, obj_z: float,
                        obj_radius: float = _DEFAULT_OBJ_RADIUS) -> bool:
    """Return whether the object is at least *partially* in clear line of sight.

    Samples the object's planar extent: the centre plus four cardinal edge points
    at ``obj_radius`` (``(+r,0) (-r,0) (0,+r) (0,-r)``). The object is visible if
    the centre **or any** edge sample has an unobstructed sightline. A single
    centre-point segment (``obj_radius <= 0``) mislabels a partially visible
    object — an edge sticking past a wall — as hidden; the extent sampling fixes
    that (the perception-eval "through-wall FP" decomposition showed those frames
    were genuine partial visibility, not detector hallucinations).

    Args:
        head_xy: Camera/head world (x, y).
        obj_xy: Object world (x, y).
        walls: Serialized wall dicts (``cx``/``cy``/``half_x``/``half_y``/``yaw``/
            ``height``), e.g. ``scene_cfg["walls"]``.
        head_z: Camera height (m).
        obj_z: Object centre height (m).
        obj_radius: Object planar half-extent to sample (m); ``<= 0`` reduces to
            the historical centre-only test.

    Returns:
        True if the line of sight to the centre or any sampled edge is unobstructed.
    """
    if _segment_los_clear(head_xy, obj_xy, walls, head_z=head_z, obj_z=obj_z):
        return True
    r = float(obj_radius)
    if r <= 0.0:
        return False
    ox, oy = float(obj_xy[0]), float(obj_xy[1])
    for sample in ((ox + r, oy), (ox - r, oy), (ox, oy + r), (ox, oy - r)):
        if _segment_los_clear(head_xy, sample, walls, head_z=head_z, obj_z=obj_z):
            return True
    return False


@dataclasses.dataclass(frozen=True)
class VisibilityConfig:
    """Tunable geometry of the visibility oracle.

    Attributes:
        max_range: Maximum detection range (m).
        half_fov: Half the head camera horizontal FOV (rad); yaw tolerance.
        head_z_offset: Camera height above the pelvis (m).
    """

    max_range: float = MAX_RANGE_M
    half_fov: float = dataclasses.field(default_factory=head_half_fov_rad)
    head_z_offset: float = CAM_HEAD_Z

    def head_z(self, base_height: float) -> float:
        """Camera world height for a given pelvis height (m)."""
        return float(base_height) + self.head_z_offset


def is_object_visible(robot_xy: XY, robot_yaw: float, base_height: float,
                      obj_xy: XY, walls: Sequence[Dict[str, float]], *,
                      obj_z: float = _DEFAULT_OBJ_Z,
                      obj_radius: float = _DEFAULT_OBJ_RADIUS,
                      cfg: Optional[VisibilityConfig] = None) -> bool:
    """Return whether a robot can see an object under the geometric oracle.

    Applies the three gates in cheap-first order: range, then horizontal FOV,
    then line of sight. Range and FOV are tested on the object *centre* (its
    localisation point); the line-of-sight test samples the object's planar
    extent so a partially visible object (an edge past a wall) counts as visible.

    Args:
        robot_xy: Robot pelvis world (x, y).
        robot_yaw: Robot yaw (rad).
        base_height: Robot pelvis height (m), for the camera height.
        obj_xy: Object world (x, y).
        walls: Serialized wall dicts (``scene_cfg["walls"]``).
        obj_z: Object centre height (m).
        obj_radius: Object planar half-extent for the sampled LOS (m).
        cfg: Oracle geometry (defaults to :class:`VisibilityConfig`).

    Returns:
        True iff the object centre is in range and FOV and at least part of the
        object is in clear line of sight.
    """
    cfg = cfg or VisibilityConfig()
    dx = obj_xy[0] - robot_xy[0]
    dy = obj_xy[1] - robot_xy[1]
    dist = math.hypot(dx, dy)
    if dist > cfg.max_range:
        return False
    if dist > 1e-6:  # a coincident object is trivially "seen"
        bearing = math.atan2(dy, dx)
        if abs(wrap_angle(bearing - robot_yaw)) > cfg.half_fov:
            return False
    head_z = cfg.head_z(base_height)
    return line_of_sight_clear(robot_xy, obj_xy, walls,
                               head_z=head_z, obj_z=obj_z, obj_radius=obj_radius)


def first_visible(robot_xy: XY, robot_yaw: float, base_height: float,
                  objects: Sequence[Dict[str, float]],
                  walls: Sequence[Dict[str, float]], *,
                  cfg: Optional[VisibilityConfig] = None) -> Optional[int]:
    """Return the index of the first (lowest-index) visible object, or ``None``.

    Args:
        robot_xy: Robot pelvis world (x, y).
        robot_yaw: Robot yaw (rad).
        base_height: Robot pelvis height (m).
        objects: Object dicts with ``x``/``y``/``size`` keys (current positions).
        walls: Serialized wall dicts.
        cfg: Oracle geometry.

    Returns:
        The index into ``objects`` of the first visible object, or ``None``.
    """
    for i, obj in enumerate(objects):
        obj_z = max(_DEFAULT_OBJ_Z, float(obj.get("size", 0.2)) / 2.0)
        obj_radius = max(_DEFAULT_OBJ_RADIUS, float(obj.get("size", 0.2)) / 2.0)
        if is_object_visible(robot_xy, robot_yaw, base_height,
                             (float(obj["x"]), float(obj["y"])), walls,
                             obj_z=obj_z, obj_radius=obj_radius, cfg=cfg):
            return i
    return None
