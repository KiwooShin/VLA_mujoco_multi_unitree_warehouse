"""viz.py — Shared kinematic visualization model for the multi-robot fleet.

The federated-physics design (docs/multi_plan.md sec 3) keeps each robot alone in
its own physics ``MjModel`` so the single-robot baseline stays valid. To *see*
all robots together — for the fleet BEV video and, critically, for
cross-visibility (robot A's camera seeing robot B) — this module builds ONE extra
``MjModel``: the warehouse plus every G1 attached under a ``"<name>_"`` prefix via
``mujoco.MjSpec.attach``. That model is NEVER stepped. Each frame the fleet copies
every robot's physics qpos (root free joint + all joint angles) into that robot's
contiguous prefixed qpos slice and calls ``mj_forward`` to refresh kinematics.

Per-robot qpos addresses are computed ONCE from the prefixed free-joint name and
each robot occupies a contiguous ``robot_nq``-length block (verified: the attach
order makes the blocks contiguous and equal to the single-robot physics ``nq``).
Each robot's torso is tinted with its callsign accent colour so viewers can track
identity in the shared render.

Public API
----------
ACCENT_RGBA — callsign -> torso accent colour.
build_viz_model(scene_cfg, callsigns) -> (MjModel, {name: qpos_addr}).
FleetViz — owns the viz model + data + renderers; ``sync``/``render_bev``/``render_ego``.
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional, Sequence, Tuple

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import mujoco
import numpy as np

from code.apps.warehouse_demo.bev import BevCamera
from code.sim.arena_build import (
    EGO_H,
    EGO_W,
    EGO_FOVY,
    G1_XML,
    GROUNDING_W,
    PROXIMITY_W,
    TP_H,
    TP_W,
)
from code.sim.arena_cameras import _set_ego_cam, get_ego_intrinsics
from code.warehouse.arena import (
    _FLOOR_RGBA,
    _add_object,
    _add_overhead_lights,
    _add_wall,
    _add_zone,
)

# Per-callsign torso accent colours (docs/multi_plan.md sec 3): Alpha red,
# Bravo blue, Charlie yellow, Delta purple. Opaque so the tint reads on video.
ACCENT_RGBA: Dict[str, Tuple[float, float, float, float]] = {
    "Alpha": (0.88, 0.12, 0.12, 1.0),
    "Bravo": (0.16, 0.34, 0.90, 1.0),
    "Charlie": (0.94, 0.80, 0.10, 1.0),
    "Delta": (0.62, 0.24, 0.82, 1.0),
}
_DEFAULT_ACCENT: Tuple[float, float, float, float] = (0.30, 0.85, 0.55, 1.0)

# BEV render size for the whole-hall fleet view (<= model offscreen buffer).
BEV_W: int = 960
BEV_H: int = 720


def _prefix(name: str) -> str:
    """Return the attach prefix for a callsign (``"Alpha"`` -> ``"alpha_"``)."""
    return f"{name.lower()}_"


def _tint_and_strip(child: mujoco.MjSpec, accent: Sequence[float]) -> None:
    """Tint the child robot's torso and strip its floor/lights before attach.

    The G1 XML ships its own ground plane and a directional light; attaching N
    robots would stamp N coincident floors and N extra lights, so both are
    deleted from every child. The torso mesh's visual (group-1) geoms are recolored
    to the callsign accent and named ``accent_torso_{i}`` so the tint is testable.

    Args:
        child: A freshly loaded G1 :class:`mujoco.MjSpec` (mutated in place).
        accent: Torso accent RGBA.
    """
    try:
        floor = child.geom("floor")
        if floor is not None:
            child.delete(floor)
    except (KeyError, ValueError):
        pass
    for light in list(child.lights):
        child.delete(light)

    torso = child.body("torso_link")
    n = 0
    for geom in torso.geoms:
        # The gray torso mesh (visual group 1) is the large chest surface.
        if geom.group == 1 and float(geom.rgba[0]) > 0.5:
            geom.rgba = list(accent)
            geom.name = f"accent_torso_{n}"
            n += 1


def _warehouse_base_spec(scene_cfg: dict) -> mujoco.MjSpec:
    """Build the warehouse-only base spec (floor + walls + objects + zones).

    Reuses the same wall/object/zone/light helpers ``build_warehouse_arena`` uses
    (imported from ``code.warehouse.arena``) so the viz geometry can never skew
    from the physics geometry — single source of truth (docs/multi_plan.md sec 2).

    Args:
        scene_cfg: Warehouse scene_cfg (``walls``, ``objects``, ``zones``).

    Returns:
        A base :class:`mujoco.MjSpec` with no robot yet attached.
    """
    spec = mujoco.MjSpec()
    try:
        spec.visual.global_.offwidth = max(EGO_W, GROUNDING_W, PROXIMITY_W, TP_W, BEV_W)
        spec.visual.global_.offheight = max(EGO_H, TP_H, BEV_H)
    except Exception:
        pass

    wb = spec.worldbody
    floor = wb.add_geom()
    floor.type = mujoco.mjtGeom.mjGEOM_PLANE
    floor.size = [0.0, 0.0, 0.05]
    floor.rgba = list(_FLOOR_RGBA)
    floor.name = "floor"

    for wall in scene_cfg.get("walls", []):
        _add_wall(wb, wall)
    for i, obj in enumerate(scene_cfg.get("objects", [])):
        _add_object(wb, i, obj)
    for zone in scene_cfg.get("zones", []):
        _add_zone(wb, zone)

    try:
        spec.visual.headlight.ambient = [0.5, 0.5, 0.5]
        spec.visual.headlight.diffuse = [0.5, 0.5, 0.5]
        spec.visual.headlight.specular = [0.2, 0.2, 0.2]
    except Exception:
        pass
    _add_overhead_lights(wb, 8.0, 6.0)
    return spec


def build_viz_model(
    scene_cfg: dict, callsigns: Sequence[str],
) -> Tuple[mujoco.MjModel, Dict[str, int]]:
    """Compile the shared viz model: warehouse + N prefixed, tinted G1 robots.

    Args:
        scene_cfg: Warehouse scene_cfg.
        callsigns: Robot callsigns to attach (order fixes the qpos block order).

    Returns:
        (model, qpos_addr) where ``qpos_addr[name]`` is the qpos index of that
        robot's free joint (start of its contiguous per-robot block).

    Raises:
        ValueError: If ``callsigns`` is empty.
    """
    if not callsigns:
        raise ValueError("callsigns must be non-empty")

    base = _warehouse_base_spec(scene_cfg)
    for name in callsigns:
        child = mujoco.MjSpec.from_file(G1_XML)
        _tint_and_strip(child, ACCENT_RGBA.get(name, _DEFAULT_ACCENT))
        frame = base.worldbody.add_frame()
        base.attach(child, prefix=_prefix(name), frame=frame)

    model = base.compile()
    qpos_addr = {
        name: int(model.joint(f"{_prefix(name)}floating_base_joint").qposadr[0])
        for name in callsigns
    }
    return model, qpos_addr


class FleetViz:
    """Owns the shared viz model, its data and the fleet renderers.

    Never call ``mj_step`` on this model. :meth:`sync` writes every robot's
    physics qpos into its prefixed slice and refreshes kinematics; :meth:`render_bev`
    and :meth:`render_ego` then draw the whole hall / one robot's ego view.
    """

    def __init__(self, scene_cfg: dict, callsigns: Sequence[str],
                 *, ego_w: int = EGO_W, ego_h: int = EGO_H,
                 bev_w: int = BEV_W, bev_h: int = BEV_H) -> None:
        """Build the viz model and allocate its data.

        Args:
            scene_cfg: Warehouse scene_cfg.
            callsigns: Robot callsigns (order fixes qpos block order).
            ego_w: Ego render width for cross-visibility renders.
            ego_h: Ego render height.
            bev_w: BEV render width.
            bev_h: BEV render height.
        """
        self.callsigns: List[str] = list(callsigns)
        self.model, self.qpos_addr = build_viz_model(scene_cfg, callsigns)
        self.data = mujoco.MjData(self.model)
        n = len(self.callsigns)
        if self.model.nq % n != 0:
            raise ValueError(
                f"viz nq {self.model.nq} not divisible by {n} robots")
        self.robot_nq: int = self.model.nq // n
        mujoco.mj_forward(self.model, self.data)

        self._ego_w, self._ego_h = ego_w, ego_h
        self._bev_w, self._bev_h = bev_w, bev_h
        self._ego_rend: Optional[mujoco.Renderer] = None
        self._bev_rend: Optional[mujoco.Renderer] = None

    # ---- Kinematic sync ----
    def sync(self, poses: Dict[str, np.ndarray]) -> None:
        """Copy each robot's physics qpos into its prefixed viz slice.

        Args:
            poses: callsign -> physics qpos array of length :attr:`robot_nq`
                (root free joint + all joint angles, same joint order as the
                single-robot physics model).

        Raises:
            KeyError: If a callsign has no viz slice.
            ValueError: If a qpos array is the wrong length.
        """
        for name, qpos in poses.items():
            a0 = self.qpos_addr[name]
            q = np.asarray(qpos, dtype=self.data.qpos.dtype)
            if q.shape[0] != self.robot_nq:
                raise ValueError(
                    f"{name} qpos len {q.shape[0]} != robot_nq {self.robot_nq}")
            self.data.qpos[a0:a0 + self.robot_nq] = q
        mujoco.mj_forward(self.model, self.data)

    def pelvis_xpos(self, name: str) -> np.ndarray:
        """World (x, y, z) of ``name``'s pelvis in the viz model (post-sync)."""
        return self.data.body(f"{_prefix(name)}pelvis").xpos.copy()

    # ---- Rendering ----
    def render_bev(self, cam: BevCamera) -> np.ndarray:
        """Render the whole-hall BEV RGB frame (uint8 H x W x 3)."""
        if self._bev_rend is None:
            self._bev_rend = mujoco.Renderer(self.model, self._bev_h, self._bev_w)
        mjv = cam.to_mjv_camera()
        self._bev_rend.update_scene(self.data, mjv)
        return self._bev_rend.render().copy()

    def render_ego(self, name: str, yaw: float) -> np.ndarray:
        """Render ``name``'s head ego camera against the shared viz scene.

        This is the cross-visibility primitive: the returned frame shows whatever
        the named robot's forward camera sees — including the OTHER robots.

        Args:
            name: Callsign whose ego view to render.
            yaw: That robot's yaw (rad).

        Returns:
            The ego RGB frame (uint8 H x W x 3).
        """
        if self._ego_rend is None:
            self._ego_rend = mujoco.Renderer(self.model, self._ego_h, self._ego_w)
        pelvis = self.pelvis_xpos(name)
        cam = mujoco.MjvCamera()
        cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        _set_ego_cam(cam, pelvis, yaw)
        self._ego_rend.update_scene(self.data, cam)
        return self._ego_rend.render().copy()

    def ego_intrinsics(self) -> dict:
        """Pinhole intrinsics for the ego renders."""
        return get_ego_intrinsics(self._ego_w, self._ego_h, EGO_FOVY)

    def close(self) -> None:
        """Release the underlying EGL renderers."""
        if self._ego_rend is not None:
            self._ego_rend.close()
            self._ego_rend = None
        if self._bev_rend is not None:
            self._bev_rend.close()
            self._bev_rend = None
