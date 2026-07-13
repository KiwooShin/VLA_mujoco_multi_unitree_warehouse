"""carry.py — Mock pickup / kinematic carry / delivery-pad release.

There is no grasp policy (docs/multi_plan.md sec 6, USER-CONFIRMED): "picking up"
an object means, once the robot's pelvis is within :data:`PICKUP_RADIUS_M` of it,
snapping the object onto that robot's right hand and re-posing it to the hand's
world pose **every control step** while carried, in BOTH the robot's own physics
model and the shared viz model that the video renders from. The object's
collisions are disabled while carried (a mock grasp must not perturb the walk
policy or block planning), and it is released — placed flat on the floor — at the
delivery-pad centre when the carrier arrives.

Carry body: the most distal *stable* right-arm body, ``right_wrist_yaw_link`` (the
palm root). The finger links (``right_hand_*``) are more distal but tiny and
grasp-driven, so the wrist yaw link is the reliable hand anchor; a small
robot-frame offset lifts the object so it reads as held.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import mujoco
import numpy as np

from code.fleet.fleet import Fleet
from code.fleet.viz import _prefix

XY = Tuple[float, float]

# Pelvis-to-object distance within which a mock pickup succeeds (m).
PICKUP_RADIUS_M: float = 0.6
# Pelvis-to-destination distance within which a carried object is released (m).
DELIVER_RADIUS_M: float = 0.5
# Right-arm body the object is anchored to while carried.
HAND_BODY: str = "right_wrist_yaw_link"
# Object offset from the hand body, expressed in the robot's yaw frame (m):
# a touch forward of the wrist and lifted so it rests visibly in the hand.
_HAND_OFFSET: Tuple[float, float, float] = (0.08, 0.0, 0.04)


def _object_geom_ids(model: mujoco.MjModel, index: int,
                     prefix: str = "") -> List[int]:
    """Return the geom ids of object ``index`` (body ``obj_i`` + any ``obj_i_*``)."""
    ids: List[int] = []
    base = f"{prefix}obj_{index}"
    for gid in range(model.ngeom):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, gid)
        if name and (name == base or name.startswith(base + "_")):
            ids.append(gid)
    return ids


class CarryManager:
    """Owns mock pickup, per-step kinematic carry and delivery-pad release.

    One instance per mission, shared across robots. It mutates object geom poses
    in the viz model (for rendering) and in each carrying robot's physics model
    (for consistency), and keeps ``scene_cfg["objects"]`` positions in sync so the
    visibility oracle and planner track the carried object.
    """

    def __init__(self, fleet: Fleet, scene_cfg: dict, *,
                 hand_body: str = HAND_BODY,
                 pickup_radius: float = PICKUP_RADIUS_M,
                 deliver_radius: float = DELIVER_RADIUS_M) -> None:
        """Precompute geom ids and bind to the fleet + scene.

        Args:
            fleet: The co-simulation fleet (units + shared viz).
            scene_cfg: The shared warehouse scene_cfg (its ``objects`` list is
                the single source of truth for object positions).
            hand_body: Right-arm body name the object anchors to.
            pickup_radius: Mock-pickup success radius (m).
            deliver_radius: Release radius at the destination (m).

        Raises:
            ValueError: If the fleet has no viz model (carry needs it to render).
        """
        if fleet.viz is None:
            raise ValueError("CarryManager requires a fleet built with build_viz=True")
        self._fleet = fleet
        self._viz = fleet.viz
        self._cfg = scene_cfg
        self._objects: List[dict] = scene_cfg["objects"]
        self._hand_body = hand_body
        self._pickup_radius = float(pickup_radius)
        self._deliver_radius = float(deliver_radius)

        # Object index -> viz geom ids (rendered) and per-robot physics geom ids,
        # plus each sub-geom's build-time z above the object's base geom (so a
        # cone's tip stays above its base when the object is re-posed).
        self._viz_geoms: Dict[int, List[int]] = {
            i: _object_geom_ids(self._viz.model, i) for i in range(len(self._objects))
        }
        self._viz_zrel = self._z_offsets(self._viz.model, self._viz_geoms)
        self._phys_geoms: Dict[str, Dict[int, List[int]]] = {}
        self._phys_zrel: Dict[str, Dict[int, Dict[int, float]]] = {}
        for name, unit in fleet.units.items():
            model = unit.teacher.model
            geoms = {i: _object_geom_ids(model, i) for i in range(len(self._objects))}
            self._phys_geoms[name] = geoms
            self._phys_zrel[name] = self._z_offsets(model, geoms)

        # The viz model is a display-only, kinematic model (never stepped): give
        # its object geoms no collisions so re-posing a carried object onto a
        # robot's arm cannot inject a degenerate contact into ``mj_forward``.
        for gids in self._viz_geoms.values():
            for gid in gids:
                self._viz.model.geom_contype[gid] = 0
                self._viz.model.geom_conaffinity[gid] = 0

        self._carry: Dict[str, int] = {}          # robot -> carried object index
        self._dest: Dict[str, XY] = {}             # robot -> delivery destination
        self.released: Dict[str, int] = {}         # robot -> delivered object index

    # -- queries ----------------------------------------------------------
    def carrying(self, robot: str) -> bool:
        """Whether ``robot`` is currently carrying an object."""
        return robot in self._carry

    def carried_index(self, robot: str) -> Optional[int]:
        """The object index ``robot`` carries, or ``None``."""
        return self._carry.get(robot)

    def is_carried(self, index: int) -> bool:
        """Whether object ``index`` is currently carried by any robot."""
        return index in self._carry.values()

    # -- pickup / deliver -------------------------------------------------
    def pickup(self, robot: str, index: int) -> bool:
        """Attempt a mock pickup of object ``index`` by ``robot``.

        Succeeds when the robot's pelvis is within the pickup radius; the object
        is anchored to the hand and its collisions disabled in the robot's
        physics model so it neither perturbs the walk nor blocks planning.

        Args:
            robot: Carrying robot callsign.
            index: Object index to pick up.

        Returns:
            True if the pickup succeeded (pelvis in range).
        """
        unit = self._fleet.units[robot]
        ox, oy = float(self._objects[index]["x"]), float(self._objects[index]["y"])
        px, py = unit.xy
        if math.hypot(ox - px, oy - py) > self._pickup_radius:
            return False
        model = unit.teacher.model
        for gid in self._phys_geoms[robot][index]:
            model.geom_contype[gid] = 0
            model.geom_conaffinity[gid] = 0
        self._carry[robot] = index
        return True

    def set_destination(self, robot: str, xy: XY) -> None:
        """Record where ``robot`` will release its carried object (pad centre)."""
        self._dest[robot] = (float(xy[0]), float(xy[1]))

    def release(self, robot: str) -> Optional[int]:
        """Place the carried object flat on the floor at its destination.

        Args:
            robot: The carrying robot.

        Returns:
            The released object index, or ``None`` if the robot carried nothing.
        """
        index = self._carry.pop(robot, None)
        if index is None:
            return None
        dest = self._dest.get(robot)
        if dest is None:
            dest = self._fleet.units[robot].xy
        self._settle(index, robot, (float(dest[0]), float(dest[1])))
        return index

    def drop_here(self, robot: str) -> Optional[int]:
        """Release the carried object flat on the floor at its *current* spot.

        Used when the carrier can no longer deliver (it has fallen): the object
        is dropped where it is right now rather than welded to the wrist forever,
        so the owner's task can fail cleanly instead of hanging.

        Args:
            robot: The carrying robot.

        Returns:
            The released object index, or ``None`` if the robot carried nothing.
        """
        index = self._carry.pop(robot, None)
        if index is None:
            return None
        here = (float(self._objects[index]["x"]), float(self._objects[index]["y"]))
        self._settle(index, robot, here)
        return index

    def _settle(self, index: int, robot: str, xy: XY) -> None:
        """Rest object ``index`` flat on the floor at ``xy`` and record it there."""
        size = float(self._objects[index].get("size", 0.2))
        self._place(index, robot, (xy[0], xy[1], size / 2.0))
        self._objects[index]["x"] = float(xy[0])
        self._objects[index]["y"] = float(xy[1])
        self.released[robot] = index
        self._dest.pop(robot, None)

    # -- per-step update --------------------------------------------------
    def update(self) -> None:
        """Re-pose every carried object to its carrier's hand; release on arrival.

        Call once per control step *after* :meth:`Fleet.step_all` (so the viz
        model is already synced). Releases a carried object when its carrier has
        reached the recorded destination.
        """
        for robot in list(self._carry):
            index = self._carry[robot]
            unit = self._fleet.units[robot]
            if getattr(unit, "fell", False):
                # A fallen carrier can never reach the pad: drop the object where
                # it is (not welded to the wrist) so the fetch fails cleanly.
                self.drop_here(robot)
                continue
            dest = self._dest.get(robot)
            if dest is not None and self._at_destination(unit, dest):
                self.release(robot)
                continue
            hand = self._hand_world_pose(robot)
            self._place(index, robot, hand)
            self._objects[index]["x"] = float(hand[0])
            self._objects[index]["y"] = float(hand[1])
        # Refresh viz kinematics so the moved geoms render at their new pose.
        mujoco.mj_forward(self._viz.model, self._viz.data)

    def _at_destination(self, unit, dest: XY) -> bool:
        """Whether a carrier has arrived at its delivery destination."""
        px, py = unit.xy
        return unit.done or math.hypot(dest[0] - px, dest[1] - py) <= self._deliver_radius

    def _hand_world_pose(self, robot: str) -> Tuple[float, float, float]:
        """World position for the carried object, from the viz hand body + offset."""
        body = self._viz.data.body(f"{_prefix(robot)}{self._hand_body}")
        hx, hy, hz = (float(body.xpos[0]), float(body.xpos[1]), float(body.xpos[2]))
        yaw = self._fleet.units[robot].yaw
        ox, oy, oz = _HAND_OFFSET
        wx = hx + ox * math.cos(yaw) - oy * math.sin(yaw)
        wy = hy + ox * math.sin(yaw) + oy * math.cos(yaw)
        return (wx, wy, hz + oz)

    def _place(self, index: int, robot: str,
               pos: Tuple[float, float, float]) -> None:
        """Set object ``index``'s geom position in the viz + carrier physics models."""
        p = np.asarray(pos, dtype=float)
        for gid in self._viz_geoms.get(index, []):
            self._viz.model.geom_pos[gid][:2] = p[:2]
            self._viz.model.geom_pos[gid][2] = p[2] + self._viz_zrel[index][gid]
        model = self._fleet.units[robot].teacher.model
        zrel = self._phys_zrel[robot][index]
        for gid in self._phys_geoms[robot].get(index, []):
            model.geom_pos[gid][:2] = p[:2]
            model.geom_pos[gid][2] = p[2] + zrel[gid]

    @staticmethod
    def _z_offsets(model: mujoco.MjModel,
                   geoms: Dict[int, List[int]]) -> Dict[int, Dict[int, float]]:
        """Each object's sub-geom z minus its base geom z (preserves cone tips)."""
        out: Dict[int, Dict[int, float]] = {}
        for index, gids in geoms.items():
            if not gids:
                out[index] = {}
                continue
            base_z = float(model.geom_pos[gids[0]][2])
            out[index] = {gid: float(model.geom_pos[gid][2]) - base_z for gid in gids}
        return out
