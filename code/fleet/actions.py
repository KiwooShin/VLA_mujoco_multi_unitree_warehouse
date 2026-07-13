"""actions.py — Bridge the comms protocol onto the fleet's world.

:class:`FleetRobotActions` is the concrete
:class:`~code.comms.protocol.RobotActions` the Phase-4 mission wires under each
robot's :class:`~code.comms.protocol.RobotProtocol`. It translates the seven
narrow protocol calls into real behaviour:

* ``can_see`` -> the geometric :mod:`~code.fleet.visibility` oracle against the
  robot's (exactly known) pose and the shared wall list;
* ``goto`` / ``deliver`` -> plan an A* path with the robot's
  :class:`~code.fleet.robot_unit.RobotUnit`, and ``arrived`` -> that unit's
  arrival state;
* ``start_search`` / ``abort_search`` -> the region
  :class:`~code.fleet.search.SearchController`;
* ``pickup`` / ``deliver`` -> the mock :class:`~code.fleet.carry.CarryManager`.

The bridge holds no MuJoCo or planner logic of its own — it is deliberately thin
so it can be unit-tested with a scripted fake unit / carry / search (see
``code/fleet/tests/test_actions.py``).
"""

from __future__ import annotations

from typing import List, Optional, Protocol, Sequence, Tuple

from code.comms.messages import ObjectQuery
from code.comms.protocol import RobotActions
from code.fleet.search import region_name_for_xy
from code.fleet.visibility import VisibilityConfig, is_object_visible
from code.warehouse.layout import WarehouseLayout

XY = Tuple[float, float]


class _UnitLike(Protocol):
    """The slice of :class:`~code.fleet.robot_unit.RobotUnit` the bridge needs."""

    @property
    def xy(self) -> XY: ...
    @property
    def yaw(self) -> float: ...
    @property
    def base_height(self) -> float: ...
    @property
    def done(self) -> bool: ...
    @property
    def fell(self) -> bool: ...
    def assign_goal(self, goal_xy: XY) -> bool: ...
    def halt(self) -> None: ...


class _SearchLike(Protocol):
    def start(self, scene_cfg: dict, region: str) -> bool: ...
    def stop(self) -> None: ...


class _CarryLike(Protocol):
    def pickup(self, robot: str, index: int) -> bool: ...
    def set_destination(self, robot: str, xy: XY) -> None: ...
    def is_carried(self, index: int) -> bool: ...


def _match_indices(objects: Sequence[dict], query: ObjectQuery) -> List[int]:
    """Return indices of scene objects matching a colour/shape query."""
    return [i for i, obj in enumerate(objects) if query.matches(obj)]


class FleetRobotActions(RobotActions):
    """Wire one robot's protocol onto its unit, visibility, search and carry."""

    def __init__(self, callsign: str, unit: _UnitLike, scene_cfg: dict,
                 search_ctrl: _SearchLike, carry: _CarryLike, *,
                 vis_cfg: Optional[VisibilityConfig] = None,
                 perception: Optional[object] = None,
                 confirm_range_m: float = 7.0,
                 layout: Optional[WarehouseLayout] = None) -> None:
        """Bind the bridge to one robot's world objects.

        Args:
            callsign: This robot's name (for carry ownership).
            unit: The robot's navigator (``RobotUnit``-like).
            scene_cfg: Shared warehouse scene_cfg (``walls`` + ``objects``; the
                object positions are updated in place by carry).
            search_ctrl: This robot's :class:`~code.fleet.search.SearchController`.
            carry: The shared :class:`~code.fleet.carry.CarryManager`.
            vis_cfg: Visibility-oracle geometry (defaults used if None).
            perception: Optional :class:`~code.fleet.perception_bridge.RobotPerception`.
                When supplied (``perception_mode="groundnet"``), a matching
                oracle-visible object within ``confirm_range_m`` is passed to the
                learned detector to CONFIRM; the DETECTOR's world-xy estimate is
                returned when it confirms, else the oracle's xy (fallback). When
                ``None`` (default), ``can_see`` is the pure geometric oracle.
            confirm_range_m: Max range (m) at which to run the detector confirmer
                (mirrors ``perception_bridge.CONFIRM_RANGE_M``).
            layout: The active warehouse layout, so :meth:`report_origin` can name
                the room this robot is currently standing in (F3). ``None`` falls
                back to the region-less "the area".
        """
        self.callsign = callsign
        self._unit = unit
        self._cfg = scene_cfg
        self._search = search_ctrl
        self._carry = carry
        self._vis = vis_cfg or VisibilityConfig()
        self._perception = perception
        self._confirm_range_m = float(confirm_range_m)
        self._layout = layout
        self.last_plan_ok: Optional[bool] = None
        # Provenance of the most recent successful can_see (for evals/video):
        # "detector" (GROUND_NET confirmed), "oracle_fallback" (oracle-visible but
        # the detector missed), or "oracle" (oracle mode / out of confirm range).
        self.last_see_source: Optional[str] = None

    # -- perception -------------------------------------------------------
    def can_see(self, query: ObjectQuery) -> Optional[XY]:
        """Return the world (x, y) of a matching object visible now, else None.

        The geometric oracle is always the visibility GATE (physics truth of wall
        occlusion). In groundnet mode the learned detector then CONFIRMS the
        oracle-visible object and, on success, refines the reported location to
        its own world-xy estimate; a detector miss falls back to the oracle xy.
        """
        walls = self._cfg.get("walls", [])
        objects = self._cfg["objects"]
        xy = self._unit.xy
        yaw = self._unit.yaw
        h = self._unit.base_height
        for i in _match_indices(objects, query):
            if self._carry.is_carried(i):
                continue  # a carried object is not "spotted" out in the world
            obj = objects[i]
            oxy = (float(obj["x"]), float(obj["y"]))
            obj_z = max(0.12, float(obj.get("size", 0.2)) / 2.0)
            if is_object_visible(xy, yaw, h, oxy, walls, obj_z=obj_z, cfg=self._vis):
                return self._confirm(query, oxy)
        return None

    def _confirm(self, query: ObjectQuery, oracle_xy: XY) -> XY:
        """Confirm an oracle-visible sighting with the detector (groundnet mode).

        Returns the detector's world-xy estimate when it confirms, else the
        oracle xy (fallback). In oracle mode (no perception) returns the oracle xy.
        """
        if self._perception is None:
            self.last_see_source = "oracle"
            return oracle_xy
        rx, ry = self._unit.xy
        dist = ((oracle_xy[0] - rx) ** 2 + (oracle_xy[1] - ry) ** 2) ** 0.5
        if dist > self._confirm_range_m:
            self.last_see_source = "oracle"
            return oracle_xy
        det = self._perception.confirm(
            query, (rx, ry, self._unit.yaw), oracle_xy=oracle_xy)
        if det is not None:
            self.last_see_source = "detector"
            return det.world_xy
        self.last_see_source = "oracle_fallback"
        return oracle_xy

    def report_origin(self) -> Tuple[XY, str]:
        """Return this robot's own ``(x, y)`` pose and current room name (F3).

        The pose is the robot's exactly-known pelvis position; the room is the
        named region it currently stands in (``room_of`` on a rooms layout, the
        "north/middle/south area" third on the hero hall).
        """
        xy = (float(self._unit.xy[0]), float(self._unit.xy[1]))
        room = (region_name_for_xy(self._layout, xy)
                if self._layout is not None else "the area")
        return (xy, room)

    # -- navigation -------------------------------------------------------
    def goto(self, xy: XY) -> None:
        """Begin navigating to a world position (plans an A* path)."""
        self.last_plan_ok = self._plan_to(xy)

    def _plan_to(self, xy: XY) -> bool:
        """Plan to ``xy``; on failure retry an approach point pulled toward the robot."""
        if self._unit.assign_goal((float(xy[0]), float(xy[1]))):
            return True
        rx, ry = self._unit.xy
        dx, dy = xy[0] - rx, xy[1] - ry
        dist = (dx * dx + dy * dy) ** 0.5
        if dist > 0.5:  # aim ~0.4 m short of the target and retry once
            k = (dist - 0.4) / dist
            return self._unit.assign_goal((rx + dx * k, ry + dy * k))
        return False

    def arrived(self) -> bool:
        """Whether the most recent nav target has been reached."""
        return bool(self._unit.done)

    def failed(self) -> bool:
        """Whether the robot can no longer make navigation progress.

        True if the unit has fallen (a terminal state that never ``arrived``\\ s)
        or its most recent goto/deliver plan was unreachable even after the
        approach-point retry in :meth:`_plan_to`.
        """
        return bool(getattr(self._unit, "fell", False)) or self.last_plan_ok is False

    def failure_reason(self) -> str:
        """Short reason for the current :meth:`failed` state."""
        if bool(getattr(self._unit, "fell", False)):
            return "robot fell"
        return "goal unreachable"

    # -- search -----------------------------------------------------------
    def start_search(self, query: ObjectQuery, region: str) -> bool:
        """Begin patrolling a named region for the object.

        Returns:
            True if the controller found a reachable patrol; False if the region
            has no coverable waypoints (the protocol then declines the command).
        """
        # A fresh search assignment clears any stale plan-failure from a prior
        # navigation so :meth:`failed` reflects only this searcher's own motion.
        self.last_plan_ok = None
        return bool(self._search.start(self._cfg, region))

    def abort_search(self) -> None:
        """Stop any in-progress search immediately."""
        self._search.stop()

    # -- manipulation -----------------------------------------------------
    def pickup(self, query: ObjectQuery) -> bool:
        """Mock-pick the nearest matching object within reach.

        Returns:
            True if the object is now held; False if no matching object exists or
            the nearest one was out of the carry manager's pickup radius.
        """
        idx = self._nearest_match(query)
        if idx is None:
            return False
        return bool(self._carry.pickup(self.callsign, idx))

    def deliver(self, destination_xy: XY) -> None:
        """Navigate to the destination while carrying the object."""
        self._carry.set_destination(self.callsign, destination_xy)
        self.last_plan_ok = self._plan_to(destination_xy)

    def _nearest_match(self, query: ObjectQuery) -> Optional[int]:
        """Index of the matching object nearest the robot's pelvis."""
        objects = self._cfg["objects"]
        rx, ry = self._unit.xy
        best: Optional[int] = None
        best_d = float("inf")
        for i in _match_indices(objects, query):
            d = (float(objects[i]["x"]) - rx) ** 2 + (float(objects[i]["y"]) - ry) ** 2
            if d < best_d:
                best_d, best = d, i
        return best
