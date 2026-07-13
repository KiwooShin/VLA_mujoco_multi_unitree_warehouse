"""allocator.py — Path-length-optimal task allocation for fleet requests.

A "fleet"-addressed request ("someone bring me the red cube") is routed by the
message bus to the allocator inbox. The allocator picks the idle robot with the
**shortest planned A\\* path** (not straight-line distance) to where the work
starts:

* if any robot can currently see the object, that is its world position;
* otherwise each candidate is scored to the centroid of *its nearest unsearched
  region* — the robot best placed to begin the divide-and-conquer search.

Ties break by callsign order. The chosen robot then receives the task exactly as
if the user had addressed it directly (``REQUEST_TASK``, ``requester="user"``),
and the decision is logged to the bus as a ``STATUS_UPDATE`` from ``"allocator"``
so it shows up in the demo transcript.

Path lengths come from the same inflated-grid A\\* the robots navigate with, so
the allocator's cost model is exactly the cost the winner will pay.
"""

from __future__ import annotations

import dataclasses
import math
from typing import Dict, Optional, Sequence, Tuple

from code.apps.warehouse_demo.planning import build_inflated_grid
from code.comms.messages import ObjectQuery
from code.fleet.search import SEARCH_REGIONS, free_centroid, region_centroid
from code.fleet.visibility import VisibilityConfig, is_object_visible
from code.planner.astar import PathNotFoundError, path_length, plan_path, shortcut_path

XY = Tuple[float, float]

# Planner settings mirroring NavParams (kept local so the allocator has no
# simulator dependency; must match code.apps.warehouse_demo.nav_core.NavParams).
_RESOLUTION_M: float = 0.10
_INFLATE_M: float = 0.40
_SNAP_M: float = 0.60


@dataclasses.dataclass(frozen=True)
class RobotPose:
    """A robot's exactly-known pose for allocation scoring."""

    xy: XY
    yaw: float
    base_height: float


def _pose_is_finite(pose: "RobotPose") -> bool:
    """Whether every component of a pose is a finite number (safe to score).

    A NaN/inf pose (e.g. a robot whose physics blew up) would raise inside the
    grid/planner; such a candidate is treated as unallocatable instead.
    """
    return all(math.isfinite(v) for v in
               (pose.xy[0], pose.xy[1], pose.yaw, pose.base_height))


@dataclasses.dataclass(frozen=True)
class AllocationResult:
    """The outcome of an allocation.

    Attributes:
        winner: Chosen callsign, or ``None`` if nobody could be scored.
        target_xy: The world point the winner was scored to (object or region
            centroid), or ``None``.
        reason: ``"visible"`` (some robot sees the object) or ``"search"``.
        costs: callsign -> planned path length (m); ``inf`` if unreachable.
        region: callsign -> assigned nearest region (search mode only).
    """

    winner: Optional[str]
    target_xy: Optional[XY]
    reason: str
    costs: Dict[str, float]
    region: Dict[str, str]

    def describe(self) -> str:
        """Human one-liner for the bus STATUS_UPDATE caption."""
        if self.winner is None:
            return "no idle robot available to take the task"
        cost = self.costs.get(self.winner, float("inf"))
        if self.reason == "visible":
            return (f"assigning {self.winner} (shortest path {cost:.1f} m to the "
                    f"object)")
        reg = self.region.get(self.winner, "?")
        return (f"nobody sees it — assigning {self.winner} to search {reg} "
                f"(shortest path {cost:.1f} m)")


def planned_path_length(scene_cfg: dict, start_xy: XY, goal_xy: XY, *,
                        resolution: float = _RESOLUTION_M,
                        inflate: float = _INFLATE_M,
                        snap: float = _SNAP_M) -> float:
    """Return the smoothed A* path length from ``start_xy`` to ``goal_xy`` (m).

    Uses the same inflated grid (walls + non-goal object stamps) the robots plan
    over, so the cost equals the navigation cost. Returns ``inf`` if unreachable.
    """
    grid = build_inflated_grid(scene_cfg, resolution, inflate, goal_xy=goal_xy)
    try:
        raw = plan_path(grid, start_xy, goal_xy, snap_radius_m=snap)
    except PathNotFoundError:
        return float("inf")
    return path_length(shortcut_path(grid, raw))


def _nearest_region(xy: XY, scene_cfg: dict,
                    regions: Sequence[str]) -> Tuple[str, XY]:
    """Return the region whose centroid is nearest ``xy`` and that centroid."""
    hall_x, hall_y = float(scene_cfg["hall_x"]), float(scene_cfg["hall_y"])
    best_r = regions[0]
    best_c = region_centroid(regions[0], hall_x, hall_y)
    best_d = float("inf")
    for r in regions:
        cx, cy = region_centroid(r, hall_x, hall_y)
        d = (cx - xy[0]) ** 2 + (cy - xy[1]) ** 2
        if d < best_d:
            best_d, best_r, best_c = d, r, (cx, cy)
    return best_r, best_c


def _visible_object_xy(poses: Dict[str, RobotPose], scene_cfg: dict,
                       query: ObjectQuery,
                       vis_cfg: VisibilityConfig) -> Optional[XY]:
    """Return the object's xy if any robot can see a matching object, else None."""
    walls = scene_cfg.get("walls", [])
    objects = scene_cfg["objects"]
    for i, obj in enumerate(objects):
        if not query.matches(obj):
            continue
        oxy = (float(obj["x"]), float(obj["y"]))
        obj_z = max(0.12, float(obj.get("size", 0.2)) / 2.0)
        for pose in poses.values():
            if not _pose_is_finite(pose):
                continue  # a broken pose can't be scored and can't "see"
            if is_object_visible(pose.xy, pose.yaw, pose.base_height, oxy, walls,
                                 obj_z=obj_z, cfg=vis_cfg):
                return oxy
    return None


def allocate(poses: Dict[str, RobotPose], scene_cfg: dict, query: ObjectQuery,
             idle: Sequence[str], *,
             vis_cfg: Optional[VisibilityConfig] = None,
             regions: Sequence[str] = SEARCH_REGIONS) -> AllocationResult:
    """Choose the idle robot with the shortest planned path to the work.

    Args:
        poses: callsign -> :class:`RobotPose` for every robot (visibility uses
            all of them; only ``idle`` robots are scored/assignable).
        scene_cfg: Shared warehouse scene_cfg.
        query: The requested object.
        idle: Idle callsigns, in callsign (tie-break) order.
        vis_cfg: Visibility-oracle geometry.
        regions: Region labels for search-mode scoring.

    Returns:
        An :class:`AllocationResult`; ``winner`` is ``None`` if ``idle`` is empty
        or no idle robot can reach any target.
    """
    vis_cfg = vis_cfg or VisibilityConfig()
    obj_xy = _visible_object_xy(poses, scene_cfg, query, vis_cfg)

    costs: Dict[str, float] = {}
    assigned_region: Dict[str, str] = {}
    targets: Dict[str, XY] = {}
    for cs in idle:
        pose = poses[cs]
        if not _pose_is_finite(pose):
            costs[cs] = float("inf")  # unallocatable: skip, never the argmin winner
            continue
        if obj_xy is not None:
            targets[cs] = obj_xy
        else:
            region, _ = _nearest_region(pose.xy, scene_cfg, regions)
            assigned_region[cs] = region
            targets[cs] = free_centroid(scene_cfg, region)
        costs[cs] = planned_path_length(scene_cfg, pose.xy, targets[cs])

    winner: Optional[str] = None
    best = float("inf")
    for cs in idle:  # idle is already in callsign/tie-break order
        if costs[cs] < best:
            best, winner = costs[cs], cs

    reason = "visible" if obj_xy is not None else "search"
    target_xy = targets.get(winner) if winner is not None else None
    return AllocationResult(winner, target_xy, reason, costs, assigned_region)
