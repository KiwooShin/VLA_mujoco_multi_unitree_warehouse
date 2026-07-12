"""Pure-pursuit waypoint following (geometry only, no MuJoCo).

:class:`WaypointFollower` turns a planned world-coordinate path (from
:func:`code.planner.astar.plan_path`) into a stream of steering targets for the
existing :func:`code.control.steer.steer` law. It is deliberately free of any
simulator or velocity-command coupling: the rollout loop calls
:meth:`WaypointFollower.target` each control step and feeds the returned point
straight into ``steer``.

Progress along the path is tracked by monotonic arc length, so loops,
crossovers or backtracks in the geometry can never re-capture an already-passed
waypoint, and a robot shoved off the path re-projects onto the nearest *forward*
point rather than snapping backward.
"""

from __future__ import annotations

import math
from typing import List, Optional, Sequence, Tuple

Point = Tuple[float, float]


class WaypointFollower:
    """Pure-pursuit follower over a fixed polyline path.

    Attributes:
        done: True once the robot has arrived within ``arrive_radius`` of the
            final goal.
        progress_fraction: Fraction of total path arc length passed, in [0, 1].
    """

    def __init__(
        self,
        path: Sequence[Point],
        arrive_radius: float = 0.35,
        lookahead: float = 0.8,
    ) -> None:
        """Initializes the follower.

        Args:
            path: World-coordinate waypoints (>= 1 point), start to goal.
            arrive_radius: Distance to the final goal at which the task is done.
            lookahead: Pure-pursuit lookahead distance in meters (> 0).

        Raises:
            ValueError: If ``path`` is empty, ``arrive_radius`` <= 0, or
                ``lookahead`` <= 0.
        """
        if len(path) == 0:
            raise ValueError("path must contain at least one waypoint")
        if arrive_radius <= 0.0:
            raise ValueError(f"arrive_radius must be > 0, got {arrive_radius}")
        if lookahead <= 0.0:
            raise ValueError(f"lookahead must be > 0, got {lookahead}")

        self._pts: List[Point] = [(float(p[0]), float(p[1])) for p in path]
        self.arrive_radius: float = float(arrive_radius)
        self.lookahead: float = float(lookahead)

        # Cumulative arc length at each waypoint; _cum[0] == 0.0.
        self._cum: List[float] = [0.0]
        for a, b in zip(self._pts, self._pts[1:]):
            self._cum.append(self._cum[-1] + math.hypot(b[0] - a[0], b[1] - a[1]))
        self._total: float = self._cum[-1]

        self._progress_arc: float = 0.0
        self.done: bool = False

    @property
    def progress_fraction(self) -> float:
        """Monotonic fraction of the path completed, in [0, 1]."""
        if self.done:
            return 1.0
        if self._total <= 0.0:
            return 0.0
        return max(0.0, min(1.0, self._progress_arc / self._total))

    @property
    def goal(self) -> Point:
        """The final goal waypoint."""
        return self._pts[-1]

    def _point_at_arc(self, s: float) -> Point:
        """Interpolates the world point at arc length ``s`` along the path."""
        if s <= 0.0 or self._total <= 0.0:
            return self._pts[0]
        if s >= self._total:
            return self._pts[-1]
        # Locate the segment containing arc length s.
        for i in range(len(self._pts) - 1):
            if self._cum[i + 1] >= s:
                seg = self._cum[i + 1] - self._cum[i]
                t = 0.0 if seg <= 0.0 else (s - self._cum[i]) / seg
                a, b = self._pts[i], self._pts[i + 1]
                return (a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1]))
        return self._pts[-1]

    def _closest_arc_forward(self, robot_xy: Point) -> float:
        """Arc length of the closest path point at or ahead of current progress.

        Projects the robot onto every path segment that is not entirely behind
        the current progress, clamping each projection so it never falls behind
        ``_progress_arc``. Returns the arc length of the nearest such point.
        """
        s_min = self._progress_arc
        best_arc = s_min
        best_d2 = float("inf")
        for i in range(len(self._pts) - 1):
            seg_end = self._cum[i + 1]
            if seg_end < s_min:
                continue  # segment fully behind current progress
            a, b = self._pts[i], self._pts[i + 1]
            seg = self._cum[i + 1] - self._cum[i]
            if seg <= 0.0:
                continue
            # Lower bound on t so that cum[i] + t*seg >= s_min.
            t_lo = max(0.0, (s_min - self._cum[i]) / seg)
            dx, dy = b[0] - a[0], b[1] - a[1]
            t = ((robot_xy[0] - a[0]) * dx + (robot_xy[1] - a[1]) * dy) / (seg * seg)
            t = min(1.0, max(t_lo, t))
            px, py = a[0] + t * dx, a[1] + t * dy
            d2 = (robot_xy[0] - px) ** 2 + (robot_xy[1] - py) ** 2
            if d2 < best_d2:
                best_d2 = d2
                best_arc = self._cum[i] + t * seg
        return max(best_arc, s_min)

    def target(self, robot_xy: Point) -> Optional[Point]:
        """Returns the current pure-pursuit steering target.

        Advances monotonic progress to the robot's closest forward path point,
        then returns the point ``lookahead`` meters further along the path. When
        the robot is within ``lookahead`` of the goal the goal itself is
        returned, and once within ``arrive_radius`` the follower is done.

        Args:
            robot_xy: Robot world position (x, y) in meters.

        Returns:
            The world-coordinate steering target, or None once the robot has
            arrived at the final goal.
        """
        if self.done:  # arrival is sticky: the task stays complete
            return None
        rxy = (float(robot_xy[0]), float(robot_xy[1]))
        goal = self._pts[-1]
        dist_goal = math.hypot(goal[0] - rxy[0], goal[1] - rxy[1])
        if dist_goal <= self.arrive_radius:
            self.done = True
            self._progress_arc = self._total
            return None

        # Single-point (degenerate) path: steer straight at the goal.
        if self._total <= 0.0:
            return goal

        closest_arc = self._closest_arc_forward(rxy)
        self._progress_arc = closest_arc  # monotonic (>= previous)

        target_arc = closest_arc + self.lookahead
        if target_arc >= self._total or dist_goal <= self.lookahead:
            return goal
        return self._point_at_arc(target_arc)
