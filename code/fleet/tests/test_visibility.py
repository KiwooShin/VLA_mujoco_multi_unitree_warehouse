"""Unit tests for the geometric visibility oracle (code.fleet.visibility).

Hand-built occlusion cases against the hero layout's walls: an object down an
open lane is visible; the same object behind a shelf or inside the NE alcove is
hidden; range and field-of-view gates behave; and the pure segment/OBB geometry
is exercised directly.
"""

from __future__ import annotations

import dataclasses
import math
import unittest

from code.fleet import visibility as V
from code.warehouse.layout import CALLSIGNS, hero_layout

_H = 0.74  # a nominal pelvis height for the camera


def _walls():
    return [dataclasses.asdict(w) for w in hero_layout().walls]


class TestFovAndRange(unittest.TestCase):
    def test_horizontal_fov_from_ego_constants(self) -> None:
        half = V.head_half_fov_rad()
        self.assertGreater(math.degrees(half), 40.0)
        self.assertLess(math.degrees(half), 60.0)

    def test_out_of_range_hidden(self) -> None:
        # No walls; object 7 m ahead exceeds the 6 m range.
        self.assertFalse(V.is_object_visible((0, 0), 0.0, _H, (7.0, 0.0), []))
        self.assertTrue(V.is_object_visible((0, 0), 0.0, _H, (5.0, 0.0), []))

    def test_behind_robot_out_of_fov(self) -> None:
        # Object directly behind the robot (facing +x) is outside the FOV cone.
        self.assertFalse(V.is_object_visible((0, 0), 0.0, _H, (-2.0, 0.0), []))
        self.assertTrue(V.is_object_visible((0, 0), 0.0, _H, (2.0, 0.0), []))

    def test_coincident_object_seen(self) -> None:
        self.assertTrue(V.is_object_visible((1, 1), 1.0, _H, (1, 1), []))


class TestLineOfSight(unittest.TestCase):
    def test_open_lane_visible(self) -> None:
        walls = _walls()
        # Alpha at its south bay facing +y sees the red-cube spot up the open
        # west lane (spot 6 at (-5, 0.5)).
        sx, sy, syaw = hero_layout().spawn_poses["Alpha"]
        self.assertTrue(
            V.is_object_visible((sx, sy), syaw, _H, (-5.0, 0.5), walls))

    def test_behind_shelf_hidden(self) -> None:
        walls = _walls()
        # From the far south, an object north of shelf row B is occluded.
        self.assertFalse(
            V.is_object_visible((2.0, -5.0), math.pi / 2, _H, (2.0, -1.0), walls))

    def test_alcove_hidden_from_hall(self) -> None:
        walls = _walls()
        # The NE-alcove spot is hidden from every robot's spawn pose.
        for cs in CALLSIGNS:
            sx, sy, syaw = hero_layout().spawn_poses[cs]
            self.assertFalse(
                V.is_object_visible((sx, sy), syaw, _H, (6.5, 4.7), walls),
                f"{cs} should not see the alcove object")

    def test_alcove_visible_from_opening(self) -> None:
        walls = _walls()
        # Standing in the east opening looking north, the alcove object is seen.
        self.assertTrue(
            V.is_object_visible((7.4, 3.0), math.atan2(1.7, -0.9), _H,
                                (6.5, 4.7), walls))

    def test_low_wall_does_not_occlude(self) -> None:
        # A short prop (height below both endpoints) never blocks the sightline.
        low = [{"cx": 1.0, "cy": 0.0, "half_x": 0.1, "half_y": 1.0,
                "yaw": 0.0, "height": 0.05}]
        self.assertTrue(V.line_of_sight_clear((0, 0), (2, 0), low,
                                              head_z=1.2, obj_z=0.1))

    def test_tall_wall_occludes(self) -> None:
        tall = [{"cx": 1.0, "cy": 0.0, "half_x": 0.1, "half_y": 1.0,
                 "yaw": 0.0, "height": 2.0}]
        self.assertFalse(V.line_of_sight_clear((0, 0), (2, 0), tall,
                                               head_z=1.2, obj_z=0.1))


class TestPartialVisibilitySampledLOS(unittest.TestCase):
    """Extent-sampled LOS: an edge peeking past a wall counts as visible.

    Hand-built wall: head at the origin looking +x at an object 2 m away; a thin
    tall wall crosses the centre sightline (y=0) but ends just below it, so the
    object's +y edge (at its radius) clears the wall.
    """

    # Blocks the y=0 centre segment (covers y in [-1.0, +0.05]) but not y=+0.1.
    _EDGE_WALL = [{"cx": 1.0, "cy": -0.475, "half_x": 0.1, "half_y": 0.525,
                   "yaw": 0.0, "height": 2.0}]
    # Covers y in [-1, 1]: blocks the centre AND every edge sample.
    _FULL_WALL = [{"cx": 1.0, "cy": 0.0, "half_x": 0.1, "half_y": 1.0,
                   "yaw": 0.0, "height": 2.0}]

    def test_centre_only_hidden_but_edge_visible(self) -> None:
        # Centre-only (radius 0) segment is blocked...
        self.assertFalse(V.line_of_sight_clear((0, 0), (2, 0), self._EDGE_WALL,
                                               head_z=1.2, obj_z=0.1,
                                               obj_radius=0.0))
        # ...but sampling the 0.2 m extent finds the +y edge peeking past the wall.
        self.assertTrue(V.line_of_sight_clear((0, 0), (2, 0), self._EDGE_WALL,
                                              head_z=1.2, obj_z=0.1,
                                              obj_radius=0.2))

    def test_is_object_visible_uses_sampled_extent(self) -> None:
        self.assertFalse(
            V.is_object_visible((0, 0), 0.0, _H, (2, 0), self._EDGE_WALL,
                                obj_z=0.1, obj_radius=0.0))
        self.assertTrue(
            V.is_object_visible((0, 0), 0.0, _H, (2, 0), self._EDGE_WALL,
                                obj_z=0.1, obj_radius=0.2))

    def test_fully_occluded_stays_hidden_even_with_radius(self) -> None:
        # A wall wider than the object's extent blocks every sample.
        self.assertFalse(V.line_of_sight_clear((0, 0), (2, 0), self._FULL_WALL,
                                               head_z=1.2, obj_z=0.1,
                                               obj_radius=0.2))
        self.assertFalse(
            V.is_object_visible((0, 0), 0.0, _H, (2, 0), self._FULL_WALL,
                                obj_z=0.1, obj_radius=0.2))

    def test_fov_and_range_gate_on_centre_not_edge(self) -> None:
        # An object whose CENTRE is behind the robot stays hidden however big its
        # extent — the FOV/range gates use the centre, only LOS samples the edges.
        self.assertFalse(
            V.is_object_visible((0, 0), 0.0, _H, (-2, 0), [], obj_radius=0.3))
        # And a centre beyond range is hidden regardless of extent.
        self.assertFalse(
            V.is_object_visible((0, 0), 0.0, _H, (7.0, 0.0), [], obj_radius=0.3))

    def test_open_object_visible_with_or_without_sampling(self) -> None:
        # No walls: visible either way (sampling never narrows the gate).
        self.assertTrue(V.line_of_sight_clear((0, 0), (2, 0), [], head_z=1.2,
                                              obj_z=0.1, obj_radius=0.0))
        self.assertTrue(V.line_of_sight_clear((0, 0), (2, 0), [], head_z=1.2,
                                              obj_z=0.1, obj_radius=0.2))


class TestSegmentGeometry(unittest.TestCase):
    def test_segment_aabb_hit_and_miss(self) -> None:
        self.assertTrue(V._segment_intersects_aabb(-2, 0, 2, 0, 1, 1))
        self.assertFalse(V._segment_intersects_aabb(-2, 5, 2, 5, 1, 1))

    def test_yawed_wall_intersection(self) -> None:
        wall = {"cx": 0.0, "cy": 0.0, "half_x": 1.0, "half_y": 0.1,
                "yaw": math.pi / 4}
        # A segment along +y crosses the 45-deg diagonal wall.
        self.assertTrue(V._segment_intersects_wall((0, -2), (0, 2), wall))
        # A far-offset parallel segment does not.
        self.assertFalse(V._segment_intersects_wall((5, -2), (5, 2), wall))

    def test_wrap_angle(self) -> None:
        self.assertAlmostEqual(V.wrap_angle(0.5), 0.5)
        self.assertAlmostEqual(V.wrap_angle(2 * math.pi + 0.5), 0.5)
        self.assertAlmostEqual(abs(V.wrap_angle(3 * math.pi)), math.pi)


if __name__ == "__main__":
    unittest.main()
