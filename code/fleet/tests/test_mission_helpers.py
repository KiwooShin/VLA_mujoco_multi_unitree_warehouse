"""Fast unit tests for the mission runner's pure helpers (no physics)."""

from __future__ import annotations

import dataclasses
import unittest

from code.comms.messages import ObjectQuery
from code.comms.protocol import RobotState
from code.fleet.mission import _OWNER_PHASE, MissionRunner, delivery_xy, resolve_query
from code.warehouse.layout import hero_layout


class TestResolveQuery(unittest.TestCase):
    def test_colour_and_shape(self) -> None:
        self.assertEqual(resolve_query("fetch the red cube to the delivery pad"),
                         ObjectQuery("red", "cube"))

    def test_shape_only(self) -> None:
        self.assertEqual(resolve_query("bring the cone over"),
                         ObjectQuery(None, "cone"))

    def test_colour_only(self) -> None:
        self.assertEqual(resolve_query("grab the blue thing"),
                         ObjectQuery("blue", None))

    def test_unresolvable(self) -> None:
        self.assertIsNone(resolve_query("do something useful"))

    def test_generic_object_reference(self) -> None:  # F4
        # "the object" / "an object" resolve to the wildcard query (any object).
        for body in ("bring the object to the destination",
                     "bring the object to destination",
                     "fetch an object", "carry the item over"):
            q = resolve_query(body)
            self.assertIsNotNone(q, body)
            self.assertTrue(q.is_generic, body)
        # A bare pronoun only counts with a fetch verb.
        self.assertTrue(resolve_query("bring me something").is_generic)
        self.assertIsNone(resolve_query("do something useful"))
        # A specific colour/shape still wins over the generic path.
        self.assertEqual(resolve_query("bring the red object"),
                         ObjectQuery("red", None))


class TestDeliveryXy(unittest.TestCase):
    def test_delivery_pad_from_layout(self) -> None:
        name, xy = delivery_xy(hero_layout())
        self.assertEqual(name, "delivery pad")
        self.assertEqual(xy, (5.8, -1.0))


class TestPhaseLabels(unittest.TestCase):
    def test_owner_phase_covers_owner_states(self) -> None:
        for st in (RobotState.OWNER_QUERYING, RobotState.OWNER_DELEGATING,
                   RobotState.OWNER_NAVIGATING, RobotState.OWNER_DELIVERING):
            self.assertIn(st, _OWNER_PHASE)
            self.assertTrue(_OWNER_PHASE[st].isupper())


@dataclasses.dataclass
class _Mission:
    owner: object


class _Proto:
    def __init__(self, idle: bool) -> None:
        self._idle = idle

    def is_idle(self) -> bool:
        return self._idle


class _Unit:
    def __init__(self, xy) -> None:
        self.xy = xy


class _Fleet:
    def __init__(self, poses) -> None:
        self.units = {c: _Unit(xy) for c, xy in poses.items()}


class _FakeRunner:
    """A duck-typed stand-in exercising the real ``_recruitable_peers`` math."""

    _recruitable_peers = MissionRunner._recruitable_peers
    _partition_searchers = MissionRunner._partition_searchers

    def __init__(self, poses, owners, idle, done, concurrent=True) -> None:
        self._concurrent = concurrent
        self.callsigns = list(poses)
        self._missions = [_Mission(o) for o in owners]
        self._done = set(done)
        self.protocols = {c: _Proto(c in set(idle)) for c in poses}
        self.fleet = _Fleet(poses)

    def _owner_done(self, owner) -> bool:
        return owner in self._done


class TestRecruitableBudget(unittest.TestCase):
    """The cross-owner searcher budget partitions the free peers disjointly."""

    # Alpha near Charlie (left), Bravo near Delta (right).
    POSES = {"Alpha": (-8.0, 0.0), "Bravo": (8.0, 0.0),
             "Charlie": (-7.0, 1.0), "Delta": (7.0, -1.0)}

    def test_single_mission_no_limit(self) -> None:
        r = _FakeRunner(self.POSES, owners=["Alpha"], idle=[], done=[],
                        concurrent=False)
        self.assertIsNone(r._recruitable_peers("Alpha"))

    def test_two_owners_two_peers_disjoint_nearest(self) -> None:
        # Both free peers idle -> each owner gets the peer NEAREST it, no overlap.
        r = _FakeRunner(self.POSES, owners=["Alpha", "Bravo"],
                        idle=["Charlie", "Delta"], done=[])
        a = r._recruitable_peers("Alpha")
        b = r._recruitable_peers("Bravo")
        self.assertEqual(a, {"Charlie"})       # Charlie is nearest Alpha
        self.assertEqual(b, {"Delta"})         # Delta is nearest Bravo
        self.assertEqual(a & b, set())         # disjoint pools (need-to-know)

    def test_far_owner_still_gets_a_searcher_not_starved(self) -> None:
        # BOTH free peers sit next to Alpha; the cap still forces one over to Bravo
        # (balanced disjoint shares — nobody is starved).
        poses = {"Alpha": (-8.0, 0.0), "Bravo": (8.0, 0.0),
                 "Charlie": (-7.0, 1.0), "Delta": (-7.5, -1.0)}
        r = _FakeRunner(poses, owners=["Alpha", "Bravo"],
                        idle=["Charlie", "Delta"], done=[])
        a = r._recruitable_peers("Alpha")
        b = r._recruitable_peers("Bravo")
        self.assertEqual(len(a), 1)
        self.assertEqual(len(b), 1)
        self.assertEqual(a & b, set())
        self.assertEqual(a | b, {"Charlie", "Delta"})

    def test_one_free_peer_short_handed_owner_empty(self) -> None:
        # 3 robots, 2 owners, one free peer nearest Alpha: Alpha reserves it, Bravo
        # gets none (it will wait in OWNER_DELEGATING, then recruit once freed).
        poses = {"Alpha": (-8.0, 0.0), "Bravo": (8.0, 0.0), "Charlie": (-7.0, 0.0)}
        r = _FakeRunner(poses, owners=["Alpha", "Bravo"], idle=["Charlie"],
                        done=[])
        self.assertEqual(r._recruitable_peers("Alpha"), {"Charlie"})
        self.assertEqual(r._recruitable_peers("Bravo"), set())

    def test_freed_robots_flow_to_still_searching_owner(self) -> None:
        # Alpha's mission finished (idle again); its freed robot + Alpha itself
        # now become recruitable for the still-active Bravo.
        poses = {"Alpha": (-8.0, 0.0), "Bravo": (8.0, 0.0), "Charlie": (-7.0, 0.0)}
        r = _FakeRunner(poses, owners=["Alpha", "Bravo"],
                        idle=["Alpha", "Charlie"], done=["Alpha"])
        b = r._recruitable_peers("Bravo")
        self.assertIn("Charlie", b)
        self.assertIn("Alpha", b)


if __name__ == "__main__":
    unittest.main()
