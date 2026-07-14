"""scenarios — the six Demo Set v2 scenario scripts + their ordered registry.

Each demo is a :class:`~code.apps.demos.scenarios.core.Scenario` (a deterministic
world + order + recording recipe). :data:`REGISTRY` lists them in the intended
gallery order; :func:`get` looks one up by name. ``produce_all`` records the
originals and builds the committed ``demo/`` deliverables from this registry.

Importing this package is cheap (no cv2 / MuJoCo): the scenario definitions and
their manifest-ambiguity preconditions are pure data, so the tests construct and
check every scenario without a simulator.
"""

from __future__ import annotations

from typing import Dict, List

from code.apps.demos.scenarios.core import Scenario
from code.apps.demos.scenarios import (clarify_fetch, dual_fetch,
                                       relay_multigoal, retask,
                                       six_robot_flagship, unseen_map)

REGISTRY: List[Scenario] = [
    clarify_fetch.SCENARIO,
    unseen_map.SCENARIO,
    dual_fetch.SCENARIO,
    relay_multigoal.SCENARIO,
    retask.SCENARIO,
    six_robot_flagship.SCENARIO,
]

_BY_NAME: Dict[str, Scenario] = {s.name: s for s in REGISTRY}


def get(name: str) -> Scenario:
    """Return the scenario named ``name`` (raises ``KeyError`` if unknown)."""
    return _BY_NAME[name]


def names() -> List[str]:
    """The scenario names in gallery order."""
    return [s.name for s in REGISTRY]


__all__ = ["Scenario", "REGISTRY", "get", "names"]
