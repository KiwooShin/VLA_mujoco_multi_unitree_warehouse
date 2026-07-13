"""arena.py — MJCF assembly for the multi-robot warehouse.

Turns a :class:`code.warehouse.layout.WarehouseLayout` into the universal
``scene_cfg`` dict (the contract every rollout loop consumes, extended with
``walls``/``zones``/``layout_name``) and compiles that into a
``mujoco.MjModel``. Object geoms keep the exact ``obj_{i}`` naming/sizing
conventions of ``code.sim.arena_build.build_arena`` so the baseline perception
stack works against the warehouse model unchanged (docs/multi_plan.md sec 3).

Public API
----------
warehouse_scene_cfg(layout, *, robot, objects, rng, target_index, instruction)
build_warehouse_arena(scene_cfg) -> mujoco.MjModel
"""

from __future__ import annotations

import dataclasses
import math
from typing import Dict, List, Optional

import mujoco
import numpy as np

from code.sim.arena_build import (
    COLORS,
    EGO_H,
    EGO_W,
    G1_XML,
    GROUNDING_H,
    GROUNDING_W,
    PROXIMITY_H,
    PROXIMITY_W,
    SHAPES,
    TP_H,
    TP_W,
    _add_geom,
    _rgb255_to_rgba1,
)
from code.warehouse.layout import WarehouseLayout

_STOP_R: float = 0.4
_HORIZON: int = 1400
_AMBIENT: float = 0.5
_FLOOR_RGBA: List[float] = [0.86, 0.86, 0.88, 1.0]


# ---------------------------------------------------------------------------
# scene_cfg assembly
# ---------------------------------------------------------------------------
def _default_objects(layout: WarehouseLayout,
                     rng: np.random.Generator) -> List[dict]:
    """Sample one uniquely coloured/shaped object per layout object spot.

    Args:
        layout: Layout supplying ``object_spots``.
        rng: RNG used to pick unique (colour, shape) combos.

    Returns:
        List of object dicts with the baseline scene keys (``color_name``,
        ``color_rgb``, ``shape_name``, ``size``, ``x``, ``y``).
    """
    spots = layout.object_spots
    combos = [(ci, si) for ci in range(len(COLORS)) for si in range(len(SHAPES))]
    n = min(len(spots), len(combos))
    chosen = rng.choice(len(combos), size=n, replace=False)
    objects: List[dict] = []
    for (sx, sy), k in zip(spots, chosen):
        ci, si = combos[int(k)]
        color_name, color_rgb = COLORS[ci]
        shape_name, size = SHAPES[si]
        objects.append({
            "color_name": color_name,
            "color_rgb": color_rgb,
            "shape_name": shape_name,
            "size": float(size),
            "x": float(sx),
            "y": float(sy),
        })
    return objects


def warehouse_scene_cfg(
    layout: WarehouseLayout,
    *,
    robot: str = "Alpha",
    objects: Optional[List[dict]] = None,
    rng: Optional[np.random.Generator] = None,
    target_index: int = 0,
    instruction: str = "",
) -> dict:
    """Build the universal ``scene_cfg`` dict for a warehouse layout.

    Args:
        layout: Source warehouse layout.
        robot: Callsign whose spawn bay the robot starts in.
        objects: Explicit object list; if None, one object per object spot is
            sampled from the baseline palettes.
        rng: RNG for default object sampling (default: seed 0).
        target_index: Index into ``objects`` naming the fetch target.
        instruction: Natural-language task; auto-generated from the target when
            empty.

    Returns:
        Dict with the baseline contract keys (``arena_size``, ``robot_xy``,
        ``robot_yaw``, ``objects``, ``target_index``, ``instruction``,
        ``stop_r``, ``horizon``, ``lighting``, ``difficulty``) plus warehouse
        keys ``walls`` (serialized WallSpecs), ``zones`` and ``layout_name``.

    Raises:
        KeyError: If ``robot`` has no spawn pose in the layout.
        ValueError: If ``target_index`` is out of range for the object list.
    """
    if robot not in layout.spawn_poses:
        raise KeyError(f"unknown robot {robot!r}; have {list(layout.spawn_poses)}")

    rx, ry, ryaw = layout.spawn_poses[robot]
    if objects is None:
        objects = _default_objects(layout, rng or np.random.default_rng(0))
    if objects and not (0 <= target_index < len(objects)):
        raise ValueError(
            f"target_index {target_index} out of range for {len(objects)} objects"
        )
    if not instruction and objects:
        tgt = objects[target_index]
        instruction = (
            f"fetch the {tgt['color_name']} {tgt['shape_name']} "
            f"to the delivery pad"
        )

    arena_size = max(layout.hall_x, layout.hall_y) / 2.0
    return {
        "arena_size": float(arena_size),
        "hall_x": float(layout.hall_x),
        "hall_y": float(layout.hall_y),
        "robot_xy": (float(rx), float(ry)),
        "robot_yaw": float(ryaw),
        "objects": objects,
        "target_index": int(target_index),
        "instruction": instruction,
        "stop_r": _STOP_R,
        "horizon": _HORIZON,
        "lighting": {"ambient": _AMBIENT},
        "difficulty": "warehouse",
        "walls": [dataclasses.asdict(w) for w in layout.walls],
        "zones": [dataclasses.asdict(z) for z in layout.zones],
        "layout_name": layout.name,
    }


# ---------------------------------------------------------------------------
# MJCF build
# ---------------------------------------------------------------------------
def _attach_robot(spec: Optional[mujoco.MjSpec] = None) -> mujoco.MjSpec:
    """Return a spec containing the G1 robot (Phase-1 single-robot seam).

    Phase 1 ships one robot: the robot spec is the base, loaded via
    ``MjSpec.from_file`` exactly like ``build_arena`` so all baseline
    qpos-slicing code stays valid verbatim. The ``spec`` argument is the
    extension seam — Phase 2 can pass a shared warehouse base spec here and
    attach N kinematically-synced robot bodies for the cross-visibility viz
    model (docs/multi_plan.md sec 3) without touching ``build_warehouse_arena``.

    Args:
        spec: Optional base spec; when None a fresh robot spec is loaded.

    Returns:
        A ``mujoco.MjSpec`` with the G1 robot present.
    """
    if spec is None:
        return mujoco.MjSpec.from_file(G1_XML)
    return spec


def _add_wall(wb: mujoco.MjsBody, wall: dict) -> None:
    """Add one wall/shelf/partition block as a yawed box geom."""
    half_x = float(wall["half_x"])
    half_y = float(wall["half_y"])
    height = float(wall["height"])
    cx, cy = float(wall["cx"]), float(wall["cy"])
    yaw = float(wall["yaw"])
    g = _add_geom(
        wb, mujoco.mjtGeom.mjGEOM_BOX,
        [half_x, half_y, height / 2.0],
        [cx, cy, height / 2.0],
        list(wall["rgba"]),
        wall["name"],
    )
    g.quat = [math.cos(yaw / 2.0), 0.0, 0.0, math.sin(yaw / 2.0)]


def _add_object(wb: mujoco.MjsBody, i: int, obj: dict) -> None:
    """Add object ``i`` using the exact build_arena naming/sizing conventions."""
    rgba = _rgb255_to_rgba1(obj["color_rgb"])
    hs = obj["size"] / 2.0
    ox, oy = float(obj["x"]), float(obj["y"])
    shape = obj["shape_name"]
    oname = f"obj_{i}"

    if shape == "ball":
        _add_geom(wb, mujoco.mjtGeom.mjGEOM_SPHERE, [hs, hs, hs],
                  [ox, oy, hs], rgba, oname)
    elif shape == "cube":
        _add_geom(wb, mujoco.mjtGeom.mjGEOM_BOX, [hs, hs, hs],
                  [ox, oy, hs], rgba, oname)
    elif shape == "cylinder":
        _add_geom(wb, mujoco.mjtGeom.mjGEOM_CYLINDER, [hs, hs * 1.6, hs],
                  [ox, oy, hs * 1.6], rgba, oname)
    elif shape == "cone":
        cone_h = hs * 2.2
        _add_geom(wb, mujoco.mjtGeom.mjGEOM_CYLINDER, [hs, cone_h * 0.5, hs],
                  [ox, oy, cone_h * 0.5], rgba, oname)
        top_rgba = [rgba[0] * 0.85, rgba[1] * 0.85, rgba[2] * 0.85, 1.0]
        _add_geom(wb, mujoco.mjtGeom.mjGEOM_BOX,
                  [hs * 0.35, hs * 0.35, cone_h * 0.45],
                  [ox, oy, cone_h + cone_h * 0.45], top_rgba, f"obj_{i}_tip")
    else:
        _add_geom(wb, mujoco.mjtGeom.mjGEOM_SPHERE, [hs, hs, hs],
                  [ox, oy, hs], rgba, oname)


def _add_zone(wb: mujoco.MjsBody, zone: dict) -> None:
    """Add a zone as a thin, non-colliding visual floor pad."""
    g = _add_geom(
        wb, mujoco.mjtGeom.mjGEOM_BOX,
        [float(zone["half_x"]), float(zone["half_y"]), 0.005],
        [float(zone["cx"]), float(zone["cy"]), 0.01],
        list(zone["rgba"]),
        f"zone_{zone['name']}",
    )
    g.contype = 0
    g.conaffinity = 0


def _add_overhead_lights(wb: mujoco.MjsBody, half_x: float,
                         half_y: float) -> None:
    """Add four overhead spot lights for an even industrial look."""
    for sx in (-0.5, 0.5):
        for sy in (-0.5, 0.5):
            try:
                light = wb.add_light()
                light.pos = [sx * half_x, sy * half_y, 6.0]
                light.dir = [0.0, 0.0, -1.0]
                light.type = mujoco.mjtLightType.mjLIGHT_SPOT
                light.cutoff = 80.0
                light.exponent = 4.0
                light.diffuse = [0.45, 0.45, 0.45]
                light.specular = [0.15, 0.15, 0.15]
            except Exception:
                return  # older mujoco spec API — headlight alone suffices


def build_warehouse_arena(scene_cfg: dict) -> mujoco.MjModel:
    """Compile a warehouse ``MjModel`` from a warehouse ``scene_cfg``.

    Args:
        scene_cfg: Dict from :func:`warehouse_scene_cfg` (needs ``walls``,
            ``objects``, ``zones``, ``lighting``).

    Returns:
        A compiled ``mujoco.MjModel`` with the G1 robot, perimeter/interior
        walls, ``obj_{i}`` object geoms, ``zone_{name}`` visual pads, a
        light-gray floor and an offscreen buffer large enough for every
        ArenaRenderer camera stream.
    """
    spec = _attach_robot()

    try:
        spec.visual.global_.offwidth = max(EGO_W, GROUNDING_W, PROXIMITY_W, TP_W)
        spec.visual.global_.offheight = max(EGO_H, GROUNDING_H, PROXIMITY_H, TP_H)
    except Exception:
        pass  # older mujoco — ignore

    wb = spec.worldbody

    for wall in scene_cfg.get("walls", []):
        _add_wall(wb, wall)
    for i, obj in enumerate(scene_cfg.get("objects", [])):
        _add_object(wb, i, obj)
    for zone in scene_cfg.get("zones", []):
        _add_zone(wb, zone)

    ambient = float(scene_cfg.get("lighting", {}).get("ambient", _AMBIENT))
    try:
        spec.visual.headlight.ambient = [ambient, ambient, ambient]
        spec.visual.headlight.diffuse = [0.5, 0.5, 0.5]
        spec.visual.headlight.specular = [0.2, 0.2, 0.2]
    except Exception:
        pass

    _add_overhead_lights(wb, 8.0, 6.0)

    model = spec.compile()

    try:
        fid = model.geom("floor").id
        model.geom_rgba[fid] = _FLOOR_RGBA
    except Exception:
        pass

    return model
