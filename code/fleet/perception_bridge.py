"""perception_bridge.py — Real GROUND_NET learned detector in the fleet loop.

Cycle-2a: replace the geometric visibility oracle's *role as sole perception*
with the actual query-conditioned heatmap detector GROUND_NET
(``runs/nx6_heatmap_B/model_best.pt``), wired in as a **confirmer** on top of the
oracle. The geometric oracle (:mod:`code.fleet.visibility`) stays the occlusion
GATE (physics truth of what a wall blocks); the learned detector CONFIRMS a
proposed sighting by independently rendering that robot's grounding camera from
the shared :class:`~code.fleet.viz.FleetViz` model and running GROUND_NET on the
query's ``(shape, color)``.

:class:`RobotPerception` — one instance per robot. Each owns its OWN
:class:`~code.perception.ground_net.GroundNetState` (so NX-7 track hysteresis and
the VF-1 heatmap cache are isolated per robot and reset per mission — the exact
cross-contamination the baseline survey flagged about ``ground()``'s process-wide
singleton), while the *detector weights* (one :class:`HeatmapDetector` /
``TinyHeatmapUNet`` object, one checkpoint load) are SHARED across all robots:
``load_shared_detector`` loads once per process and the same model object is
assigned into every per-robot state's ``.detector`` field. Only the mutable
``track_*`` / ``last_heatmap`` fields differ per state.

Design note (measured domain shift, docs-worthy): GROUND_NET was trained on the
single-robot goto/search arena, whose floor is the G1 blue checker; the shared
warehouse viz model uses a light-grey industrial floor. That appearance shift
leaves the detector's geometry decode accurate (~0.03 m) but collapses its raw
confidence (~0.95 in-domain -> ~0.1 on the warehouse floor, with occasional
spurious top-edge peaks). The confirmer therefore accepts a detection only when
it clears a domain-calibrated confidence floor AND is geometrically consistent
with the oracle's occlusion-truth hypothesis (range + bearing) — a standard
sensor-fusion confirmation that admits the detector's accurate low-confidence
detections while rejecting its spurious peaks. All thresholds are module
constants, reported by :mod:`code.fleet.perception_eval`.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from typing import Optional, Tuple

os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco
import numpy as np

from code.comms.messages import ObjectQuery
from code.perception import ground_net as _gn
from code.perception.geometry import get_ego_intrinsics_rendered
from code.sim.arena_build import GROUNDING_H, GROUNDING_PITCH, GROUNDING_W
from code.sim.arena_cameras import _set_ego_cam

XY = Tuple[float, float]

# --- Confirmation thresholds (all measured / reported by perception_eval) ----
# Domain-calibrated confidence floor for a confirmation on the warehouse viz
# floor. Far below GROUND_NET's in-domain deploy tau (0.64, docs/nx14) because
# the light-floor domain shift suppresses confidence; kept honest by the
# geometry-consistency gate below (a spurious peak that happens to clear this
# floor is rejected unless it also matches the oracle's range+bearing).
CONFIRM_TAU: float = 0.08
# The learned detector only CONFIRMS when its independent (dist, bearing) agrees
# with the oracle's hypothesis within these gates — this is what makes it a
# confirmation rather than a blind (and, post-domain-shift, unreliable) detector.
CONFIRM_DIST_GATE_M: float = 1.5
CONFIRM_BEARING_GATE_DEG: float = 22.0
# The grounding camera's effective confirmation range (m). Beyond this the
# 26-degree grounding cam pushes the target to the top image edge where the
# learned detector is unreliable; the oracle already caps sightings at 6 m
# (visibility.MAX_RANGE_M) so this gate is rarely the binding one.
CONFIRM_RANGE_M: float = 7.0
# Half the grounding render's horizontal FOV (deg): FOVY=45 at 480x360 4:3 ->
# fovx = 2*atan(tan(22.5deg)*4/3) ~= 57.8deg, half ~= 28.9deg. The geometric
# visibility oracle's FOV (from EGO_FOVY=90) is far wider (~53deg half), so an
# object can be oracle-visible yet outside the grounding frame; the confirmer
# only runs the detector within THIS narrower, honest field of view.
GROUNDING_HALF_FOV_DEG: float = 28.0

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
# The warehouse-domain fine-tune (Cycle-2b, this task) and the original
# playground-trained baseline. The fine-tune becomes the default the moment its
# weights exist on disk; until then the loader falls back to the baseline so a
# fresh clone (no fine-tune yet) still runs.
_WAREHOUSE_FT_CKPT = os.path.join(_REPO_ROOT, "runs", "nx6_warehouse_ft", "model_best.pt")
_ORIGINAL_CKPT = os.path.join(_REPO_ROOT, "runs", "nx6_heatmap_B", "model_best.pt")
_DEVICE_DEFAULT = os.environ.get("GROUND_NET_DEVICE", "cuda")


def resolve_ckpt_path(*, ft_ckpt: str = _WAREHOUSE_FT_CKPT,
                      orig_ckpt: str = _ORIGINAL_CKPT,
                      env_var: str = "GROUND_NET_CKPT") -> str:
    """Resolve which GROUND_NET checkpoint the fleet should load.

    Resolution order (highest priority first), documented so operators know how
    to override it:

      1. The ``GROUND_NET_CKPT`` environment variable — an explicit operator
         override (used as-is, whether or not the file exists, so a bad path
         surfaces loudly rather than silently falling back).
      2. ``runs/nx6_warehouse_ft/model_best.pt`` — the warehouse-domain
         fine-tune. Becomes the default automatically WHEN its weights exist.
      3. ``runs/nx6_heatmap_B/model_best.pt`` — the original playground-trained
         baseline, the fallback when no fine-tune is present (e.g. a fresh clone).

    Args:
        ft_ckpt: Fine-tune checkpoint path (overridable for testing).
        orig_ckpt: Original baseline checkpoint path (overridable for testing).
        env_var: Environment variable consulted first (overridable for testing).

    Returns:
        The resolved checkpoint path.
    """
    env = os.environ.get(env_var)
    if env:
        return env
    if os.path.exists(ft_ckpt):
        return ft_ckpt
    return orig_ckpt


# Snapshot at import for logging / back-compat; load_shared_detector re-resolves
# at call time so a fine-tune produced after import is still picked up.
_CKPT_DEFAULT = resolve_ckpt_path()

# Process-wide shared detector cache: loaded once, the same model object handed
# to every per-robot GroundNetState (see load_shared_detector).
_SHARED_DETECTOR = None            # type: object | None
_SHARED_LOAD_TRIED = False
_SHARED_CLASS_NAMES = None         # type: list | None
_SHARED_COLOR_NAMES = None         # type: list | None


def detection_world_xy(pelvis_xy: XY, yaw: float, dist_m: float,
                       bearing_deg: float) -> XY:
    """Convert an egocentric detection to a world-frame ``(x, y)`` estimate.

    GROUND_NET reports ``dist_m`` relative to the robot's pelvis origin and
    ``bearing_deg`` as a yaw error (positive = target to the LEFT / CCW of the
    robot's heading, per :func:`code.perception.geometry.cam_to_egocentric`).
    The world bearing is therefore ``yaw + radians(bearing_deg)``.

    Args:
        pelvis_xy: Robot pelvis world ``(x, y)`` (m).
        yaw: Robot heading (rad).
        dist_m: Detector range to target from the pelvis origin (m).
        bearing_deg: Detector yaw error (deg, +CCW/left).

    Returns:
        The detector's world-frame ``(x, y)`` estimate (m).
    """
    world_bearing = yaw + math.radians(bearing_deg)
    return (float(pelvis_xy[0] + dist_m * math.cos(world_bearing)),
            float(pelvis_xy[1] + dist_m * math.sin(world_bearing)))


def oracle_range_bearing(robot_xy: XY, yaw: float, obj_xy: XY) -> Tuple[float, float]:
    """Return the oracle hypothesis ``(dist_m, bearing_deg)`` for an object.

    Args:
        robot_xy: Robot pelvis world ``(x, y)`` (m).
        yaw: Robot heading (rad).
        obj_xy: Object world ``(x, y)`` (m).

    Returns:
        ``(dist_m, bearing_deg)`` — straight-line range and the signed yaw error
        (deg, +CCW/left) of the object relative to the robot's heading.
    """
    dx, dy = obj_xy[0] - robot_xy[0], obj_xy[1] - robot_xy[1]
    dist = math.hypot(dx, dy)
    bearing = math.atan2(dy, dx) - yaw
    bearing = (bearing + math.pi) % (2.0 * math.pi) - math.pi
    return dist, math.degrees(bearing)


@dataclass
class DetectionResult:
    """One accepted GROUND_NET confirmation (or classical-HSV fallback).

    Returned by :meth:`RobotPerception.confirm` only when the detection clears
    the confidence floor (and, when an ``oracle_xy`` hypothesis is supplied, the
    geometry-consistency gate). ``world_xy`` is the DETECTOR's own estimate — the
    caller reports this as the found location, not the oracle's exact xy.
    """

    world_xy: XY                       # detector's world-frame (x, y) estimate
    dist_m: float                      # detector range (m, pelvis origin)
    bearing_deg: float                 # detector yaw error (deg, +CCW/left)
    confidence: float                  # raw peak sigmoid confidence [0, 1]
    source: str                        # "groundnet" | "hsv"
    callsign: str = ""                 # confirming robot
    query_desc: str = ""               # e.g. "red cube"
    heatmap: Optional[np.ndarray] = field(default=None, repr=False)   # (144,192) [0,1]
    cam_rgb: Optional[np.ndarray] = field(default=None, repr=False)   # grounding frame

    def caption(self) -> str:
        """ASCII one-line video caption, e.g. ``GROUND_NET: red cube 2.3m @ -14deg (conf 0.87)``."""
        tag = "GROUND_NET" if self.source == "groundnet" else "HSV-FALLBACK"
        return (f"{tag}: {self.query_desc} {self.dist_m:.1f}m "
                f"@ {self.bearing_deg:+.0f}deg (conf {self.confidence:.2f})")


def load_shared_detector(ckpt_path: Optional[str] = None,
                         device: str = _DEVICE_DEFAULT):
    """Load the GROUND_NET checkpoint once per process; share the model object.

    The same :class:`~code.perception.detector.model.HeatmapDetector` instance is
    returned on every call (sticky ``None`` if the first load failed — e.g. the
    deploy repo ships no weights), so it can be assigned into every per-robot
    :class:`~code.perception.ground_net.GroundNetState`'s ``.detector`` field.

    Args:
        ckpt_path: Explicit checkpoint path; when ``None`` (the default) the path
            is resolved via :func:`resolve_ckpt_path` (env var > warehouse
            fine-tune > original baseline) at call time.
        device: Torch device string ("cuda" falls back to "cpu" if unavailable).

    Returns:
        The shared ``HeatmapDetector``, or ``None`` if loading failed.
    """
    global _SHARED_DETECTOR, _SHARED_LOAD_TRIED, _SHARED_CLASS_NAMES, _SHARED_COLOR_NAMES
    if _SHARED_LOAD_TRIED:
        return _SHARED_DETECTOR
    _SHARED_LOAD_TRIED = True
    if ckpt_path is None:
        ckpt_path = resolve_ckpt_path()
    try:
        import torch
        from code.perception.detector.model import (CLASS_NAMES, COLOR_NAMES,
                                                     HeatmapDetector)
        dev = device if (device == "cpu" or torch.cuda.is_available()) else "cpu"
        _SHARED_DETECTOR = HeatmapDetector.load(ckpt_path, device=dev)
        _SHARED_CLASS_NAMES = CLASS_NAMES
        _SHARED_COLOR_NAMES = COLOR_NAMES
        print(f"[perception] GROUND_NET shared detector loaded {ckpt_path!r} "
              f"on device={dev!r}", flush=True)
    except Exception as e:  # noqa: BLE001 - graceful classical fallback
        _SHARED_DETECTOR = None
        print(f"[perception] GROUND_NET checkpoint unavailable ({e!r}); "
              f"RobotPerception falls back to the classical HSV+depth pipeline.",
              flush=True)
    return _SHARED_DETECTOR


class GroundingCamRenderer:
    """Renders the 480x360 26-degree grounding camera from a shared viz model.

    One instance per fleet (the render is stateless per call, so it is safe to
    share across every robot's :class:`RobotPerception`): it holds a single
    ``mujoco.Renderer`` bound to the shared viz model, keeping EGL-context use to
    +1 for the whole fleet. Mirrors
    :meth:`code.sim.arena_render.ArenaRenderer.render_grounding` but against the
    fleet's shared kinematic model so other robots + the carried object are in
    view exactly as they physically are this step.
    """

    def __init__(self, model: mujoco.MjModel) -> None:
        """Allocate the grounding renderer + camera for ``model``."""
        self._rend = mujoco.Renderer(model, GROUNDING_H, GROUNDING_W)
        self._cam = mujoco.MjvCamera()
        self._cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        self._intr = get_ego_intrinsics_rendered(GROUNDING_W, GROUNDING_H)
        self._intr["pitch_deg"] = GROUNDING_PITCH
        self._intr["is_proximity"] = False
        self._intr["is_widefov"] = False

    def render(self, data: mujoco.MjData, pelvis_xyz, yaw: float
               ) -> Tuple[np.ndarray, np.ndarray, dict]:
        """Render RGB + metric depth from the grounding cam at a robot's pose.

        Args:
            data: The (already-synced) shared viz ``MjData``.
            pelvis_xyz: Robot pelvis world ``(x, y, z)`` — camera mount origin.
            yaw: Robot heading (rad).

        Returns:
            ``(rgb, depth, intrinsics)``: ``rgb`` (360,480,3) uint8, ``depth``
            (360,480) float32 metres, and the grounding intrinsics dict.
        """
        _set_ego_cam(self._cam, np.asarray(pelvis_xyz, dtype=float), yaw,
                     pitch_deg=GROUNDING_PITCH)
        self._rend.update_scene(data, self._cam)
        rgb = self._rend.render().copy()
        self._rend.enable_depth_rendering()
        self._rend.update_scene(data, self._cam)
        depth = self._rend.render().copy().astype(np.float32)
        self._rend.disable_depth_rendering()
        return rgb, depth, dict(self._intr)

    def close(self) -> None:
        """Release the underlying EGL renderer."""
        self._rend.close()


class RobotPerception:
    """One robot's GROUND_NET confirmer over the shared viz model.

    Owns a per-robot :class:`~code.perception.ground_net.GroundNetState` (isolated
    track hysteresis + heatmap cache); shares the detector weights and the
    grounding renderer across the fleet.
    """

    def __init__(self, callsign: str, viz, *, detector=None, renderer=None,
                 tau: float = CONFIRM_TAU, hysteresis: bool = False,
                 tau_track: Optional[float] = None,
                 class_names=None, color_names=None) -> None:
        """Bind one robot's confirmer.

        Args:
            callsign: This robot's name (for pelvis lookup + labelling).
            viz: The shared :class:`~code.fleet.viz.FleetViz` (rendered from).
            detector: Shared ``HeatmapDetector`` (see :func:`load_shared_detector`),
                or ``None`` to use the classical-HSV fallback.
            renderer: Shared :class:`GroundingCamRenderer` (one per fleet).
            tau: Confidence floor for a confirmation.
            hysteresis: Enable NX-7 acquire/track hysteresis (per-robot state).
            tau_track: Lower confidence for a hysteresis track-continuation
                (defaults to ``tau``, making the track path inert).
            class_names: Detector class vocabulary (defaults to the shared load's).
            color_names: Detector color vocabulary (defaults to the shared load's).
        """
        self.callsign = callsign
        self._viz = viz
        self._renderer = renderer
        self._tau = float(tau)
        self._tau_track = float(tau_track) if tau_track is not None else float(tau)
        self._hysteresis = bool(hysteresis)
        # Per-robot mutable backend state — the anti-singleton isolation.
        self._state = _gn.GroundNetState()
        if detector is not None:
            self._state.detector = detector
            self._state.class_names = class_names or _SHARED_CLASS_NAMES
            self._state.color_names = color_names or _SHARED_COLOR_NAMES
        # Most-recent accepted confirmation (for the video overlay) + a pop-once
        # slot the mission loop drains to timestamp events.
        self.last_confirmation: Optional[DetectionResult] = None
        self._pending: Optional[DetectionResult] = None

    @property
    def has_detector(self) -> bool:
        """Whether the learned detector is loaded (vs. HSV fallback)."""
        return self._state.detector is not None

    def reset(self) -> None:
        """Clear per-mission state: track hysteresis + cached confirmations.

        Must be called at the start of every mission so one mission's last
        detection cannot validate an unrelated blob at the next mission's start
        (the singleton track-leak bug, guarded per-robot here).
        """
        _gn.reset_track(self._state)
        self._state.last_heatmap = None
        self.last_confirmation = None
        self._pending = None

    def pop_confirmation(self) -> Optional[DetectionResult]:
        """Return and clear the confirmation produced since the last pop (or None)."""
        ev, self._pending = self._pending, None
        return ev

    def confirm(self, query: ObjectQuery, robot_pose: Tuple[float, float, float],
                viz=None, *, oracle_xy: Optional[XY] = None
                ) -> Optional[DetectionResult]:
        """Run GROUND_NET to confirm a sighting of ``query`` at ``robot_pose``.

        Renders the robot's grounding camera from the shared viz model at its
        pose, runs the learned detector (or the classical-HSV fallback) for the
        query's ``(shape, color)``, converts ``(dist, bearing, confidence)`` to a
        world-frame ``(x, y)`` estimate, and returns it only when the detection
        clears the confidence floor — and, if ``oracle_xy`` is given, is
        geometrically consistent with that occlusion-truth hypothesis.

        Args:
            query: The colour/shape referent to confirm.
            robot_pose: Robot pose ``(x, y, yaw)`` (world m / rad).
            viz: Shared viz to render from (defaults to the one bound at init).
            oracle_xy: The oracle's proposed object ``(x, y)``. When supplied the
                confirmation additionally requires the detector's range+bearing to
                match it (the mission-loop confirmer passes this); when ``None``
                the raw detector decision is returned (the perception eval path).

        Returns:
            A :class:`DetectionResult` when confirmed, else ``None``.
        """
        viz = viz or self._viz
        rx, ry, yaw = float(robot_pose[0]), float(robot_pose[1]), float(robot_pose[2])
        color = (query.color_name or "").lower().strip()
        shape = (query.shape_name or "").lower().strip()
        if not color or not shape:
            return None  # detector needs a fully specified (shape, color) query

        pelvis = viz.pelvis_xpos(self.callsign)
        pelvis_xyz = (rx, ry, float(pelvis[2]))
        rgb, depth, intr = self._renderer.render(viz.data, pelvis_xyz, yaw)

        if self.has_detector:
            # _gn.infer already applies the tau-acquire / tau-track hysteresis
            # gate; trust its not_visible decision (re-thresholding here would
            # defeat the NX-7 track-continuation path at tau_track <= conf < tau).
            res = _gn.infer(self._state, rgb, depth, color, shape, intr,
                            tau=self._tau, hysteresis=self._hysteresis,
                            tau_track=self._tau_track)
            heat = (self._state.last_heatmap or {}).get("prob")
            source = "groundnet"
            conf = float(res.confidence)
            accepted = not res.not_visible
        else:
            from code.perception.hsv_pipeline import ground_classical
            res = ground_classical(rgb, depth, color, shape, intr)
            heat = None
            source = "hsv"
            conf = float(res.confidence)
            accepted = (not res.not_visible) and conf >= self._tau

        if not accepted:
            return None
        dist = float(res.dist)
        bearing_deg = math.degrees(math.atan2(res.sin_th, res.cos_th))

        if oracle_xy is not None:
            o_dist, o_bear = oracle_range_bearing((rx, ry), yaw, oracle_xy)
            if abs(dist - o_dist) > CONFIRM_DIST_GATE_M:
                return None
            d_bear = abs((bearing_deg - o_bear + 180.0) % 360.0 - 180.0)
            if d_bear > CONFIRM_BEARING_GATE_DEG:
                return None

        world_xy = detection_world_xy((rx, ry), yaw, dist, bearing_deg)
        det = DetectionResult(
            world_xy=world_xy, dist_m=dist, bearing_deg=bearing_deg,
            confidence=conf, source=source, callsign=self.callsign,
            query_desc=query.describe(),
            heatmap=(np.asarray(heat, dtype=np.float32).copy()
                     if heat is not None else None),
            cam_rgb=rgb)
        self.last_confirmation = det
        self._pending = det
        return det
