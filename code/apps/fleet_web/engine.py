"""engine.py — The MuJoCo/EGL sim engine behind the fleet web demo.

Isolates every MuJoCo/EGL/rendering concern behind :class:`SimEngine` so
:class:`code.apps.fleet_web.service.FleetService` can be exercised in tests with
a lightweight fake. The real :class:`MujocoFleetEngine` drives the *public*
surface of :class:`code.fleet.mission.MissionRunner` only:

* ``reset()`` builds ONE long-lived ``MissionRunner`` (reusing the four
  ``WBCTeacher`` walk policies); it is never rebuilt per mission,
* ``submit(text)`` starts a mission; ``run(max_steps, on_step)`` drives it to a
  terminal outcome while our ``on_step`` renders/paces and may return ``False``
  to cancel promptly (the public stop contract),
* ``reset_mission()`` clears the runner's per-mission state between orders (the
  lifecycle API), so successive orders run on the same fleet in a continuous
  world — no per-mission rebuild, no ``StopSim`` workaround,
* between missions the fleet is stepped via ``Fleet.step_all`` so idle robots
  stand in the live BEV.

No state in ``code/fleet`` is mutated beyond the public lifecycle calls; the
engine only reads public accessors and renders the shared viz model.
"""

from __future__ import annotations

import abc
import dataclasses
from typing import Callable, List, Optional, Sequence

from code.apps.fleet_web.status import RobotSnap


@dataclasses.dataclass(frozen=True)
class MsgSnap:
    """A minimal, rendered view of one bus message for the transcript log."""

    sender: str
    recipient: str
    text: str


class SimEngine(abc.ABC):
    """Interface the sim thread drives; real and fake backends implement it."""

    @property
    @abc.abstractmethod
    def callsigns(self) -> Sequence[str]:
        """The fleet's callsigns in priority order."""

    @abc.abstractmethod
    def reset(self) -> None:
        """Build the long-lived world so robots stand idle at their home bays."""

    def reset_mission(self) -> None:
        """Clear per-mission state so the next order runs on the same fleet.

        The lifecycle counterpart of :meth:`reset`: reuses the built world (no
        rebuild). The default is a no-op for trivial fakes that carry no state.
        """

    @abc.abstractmethod
    def submit(self, text: str) -> None:
        """Begin a mission from an already-validated order."""

    @abc.abstractmethod
    def run_mission(self, on_step: Callable[[int], object], max_steps: int) -> str:
        """Drive the active mission to a terminal outcome.

        Calls ``on_step(t)`` after each sim step; ``on_step`` returning ``False``
        cancels the run promptly (the public stop contract). Returns an outcome
        string: ``"complete"`` / ``"failed"`` / ``"timeout"`` / ``"stopped"``.
        """

    @abc.abstractmethod
    def idle_step(self) -> None:
        """Advance the sim one step while no mission is active."""

    @abc.abstractmethod
    def render_jpeg(self) -> Optional[bytes]:
        """Render + JPEG-encode the current BEV frame (``None`` if unavailable)."""

    @abc.abstractmethod
    def robots(self) -> List[RobotSnap]:
        """Per-robot raw snapshots in callsign order."""

    @abc.abstractmethod
    def bus_snaps(self) -> List[MsgSnap]:
        """The current runner's full transcript as rendered triples."""

    @abc.abstractmethod
    def mission_phase(self) -> str:
        """A short HUD phrase for what the fleet is doing right now."""

    @abc.abstractmethod
    def task_desc(self) -> str:
        """The active object phrase (e.g. ``"red cube"``), or ``""``."""

    @abc.abstractmethod
    def object_on_pad(self) -> bool:
        """Whether the requested object now rests on the delivery pad."""

    def close(self) -> None:
        """Release any resources (default: no-op)."""


# Per-callsign overlay colours in BGR (cv2 order), matching the torso accents.
_ACCENT_BGR = {
    "Alpha": (58, 57, 230), "Bravo": (213, 123, 58),
    "Charlie": (58, 195, 232), "Delta": (208, 79, 158),
}
_JPEG_QUALITY = 80


class MujocoFleetEngine(SimEngine):
    """The real EGL-backed engine over one long-lived ``MissionRunner``."""

    def __init__(self, *, seed: int = 0, use_gpu: bool = True,
                 layout_name: str = "hero", locomotion: str = "teacher",
                 vla_ckpt: Optional[str] = None,
                 vla_device: Optional[str] = None) -> None:
        """Configure the engine (no MuJoCo work until :meth:`reset`).

        Args:
            seed: RNG seed (curated objects are used, so this only affects
                incidental placement); fixed for a reproducible demo scene.
            use_gpu: Prefer CUDA for the walk policies.
            layout_name: Warehouse layout ("hero" is the only tuned layout).
            locomotion: ``"teacher"`` (default; WBC walk policy) or ``"vla"``
                (F5: the trained GroundedNav policy, one model shared by the
                fleet) — passed straight through to the ``MissionRunner``.
            vla_ckpt: GroundedNav checkpoint for ``locomotion="vla"`` (None -> F5
                default).
            vla_device: Torch device for the VLA policy (None -> auto).
        """
        self._seed = int(seed)
        self._use_gpu = bool(use_gpu)
        self._layout_name = layout_name
        self._locomotion = locomotion
        self._vla_ckpt = vla_ckpt
        self._vla_device = vla_device
        self._teachers = None
        self._runner = None
        self._cam = None
        self._layout = None
        self._pulse = 0

    # -- construction -----------------------------------------------------
    @property
    def callsigns(self) -> Sequence[str]:
        from code.warehouse.layout import CALLSIGNS

        return CALLSIGNS

    def _build_objects(self) -> List[dict]:
        """Curated, distinctly-coloured objects (one per hero spot)."""
        from code.sim.arena_build import COLORS

        cmap = dict(COLORS)
        palette = [("blue", "cube"), ("green", "cylinder"), ("yellow", "ball"),
                   ("purple", "cube"), ("orange", "cone"), ("cyan", "cylinder"),
                   ("red", "cube"), ("blue", "ball")]
        objs: List[dict] = []
        for (x, y), (color, shape) in zip(self._layout.object_spots, palette):
            objs.append({"color_name": color, "color_rgb": cmap[color],
                         "shape_name": shape, "size": 0.24,
                         "x": float(x), "y": float(y)})
        return objs

    def reset(self) -> None:
        """Build teachers + ONE long-lived idle ``MissionRunner`` (once)."""
        from code.apps.warehouse_demo import bev as bevmod
        from code.fleet.mission import MissionRunner
        from code.fleet.viz import BEV_H, BEV_W
        from code.sim.teacher import WBCTeacher
        from code.warehouse.layout import CALLSIGNS, hero_layout

        if self._layout is None:
            self._layout = hero_layout()
        if self._teachers is None:
            self._teachers = {cs: WBCTeacher(use_gpu=self._use_gpu)
                              for cs in CALLSIGNS}
        old = self._runner
        self._runner = MissionRunner(
            layout=self._layout, objects=self._build_objects(),
            teachers=self._teachers, seed=self._seed, use_gpu=self._use_gpu,
            locomotion=self._locomotion, vla_ckpt=self._vla_ckpt,
            vla_device=self._vla_device)
        if old is not None:
            old.close()
        viz = self._runner.fleet.viz
        self._cam = bevmod.fit_bev_camera(
            self._layout.hall_x, self._layout.hall_y, width=BEV_W, height=BEV_H,
            fovy_deg=float(viz.model.vis.global_.fovy))

    def reset_mission(self) -> None:
        """Clear the runner's per-mission state for the next order (lifecycle API)."""
        if self._runner is not None:
            self._runner.reset_mission()

    # -- mission ----------------------------------------------------------
    def submit(self, text: str) -> None:
        self._runner.submit(text)

    def run_mission(self, on_step: Callable[[int], object], max_steps: int) -> str:
        def _hook(_runner, t: int) -> object:
            return on_step(t)  # returning False cancels the run (stop contract)

        return self._runner.run(max_steps, on_step=_hook).outcome

    def idle_step(self) -> None:
        self._runner.fleet.step_all()

    # -- snapshots --------------------------------------------------------
    def render_jpeg(self) -> Optional[bytes]:
        if self._runner is None or self._runner.fleet.viz is None:
            return None
        import cv2
        import numpy as np

        from code.apps.warehouse_demo import bev as bevmod

        viz = self._runner.fleet.viz
        rgb = viz.render_bev(self._cam)
        frame = np.ascontiguousarray(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        self._draw_overlay(frame, bevmod, cv2, np)
        ok, buf = cv2.imencode(".jpg", frame,
                               [cv2.IMWRITE_JPEG_QUALITY, _JPEG_QUALITY])
        return buf.tobytes() if ok else None

    def _draw_overlay(self, frame, bevmod, cv2, np) -> None:
        """Draw trails, robots, labels, the target ring and a phase HUD."""
        mr = self._runner
        cam = self._cam
        for name in mr.callsigns:
            unit = mr.fleet.units[name]
            color = _ACCENT_BGR.get(name, (200, 200, 200))
            trail = mr.trails.get(name, [])
            if len(trail) >= 2:
                stride = max(1, len(trail) // 400)
                pts = trail[::stride]
                if pts[-1] != trail[-1]:
                    pts.append(trail[-1])
                bevmod.draw_polyline(frame, cam, pts, color, thickness=2, z=0.02)
            bevmod.draw_robot(frame, cam, unit.xy, unit.yaw, color=color)
            u, v = cam.project_xy(unit.xy, z=1.7)
            for th, col in ((4, (0, 0, 0)), (1, color)):
                cv2.putText(frame, name, (u - 22, v), cv2.FONT_HERSHEY_SIMPLEX,
                            0.55, col, th, cv2.LINE_AA)
        self._pulse += 1
        tgt = mr.known_target_xy()  # F2: only once a robot has located the object
        if tgt is not None:
            r = 12 + int(4 * abs(np.sin(self._pulse * 0.15)))
            bevmod.draw_marker(frame, cam, tgt, color=(60, 60, 255), radius=r,
                               z=0.3)
            bevmod.draw_marker(frame, cam, tgt, color=(255, 255, 255),
                               radius=r + 3, z=0.3)
        hud = [f"FLEET BEV   {mr.phase()}"]
        if mr.task is not None:
            hud.append(f"task: fetch the {mr.task.query.describe()} "
                       f"-> {mr.task.destination_name}")
        bevmod.put_hud(frame, hud)

    def robots(self) -> List[RobotSnap]:
        mr = self._runner
        owner = mr.primary_owner
        tdesc = self.task_desc()
        out: List[RobotSnap] = []
        for cs in mr.callsigns:
            unit = mr.fleet.units[cs]
            proto = mr.protocols[cs]
            dist = unit.distance_to_goal()
            out.append(RobotSnap(
                name=cs, coord_state=proto.state.name, motion=unit.state.value,
                dist_to_goal=(dist if dist != float("inf") else None),
                carrying=mr.carry.carrying(cs), is_owner=(cs == owner),
                task_desc=tdesc if cs == owner else ""))
        return out

    def bus_snaps(self) -> List[MsgSnap]:
        from code.comms.bus import format_line

        snaps: List[MsgSnap] = []
        for m in self._runner.bus.transcript:
            line = format_line(m)
            prefix = f"t={m.t_step} {m.sender}->{m.recipient} "
            text = line[len(prefix):] if line.startswith(prefix) else line
            snaps.append(MsgSnap(m.sender, m.recipient, text))
        return snaps

    def mission_phase(self) -> str:
        return self._runner.phase() if self._runner is not None else "STANDING BY"

    def task_desc(self) -> str:
        if self._runner is None or self._runner.task is None:
            return ""
        return self._runner.task.query.describe()

    def object_on_pad(self) -> bool:
        return bool(self._runner is not None and self._runner.object_on_pad())

    def close(self) -> None:
        if self._runner is not None:
            self._runner.close()
            self._runner = None
