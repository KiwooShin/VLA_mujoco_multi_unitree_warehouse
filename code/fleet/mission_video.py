"""mission_video.py — Flagship collaborative-fetch mission video.

Records the full canonical Phase-4 story to an MP4: the whole-hall bird's-eye
view (drawn from the shared :class:`~code.fleet.viz.FleetViz` model) with, per
robot, its accent-coloured trail, heading and name; a pulsing highlight ring on
the requested object (which visibly rides in the finder/owner's hand while
carried); a phase HUD ("QUERYING PEERS", "SEARCHING north", "CARRYING TO PAD"...);
and a right-hand panel that streams the live comms transcript, each line coloured
by its sender, exactly as the robots speak. Output lands in ``ops/phase4/``.

Usage
-----
PYTHONPATH=. MUJOCO_GL=egl python code/fleet/mission_video.py \\
    --scenario C --out ops/phase4 --decimation 4
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import numpy as np

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from code.apps.warehouse_demo import bev as bevmod
from code.fleet.mission import MissionRunner
from code.fleet.viz import BEV_H, BEV_W
from code.sim.arena_build import COLORS
from code.warehouse.layout import hero_layout

Point = Tuple[float, float]
_DEFAULT_OUT = str(_REPO / "ops" / "phase4")
_CMAP = dict(COLORS)

# Per-sender overlay colours in BGR (cv2 order).
_SENDER_BGR: Dict[str, Tuple[int, int, int]] = {
    "Alpha": (40, 40, 224), "Bravo": (224, 90, 40),
    "Charlie": (40, 205, 238), "Delta": (210, 60, 158),
    "user": (240, 240, 240), "allocator": (150, 230, 150),
}
_PANEL_W: int = 360
# Max polyline points drawn per robot trail per frame. Trails grow by one point
# per control step, so drawing every point made overlay time grow linearly per
# frame (quadratic per mission — profiled at ~87% of render wall time); striding
# to a bounded point count keeps the whole path visible at constant cost.
_TRAIL_MAX_PTS: int = 400

# cv2's Hershey fonts draw non-ASCII glyphs as "?", so panel text is mapped to
# ASCII before drawing (the code/comms formatter itself is untouched).
_ASCII_MAP: Dict[str, str] = {
    "—": "-", "–": "-",          # em/en dash
    "‘": "'", "’": "'",          # curly single quotes
    "“": '"', "”": '"',          # curly double quotes
    "→": "->", "…": "...",       # arrow, ellipsis
}
_FILLER = (("orange", "cube", 0.24), ("blue", "cylinder", 0.22),
           ("green", "ball", 0.24), ("yellow", "cone", 0.26),
           ("purple", "cube", 0.24), ("cyan", "cylinder", 0.22),
           ("blue", "ball", 0.24))

# Target spot per scenario (hero layout): A visible to Alpha, B visible to a
# peer only, C hidden from all (NE alcove) -> full delegated search.
_SCENARIO_SPOT: Dict[str, int] = {"A": 6, "B": 7, "C": 5}


def _build_objects(spot: int) -> List[dict]:
    """Red cube at ``spot`` + distinct fillers elsewhere (hero layout)."""
    objs: List[dict] = []
    fi = 0
    for i, (x, y) in enumerate(hero_layout().object_spots):
        if i == spot:
            c, s, sz = "red", "cube", 0.24
        else:
            c, s, sz = _FILLER[fi % len(_FILLER)]
            fi += 1
        objs.append({"color_name": c, "color_rgb": _CMAP[c], "shape_name": s,
                     "size": float(sz), "x": float(x), "y": float(y)})
    return objs


def _ascii(text: str) -> str:
    """Map known typographic characters to ASCII and drop the rest.

    cv2 cannot rasterize non-ASCII text (each such glyph renders as ``?``), so
    every panel line passes through this before ``putText``. Known punctuation
    is replaced with a readable equivalent; anything else non-ASCII is dropped.
    """
    for src, dst in _ASCII_MAP.items():
        text = text.replace(src, dst)
    return text.encode("ascii", errors="ignore").decode("ascii")


def _wrap(text: str, width_chars: int) -> List[str]:
    """Greedy word-wrap ``text`` to ``width_chars`` per line."""
    words = text.split()
    lines: List[str] = []
    cur = ""
    for w in words:
        if cur and len(cur) + 1 + len(w) > width_chars:
            lines.append(cur)
            cur = w
        else:
            cur = f"{cur} {w}".strip()
    if cur:
        lines.append(cur)
    return lines or [""]


def render_transcript_panel(mr: MissionRunner, height: int, *,
                            width: int = _PANEL_W, last_n: int = 12) -> np.ndarray:
    """Render the recent comms transcript as a sender-coloured side panel."""
    import cv2

    panel = np.full((height, width, 3), 24, dtype=np.uint8)
    cv2.putText(panel, "COMMS", (12, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (235, 235, 235), 1, cv2.LINE_AA)
    cv2.line(panel, (10, 36), (width - 10, 36), (80, 80, 80), 1)

    msgs = mr.bus.transcript[-last_n:]
    y = 60
    for m in msgs:
        color = _SENDER_BGR.get(m.sender, (200, 200, 200))
        head = _ascii(f"{m.sender}->{m.recipient}")
        phrase = _ascii(_line_phrase(m))
        for k, line in enumerate([head] + _wrap(phrase, 40)):
            if y > height - 10:
                break
            org = (14 if k == 0 else 22, y)
            scale = 0.42 if k == 0 else 0.40
            cv2.putText(panel, line, org, cv2.FONT_HERSHEY_SIMPLEX, scale,
                        color if k == 0 else (205, 205, 205), 1, cv2.LINE_AA)
            y += 16
        y += 4
    return panel


def _line_phrase(msg) -> str:
    """The human phrase half of a transcript line (without the t=/sender prefix)."""
    from code.comms.bus import _phrase

    return f"{msg.performative.name}: {_phrase(msg.performative, msg.payload)}"


def draw_overlay(frame: np.ndarray, cam: bevmod.BevCamera, mr: MissionRunner,
                 t: int) -> None:
    """Draw trails, robots, labels, the target ring and the phase HUD (in-place)."""
    import cv2

    for name in mr.callsigns:
        unit = mr.fleet.units[name]
        color = _SENDER_BGR.get(name, (200, 200, 200))
        trail = mr.trails[name]
        if len(trail) >= 2:
            # Stride the trail to a bounded point count (constant per-frame
            # cost; keeps the last point so the trail always meets the robot).
            stride = max(1, len(trail) // _TRAIL_MAX_PTS)
            pts = trail[::stride]
            if pts[-1] != trail[-1]:
                pts.append(trail[-1])
            bevmod.draw_polyline(frame, cam, pts, color, thickness=2, z=0.02)
        bevmod.draw_robot(frame, cam, unit.xy, unit.yaw, color=color)
        u, v = cam.project_xy(unit.xy, z=1.7)
        for th, col in ((4, (0, 0, 0)), (1, color)):
            cv2.putText(frame, name, (u - 22, v), cv2.FONT_HERSHEY_SIMPLEX,
                        0.55, col, th, cv2.LINE_AA)

    tgt = mr.target_xy()
    if tgt is not None:
        r = 12 + int(4 * abs(np.sin(t * 0.15)))  # gentle pulse
        bevmod.draw_marker(frame, cam, tgt, color=(60, 60, 255), radius=r, z=0.3)
        bevmod.draw_marker(frame, cam, tgt, color=(255, 255, 255), radius=r + 3,
                           z=0.3)

    hud = [f"step {t}   {mr.phase()}"]
    if mr.task is not None:
        hud.append(f"task: fetch the {mr.task.query.describe()} -> {mr.task.destination_name}")
    bevmod.put_hud(frame, hud)


def record_mission_video(scenario: str, out_dir: str, *, decimation: int,
                         fps: int, max_steps: int, seed: int) -> Optional[str]:
    """Run a scenario mission and write the composited MP4. Returns its path."""
    import cv2

    os.makedirs(out_dir, exist_ok=True)
    spot = _SCENARIO_SPOT.get(scenario, 5)
    mr = MissionRunner(layout=hero_layout(), objects=_build_objects(spot),
                       seed=seed, use_gpu=True, search_deadline_steps=max_steps)
    viz = mr.fleet.viz
    assert viz is not None
    cam = bevmod.fit_bev_camera(mr.layout.hall_x, mr.layout.hall_y,
                                width=BEV_W, height=BEV_H,
                                fovy_deg=float(viz.model.vis.global_.fovy))
    mr.submit("Alpha, fetch the red cube to the delivery pad")

    frames: List[np.ndarray] = []

    def compose(runner: MissionRunner, t: int) -> np.ndarray:
        frame = np.ascontiguousarray(
            cv2.cvtColor(viz.render_bev(cam), cv2.COLOR_RGB2BGR))
        draw_overlay(frame, cam, runner, t)
        panel = render_transcript_panel(runner, frame.shape[0])
        return np.hstack([frame, panel])

    def on_step(runner: MissionRunner, t: int) -> None:
        if t % decimation == 0:
            frames.append(compose(runner, t))

    mr.run(max_steps, on_step=on_step)
    frames.append(compose(mr, mr._steps))  # final state incl. TASK_COMPLETE line
    if frames:  # hold the final frame so the ending reads
        frames.extend([frames[-1]] * fps)

    path = None
    if frames:
        path = os.path.join(out_dir, f"mission_{scenario}.mp4")
        h, w = frames[0].shape[:2]
        writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
        for f in frames:
            writer.write(f)
        writer.release()
    print(f"[mission-video] {path}  frames={len(frames)}  steps={mr._steps}  "
          f"outcome={mr._result().outcome}  on_pad={mr.object_on_pad()}  "
          f"fell={mr.fleet.any_fell}", flush=True)
    mr.close()
    return path


def main(argv: Optional[List[str]] = None) -> None:
    ap = argparse.ArgumentParser(description="Flagship collaborative-fetch video")
    ap.add_argument("--scenario", choices=("A", "B", "C"), default="C")
    ap.add_argument("--out", type=str, default=_DEFAULT_OUT)
    ap.add_argument("--decimation", type=int, default=4)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--max-steps", type=int, default=9000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    record_mission_video(args.scenario, args.out, decimation=args.decimation,
                         fps=args.fps, max_steps=args.max_steps, seed=args.seed)


if __name__ == "__main__":
    main()
