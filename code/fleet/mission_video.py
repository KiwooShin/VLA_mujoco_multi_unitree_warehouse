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
from code.comms.messages import Performative
from code.fleet.mission import MissionRunner
from code.fleet.viz import BEV_H, BEV_W
from code.sim.arena_build import COLORS
from code.warehouse.layout import (WarehouseLayout, callsigns_for_layout,
                                   hero_layout, rooms6_layout, rooms_layout)

Point = Tuple[float, float]
_DEFAULT_OUT = str(_REPO / "ops" / "phase4")
_CMAP = dict(COLORS)
_LAYOUTS = {"hero": hero_layout, "rooms": rooms_layout, "rooms6": rooms6_layout}
# Seconds of simulated time per control step (50 Hz control loop) — drives the
# "sim time" HUD readout.
_SIM_DT: float = 0.02

# Per-sender overlay colours in BGR (cv2 order).
_SENDER_BGR: Dict[str, Tuple[int, int, int]] = {
    "Alpha": (40, 40, 224), "Bravo": (224, 90, 40),
    "Charlie": (40, 205, 238), "Delta": (210, 60, 158),
    "Echo": (189, 204, 26), "Foxtrot": (26, 138, 247),  # teal, orange (F6)
    "user": (240, 240, 240), "allocator": (150, 230, 150),
}
_PANEL_W: int = 360
# Detector-inset geometry + how long (sim steps) a confirmation inset lingers
# after its last frame (~2 s of video time at 50 Hz control).
_INSET_W: int = 264
_INSET_H: int = 198
_INSET_HOLD_STEPS: int = 100
# F1 — communicating-robot ego insets: size, and how long (sim steps ~ 2 s of
# video time at 50 Hz control) an exchange keeps a robot's inset lit before it
# fades out. Peer<->peer performatives that count as an active exchange.
_COMM_INSET_W: int = 208
_COMM_INSET_H: int = 156
_COMM_HOLD_STEPS: int = 100
_COMM_PERFS = (Performative.QUERY_VISIBILITY, Performative.REPORT_VISIBILITY,
               Performative.COMMAND_SEARCH, Performative.ACCEPT,
               Performative.REJECT, Performative.REPORT_FOUND)
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

# Target spot per (layout, scenario). Hero: A visible to Alpha, B visible to a
# peer only, C hidden from all (NE alcove) -> full delegated search, D uses the
# peer-visible spot for a fleet-addressed allocator demo. Rooms: C is the deep
# back-room corner (the room-to-room exploration story); F4 the generic-command
# demo (any object, first found wins).
_SCENARIO_SPOT: Dict[str, Dict[str, int]] = {
    "hero": {"A": 6, "B": 7, "C": 5, "D": 7, "F4": 7},
    "rooms": {"A": 4, "B": 5, "C": 8, "D": 5, "F4": 8},
    # rooms6 (six-robot): C is the deep back-room NE corner (spot 11) — hidden
    # from every bay, so all five idle peers are delegated a room to search
    # (three searchable rooms + two reserve) before the back-room searcher finds it.
    "rooms6": {"A": 0, "B": 1, "C": 11, "D": 6, "F4": 11},
}


def _build_objects(spot: int, layout: WarehouseLayout) -> List[dict]:
    """Red cube at ``spot`` + distinct fillers elsewhere, for any layout."""
    objs: List[dict] = []
    fi = 0
    for i, (x, y) in enumerate(layout.object_spots):
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

    return (f"{msg.performative.name}: "
            f"{_phrase(msg.performative, msg.payload, msg.sender)}")


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

    # F2: no target ring until a robot has actually located the object; then it
    # sits at the reported position, and rides the carried object once picked up.
    tgt = mr.known_target_xy()
    if tgt is not None:
        r = 12 + int(4 * abs(np.sin(t * 0.15)))  # gentle pulse
        bevmod.draw_marker(frame, cam, tgt, color=(60, 60, 255), radius=r, z=0.3)
        bevmod.draw_marker(frame, cam, tgt, color=(255, 255, 255), radius=r + 3,
                           z=0.3)

    hud = [f"sim time {t * _SIM_DT:5.1f} s   step {t}   {mr.phase()}"]
    if mr.task is not None:
        hud.append(f"task: fetch the {mr.task.query.describe()} -> {mr.task.destination_name}")
    # F4: pin the fleet allocator's decision so it stays visible for the whole
    # clip (the bus STATUS_UPDATE that speaks it scrolls off the 12-line panel
    # within the opening query burst).
    alloc = getattr(mr, "allocation", None)
    if alloc is not None:
        hud.append(f"allocator: {_ascii(alloc.describe())}")
    bevmod.put_hud(frame, hud)


def _recent_confirmation(mr: MissionRunner, t: int):
    """The most recent detector confirmation still within the linger window, or None."""
    best = None
    for step, det in mr.confirmations:
        if step <= t and (t - step) <= _INSET_HOLD_STEPS:
            best = (step, det)
    return best


def draw_detector_inset(frame: np.ndarray, det, step: int, t: int) -> None:
    """Overlay the grounding-cam frame + heatmap peak marker + caption (in-place).

    Shows what the learned detector actually saw when it confirmed: the 480x360
    grounding frame (downscaled), a translucent heatmap wash + a ring on the
    detector's peak pixel, a title bar and the ASCII caption line, plus a
    transcript-style GROUND_NET event line beneath it.
    """
    import cv2

    inset = cv2.resize(cv2.cvtColor(det.cam_rgb, cv2.COLOR_RGB2BGR),
                       (_INSET_W, _INSET_H), interpolation=cv2.INTER_AREA)
    if det.heatmap is not None:
        heat = np.clip(det.heatmap, 0.0, 1.0)
        heat_u8 = cv2.resize((heat * 255).astype(np.uint8), (_INSET_W, _INSET_H),
                             interpolation=cv2.INTER_LINEAR)
        cmap = cv2.applyColorMap(heat_u8, cv2.COLORMAP_JET)
        inset = cv2.addWeighted(inset, 0.7, cmap, 0.3, 0.0)
        py, px = np.unravel_index(int(np.argmax(heat)), heat.shape)
        mx = int(px / heat.shape[1] * _INSET_W)
        my = int(py / heat.shape[0] * _INSET_H)
        cv2.circle(inset, (mx, my), 9, (60, 255, 255), 2, cv2.LINE_AA)
        cv2.drawMarker(inset, (mx, my), (60, 255, 255), cv2.MARKER_CROSS, 14, 1)

    x0, y0 = 12, 12
    h, w = inset.shape[:2]
    cv2.rectangle(frame, (x0 - 3, y0 - 3), (x0 + w + 3, y0 + h + 3), (60, 255, 120), 2)
    frame[y0:y0 + h, x0:x0 + w] = inset
    title = f"GROUND_NET grounding cam ({det.callsign})"
    cv2.rectangle(frame, (x0 - 3, y0 - 22), (x0 + w + 3, y0 - 3), (30, 30, 30), -1)
    cv2.putText(frame, _ascii(title), (x0 + 2, y0 - 8), cv2.FONT_HERSHEY_SIMPLEX,
                0.42, (120, 255, 160), 1, cv2.LINE_AA)
    cap = _ascii(det.caption())
    cy = y0 + h + 18
    for th, col in ((3, (0, 0, 0)), (1, (140, 255, 170))):
        cv2.putText(frame, cap, (x0, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.46, col, th,
                    cv2.LINE_AA)
    evline = _ascii(f"[detector] {det.callsign} confirms {det.query_desc} @ "
                    f"({det.world_xy[0]:.1f},{det.world_xy[1]:.1f})  t={step}")
    cv2.putText(frame, evline, (x0, cy + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.40,
                (150, 230, 150), 1, cv2.LINE_AA)


def _active_comm_robots(mr: MissionRunner, t: int) -> Dict[str, float]:
    """Robots currently in a peer<->peer message exchange -> border freshness.

    F1: a robot lights up while it is exchanging QUERY/REPORT/COMMAND/ACCEPT
    (etc.) with a peer, fading over ``_COMM_HOLD_STEPS`` (~2 s of video) after
    the exchange's last message. Returns ``{callsign: freshness in (0, 1]}`` for
    both endpoints of every recent exchange; idle robots are absent (hidden).
    """
    robots = set(mr.callsigns)
    fresh: Dict[str, float] = {}
    for m in mr.bus.transcript:
        if m.performative not in _COMM_PERFS:
            continue
        if m.sender not in robots or m.recipient not in robots:
            continue  # only robot<->robot exchanges (skip user/allocator traffic)
        age = t - m.t_step
        if age < 0 or age > _COMM_HOLD_STEPS:
            continue
        f = 1.0 - age / _COMM_HOLD_STEPS
        for who in (m.sender, m.recipient):
            fresh[who] = max(fresh.get(who, 0.0), f)
    return fresh


def draw_comm_insets(frame: np.ndarray, viz, mr: MissionRunner,
                     active: Dict[str, float]) -> None:
    """Overlay ego-camera insets for the robots currently communicating (F1).

    Each active robot gets a small live ego render (from the shared viz model at
    its pose) with an EMPHASIZED, glowing border in its accent colour whose
    brightness/thickness tracks the exchange's freshness; the insets sit in a row
    along the bottom of the BEV. Non-communicating robots' views are hidden.
    """
    import cv2

    names = [cs for cs in mr.callsigns if cs in active]
    if not names:
        return
    h, w = frame.shape[:2]
    gap = 14
    # Scale the inset width down so the whole row fits within the frame even when
    # many robots communicate at once (up to six at N=6): a fixed 208 px inset ran
    # 6 wide off the 960 px BEV. Height tracks width to preserve the ego aspect.
    margin = 12
    avail = w - 2 * margin - (len(names) - 1) * gap
    iw = min(_COMM_INSET_W, max(96, avail // len(names)))
    ih = min(_COMM_INSET_H, int(round(iw * _COMM_INSET_H / _COMM_INSET_W)))
    total = len(names) * iw + (len(names) - 1) * gap
    x = max(margin, (w - total) // 2)
    y0 = h - ih - 34
    for cs in names:
        f = active[cs]
        unit = mr.fleet.units[cs]
        ego = viz.render_ego(cs, unit.yaw)
        inset = cv2.resize(cv2.cvtColor(ego, cv2.COLOR_RGB2BGR), (iw, ih),
                           interpolation=cv2.INTER_AREA)
        frame[y0:y0 + ih, x:x + iw] = inset
        accent = _SENDER_BGR.get(cs, (200, 200, 200))
        # Glow: concentric borders in the accent colour, brightest (thick) when
        # the exchange is fresh, fading outward and with age.
        for k in range(4, 0, -1):
            a = f * (k / 4.0)
            col = tuple(int(c * (0.35 + 0.65 * a)) for c in accent)
            pad = 3 + (4 - k) * 3
            cv2.rectangle(frame, (x - pad, y0 - pad),
                          (x + iw + pad, y0 + ih + pad), col, 2, cv2.LINE_AA)
        cv2.rectangle(frame, (x, y0), (x + iw, y0 + ih),
                      accent, max(2, int(2 + 3 * f)), cv2.LINE_AA)
        # Title chip.
        cv2.rectangle(frame, (x, y0 - 20), (x + iw, y0), (25, 25, 25), -1)
        label = f"{cs} ego  (in comms)"
        cv2.putText(frame, _ascii(label), (x + 6, y0 - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.44, accent, 1, cv2.LINE_AA)
        x += iw + gap


def record_mission_video(scenario: str, out_dir: str, *, decimation: int,
                         fps: int, max_steps: int, seed: int,
                         perception_mode: str = "oracle",
                         layout_name: str = "hero",
                         command: Optional[str] = None,
                         locomotion: str = "teacher",
                         vla_ckpt: Optional[str] = None,
                         vla_device: Optional[str] = None) -> Optional[str]:
    """Run a scenario mission and write the composited MP4. Returns its path."""
    import cv2

    os.makedirs(out_dir, exist_ok=True)
    layout = _LAYOUTS.get(layout_name, hero_layout)()
    spot = _SCENARIO_SPOT.get(layout_name, {}).get(scenario, 5)
    mr = MissionRunner(layout=layout, objects=_build_objects(spot, layout),
                       callsigns=callsigns_for_layout(layout),
                       seed=seed, use_gpu=True, search_deadline_steps=max_steps,
                       perception_mode=perception_mode, locomotion=locomotion,
                       vla_ckpt=vla_ckpt, vla_device=vla_device)
    viz = mr.fleet.viz
    assert viz is not None
    cam = bevmod.fit_bev_camera(mr.layout.hall_x, mr.layout.hall_y,
                                width=BEV_W, height=BEV_H,
                                fovy_deg=float(viz.model.vis.global_.fovy))
    if command is not None:
        mr.submit(command)
    elif scenario == "D":
        # Fleet-addressed order: the allocator (not the user) chooses the robot.
        mr.submit("someone bring me the red cube")
    else:
        mr.submit("Alpha, fetch the red cube to the delivery pad")

    frames: List[np.ndarray] = []

    def compose(runner: MissionRunner, t: int) -> np.ndarray:
        frame = np.ascontiguousarray(
            cv2.cvtColor(viz.render_bev(cam), cv2.COLOR_RGB2BGR))
        draw_overlay(frame, cam, runner, t)
        if perception_mode == "groundnet":
            recent = _recent_confirmation(runner, t)
            if recent is not None:
                draw_detector_inset(frame, recent[1], recent[0], t)
        draw_comm_insets(frame, viz, runner, _active_comm_robots(runner, t))  # F1
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
    ap.add_argument("--scenario", choices=("A", "B", "C", "D", "F4"), default="C")
    ap.add_argument("--layout", choices=tuple(_LAYOUTS), default="hero")
    ap.add_argument("--command", type=str, default=None,
                    help="override the submitted order (e.g. a generic F4 command)")
    ap.add_argument("--out", type=str, default=_DEFAULT_OUT)
    ap.add_argument("--decimation", type=int, default=4)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--max-steps", type=int, default=9000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--perception", choices=("oracle", "groundnet"),
                    default="oracle",
                    help="groundnet overlays live GROUND_NET detector insets")
    ap.add_argument("--locomotion", choices=("teacher", "vla"), default="teacher",
                    help="WBC walk policy (default) or the trained VLA policy (F5)")
    ap.add_argument("--ckpt", type=str, default=None,
                    help="GroundedNav checkpoint for --locomotion vla (default: F5 fine-tune)")
    ap.add_argument("--device", type=str, default=None,
                    help="Torch device for the VLA policy (cuda|cpu; default auto)")
    args = ap.parse_args(argv)
    record_mission_video(args.scenario, args.out, decimation=args.decimation,
                         fps=args.fps, max_steps=args.max_steps, seed=args.seed,
                         perception_mode=args.perception,
                         layout_name=args.layout, command=args.command,
                         locomotion=args.locomotion, vla_ckpt=args.ckpt,
                         vla_device=args.device)


if __name__ == "__main__":
    main()
