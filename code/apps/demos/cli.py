"""cli.py — Render a Demo Set v2 sample clip end-to-end.

Builds a MissionRunner (oracle perception + teacher locomotion), composes every
frame with :class:`DemoComposer`, records an mp4 via :class:`DemoRecorder`, and
optionally a palette GIF. Used to produce ``ops/demo2/sample.mp4``.

Usage
-----
PYTHONPATH=. MUJOCO_GL=egl python -m code.apps.demos.cli \\
    --out ops/demo2/sample.mp4 --layout rooms --steps 900 --decimation 4 --gif
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Optional

import code.apps.demos  # noqa: F401 - triggers code/__init__ EGL vendor pin
from code.apps.warehouse_datagen.egl_gpu import force_nvidia_egl

force_nvidia_egl()  # standalone production run: force GPU EGL before MuJoCo init

from code.apps.demos.composer import DemoComposer  # noqa: E402
from code.apps.demos.recorder import DemoRecorder  # noqa: E402
from code.fleet.mission import MissionRunner  # noqa: E402
from code.sim.arena_build import COLORS
from code.warehouse.layout import (callsigns_for_layout, hero_layout,
                                   rooms6_layout, rooms_layout)

_LAYOUTS = {"hero": hero_layout, "rooms": rooms_layout, "rooms6": rooms6_layout}
_CMAP = dict(COLORS)
_FILLER = (("orange", "cube"), ("blue", "cylinder"), ("green", "ball"),
           ("yellow", "cone"), ("purple", "cube"), ("cyan", "cylinder"),
           ("blue", "ball"))


def _objects(layout, target_spot: int) -> List[dict]:
    """Red cube at ``target_spot``; distinct fillers elsewhere."""
    objs: List[dict] = []
    fi = 0
    for i, (x, y) in enumerate(layout.object_spots):
        if i == target_spot:
            c, s = "red", "cube"
        else:
            c, s = _FILLER[fi % len(_FILLER)]
            fi += 1
        objs.append({"color_name": c, "color_rgb": _CMAP[c], "shape_name": s,
                     "size": 0.24, "x": float(x), "y": float(y)})
    return objs


def render_sample(out: str, *, layout_name: str = "rooms", target_spot: int = 8,
                  steps: int = 900, decimation: int = 4, fps: int = 30,
                  seed: int = 0, make_gif: bool = False,
                  command: str = "Alpha, fetch the red cube to the delivery pad",
                  title: str = "Collaborative Search & Fetch",
                  description: str = "Delegated search, then deliver to the pad",
                  ) -> str:
    """Record one sample clip and return its mp4 path."""
    layout = _LAYOUTS.get(layout_name, rooms_layout)()
    spot = min(target_spot, len(layout.object_spots) - 1)
    callsigns = callsigns_for_layout(layout)
    mr = MissionRunner(layout=layout, objects=_objects(layout, spot),
                       callsigns=callsigns, seed=seed, use_gpu=True,
                       perception_mode="oracle", locomotion="teacher",
                       search_deadline_steps=steps)
    composer = DemoComposer(mr.fleet.viz, layout.hall_x, layout.hall_y,
                            title=title, description=description, fps=fps)
    recorder = DemoRecorder(composer, fps=fps, decimation=decimation)
    mr.submit(command)
    try:
        path = recorder.record(mr, out, max_steps=steps)
    finally:
        composer.close()
        mr.close()
    print(f"[demo2-sample] wrote {path}  ({Path(path).stat().st_size/1e6:.2f} MB)")
    if make_gif:
        gif = str(Path(out).with_suffix(".gif"))
        DemoRecorder.to_gif(path, gif, fps=12, width=760)
        print(f"[demo2-sample] wrote {gif}  ({Path(gif).stat().st_size/1e6:.2f} MB)")
    return path


def main(argv: Optional[List[str]] = None) -> None:
    ap = argparse.ArgumentParser(description="Render a Demo Set v2 sample clip")
    ap.add_argument("--out", default="ops/demo2/sample.mp4")
    ap.add_argument("--layout", choices=tuple(_LAYOUTS), default="rooms")
    ap.add_argument("--spot", type=int, default=8, help="object_spot for the red cube")
    ap.add_argument("--steps", type=int, default=900)
    ap.add_argument("--decimation", type=int, default=4)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--command", default="Alpha, fetch the red cube to the delivery pad")
    ap.add_argument("--title", default="Collaborative Search & Fetch")
    ap.add_argument("--description", default="Delegated search, then deliver to the pad")
    ap.add_argument("--gif", action="store_true")
    args = ap.parse_args(argv)
    render_sample(args.out, layout_name=args.layout, target_spot=args.spot,
                  steps=args.steps, decimation=args.decimation, fps=args.fps,
                  seed=args.seed, make_gif=args.gif, command=args.command,
                  title=args.title, description=args.description)


if __name__ == "__main__":
    main()
