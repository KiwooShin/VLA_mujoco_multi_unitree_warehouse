"""__main__.py — ``python -m code.apps.fleet_web`` entry point.

Boots one :class:`code.apps.fleet_web.service.FleetService` (the background sim
thread), wraps it in the Flask app and serves it locally. Run from the repo
root::

    PYTHONPATH=. MUJOCO_GL=egl python -m code.apps.fleet_web --port 7799
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import List, Optional

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")


def main(argv: Optional[List[str]] = None) -> int:
    """Parse args, start the service, and run the Flask server."""
    ap = argparse.ArgumentParser(
        prog="python -m code.apps.fleet_web",
        description="Live interactive web demo of the warehouse robot fleet.")
    ap.add_argument("--port", type=int, default=7799, help="TCP port to serve on.")
    ap.add_argument("--host", type=str, default="127.0.0.1", help="Bind address.")
    ap.add_argument("--layout", type=str, default="hero", choices=("hero",),
                    help="Warehouse layout (only the tuned hero hall).")
    ap.add_argument("--seed", type=int, default=0, help="Scene RNG seed.")
    ap.add_argument("--max-steps", type=int, default=6000,
                    help="Per-mission control-step budget.")
    ap.add_argument("--steps-per-frame", type=int, default=4,
                    help="Sim steps advanced between rendered frames.")
    ap.add_argument("--fps", type=float, default=18.0,
                    help="Target rendered-frame rate (sim pace scales with it).")
    ap.add_argument("--no-gpu", action="store_true",
                    help="Force CPU walk policies (slower).")
    ap.add_argument("--locomotion", choices=("teacher", "vla"), default="teacher",
                    help="WBC walk policy (default) or the trained VLA policy (F5).")
    ap.add_argument("--ckpt", type=str, default=None,
                    help="GroundedNav checkpoint for --locomotion vla (default: F5 fine-tune).")
    ap.add_argument("--device", type=str, default=None,
                    help="Torch device for the VLA policy (cuda|cpu; default auto).")
    args = ap.parse_args(argv)

    from code.apps.fleet_web.app import create_app
    from code.apps.fleet_web.engine import MujocoFleetEngine
    from code.apps.fleet_web.service import FleetService

    engine = MujocoFleetEngine(seed=args.seed, use_gpu=not args.no_gpu,
                               layout_name=args.layout,
                               locomotion=args.locomotion, vla_ckpt=args.ckpt,
                               vla_device=args.device)
    service = FleetService(engine, max_steps=args.max_steps,
                           steps_per_frame=args.steps_per_frame,
                           target_fps=args.fps)
    service.start()
    app = create_app(service)

    print(f"[fleet_web] serving http://{args.host}:{args.port}  "
          f"(layout={args.layout} seed={args.seed})", flush=True)
    try:
        import logging

        logging.getLogger("werkzeug").setLevel(logging.WARNING)
        app.run(host=args.host, port=args.port, debug=False, use_reloader=False,
                threaded=True)
    finally:
        service.stop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
