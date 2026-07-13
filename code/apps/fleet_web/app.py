"""app.py — Flask surface for the fleet web demo.

An application factory (:func:`create_app`) wraps a single
:class:`code.apps.fleet_web.service.FleetService` — the service *is* the only
shared state, so there are no module-level mutable globals. Four routes:

* ``GET  /``            — the single-page dashboard (:mod:`code.apps.fleet_web.ui`),
* ``GET  /stream``      — an MJPEG multipart stream of the live BEV,
* ``GET  /state``       — JSON snapshot (robots, transcript since ``?after=``, mission),
* ``POST /command``     — submit a typed order ``{"text": ...}`` (validated).
"""

from __future__ import annotations

import functools
import time
from typing import Iterator, Optional

_BOUNDARY = "frame"
_STREAM_FPS = 20.0


@functools.lru_cache(maxsize=1)
def _placeholder_jpeg() -> bytes:
    """A dark 'booting' frame served until the first real BEV frame exists."""
    try:
        import cv2
        import numpy as np

        img = np.full((720, 960, 3), 12, dtype=np.uint8)
        cv2.putText(img, "Warehouse Fleet", (60, 340),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.4, (120, 170, 90), 3, cv2.LINE_AA)
        cv2.putText(img, "booting the fleet...", (62, 400),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (120, 130, 140), 2, cv2.LINE_AA)
        ok, buf = cv2.imencode(".jpg", img)
        return buf.tobytes() if ok else b""
    except Exception:
        return b""


def mjpeg_frames(service, *, fps: float = _STREAM_FPS,
                 boundary: str = _BOUNDARY,
                 max_frames: Optional[int] = None) -> Iterator[bytes]:
    """Yield MJPEG multipart chunks from the service's latest BEV frame.

    Args:
        service: The :class:`FleetService` (or a fake) exposing ``latest_jpeg``
            and ``stopped``.
        fps: Cap on stream frame rate.
        boundary: Multipart boundary token.
        max_frames: Stop after this many chunks (tests); ``None`` = run forever.

    Yields:
        ``--boundary`` framed JPEG payloads.
    """
    period = 1.0 / fps
    sep = f"--{boundary}\r\nContent-Type: image/jpeg\r\n\r\n".encode()
    count = 0
    while not getattr(service, "stopped", False):
        frame = service.latest_jpeg() or _placeholder_jpeg()
        if frame:
            yield sep + frame + b"\r\n"
        count += 1
        if max_frames is not None and count >= max_frames:
            return
        time.sleep(period)


def create_app(service):
    """Build the Flask app bound to one :class:`FleetService`.

    Args:
        service: A started (or fake) service providing ``snapshot_state``,
            ``latest_jpeg``, ``submit_command`` and ``callsigns`` behaviour.

    Returns:
        A configured :class:`flask.Flask` app (no server started).
    """
    from flask import Flask, Response, jsonify, render_template_string, request

    from code.apps.fleet_web.ui import PAGE, render_examples_accents

    app = Flask(__name__)
    app.config["FLEET_SERVICE"] = service
    examples, accents = render_examples_accents()

    @app.route("/")
    def index() -> str:
        return render_template_string(PAGE, examples=examples, accents=accents)

    @app.route("/state")
    def state() -> Response:
        after = request.args.get("after", default=0, type=int) or 0
        return jsonify(service.snapshot_state(after))

    @app.route("/command", methods=["POST"])
    def command() -> Response:
        data = request.get_json(silent=True) or {}
        text = data.get("text", "") if isinstance(data, dict) else ""
        result = service.submit_command(text)
        return jsonify(result)

    @app.route("/stream")
    def stream() -> Response:
        return Response(
            mjpeg_frames(service),
            mimetype=f"multipart/x-mixed-replace; boundary={_BOUNDARY}")

    return app
