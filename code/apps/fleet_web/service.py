"""service.py — The single background sim loop behind the fleet web demo.

:class:`FleetService` owns exactly one worker thread that builds the fleet once,
steps it continuously (idle robots stand, active missions walk), accepts queued
orders, and drives missions strictly one at a time. It exposes a small,
thread-safe snapshot API the Flask layer polls: the latest BEV JPEG, per-robot
status chips, an incrementally-fetchable comms transcript, and mission state.

Concurrency model: the worker thread is the *only* writer of the sim; Flask
request threads never touch the engine, they read cached snapshots guarded by an
``RLock``. Rendering (potentially slow under machine load) happens off-lock so a
stalled frame never blocks ``/state`` or ``/command``. Pacing targets a steady
frame rate and degrades to slow-motion (never fast-forward) under contention.
"""

from __future__ import annotations

import threading
import time
import traceback
from typing import Dict, List, Optional

from code.apps.fleet_web.commands import validate_command
from code.apps.fleet_web.engine import SimEngine
from code.apps.fleet_web.status import robot_view
from code.apps.fleet_web.transcript import TranscriptLog, kind_for


class FleetService:
    """Runs the fleet sim in one thread and serves thread-safe snapshots."""

    def __init__(self, engine: Optional[SimEngine] = None, *,
                 max_steps: int = 6000, steps_per_frame: int = 4,
                 target_fps: float = 18.0) -> None:
        """Configure the service (no sim work until :meth:`start`).

        Args:
            engine: The sim backend; a real :class:`MujocoFleetEngine` is built
                lazily on :meth:`start` if omitted (tests inject a fake).
            max_steps: Hard per-mission step cap handed to ``MissionRunner.run``.
            steps_per_frame: Sim steps advanced between rendered frames (larger
                = faster motion, fewer frames).
            target_fps: Target rendered-frame rate; sim pace = ``steps_per_frame
                * target_fps`` steps/s when the machine keeps up.
        """
        self._engine = engine
        self._max_steps = int(max_steps)
        self._steps_per_frame = max(1, int(steps_per_frame))
        self._target_fps = float(target_fps)

        self._lock = threading.RLock()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

        self._transcript = TranscriptLog()
        self._ingested = 0
        self._queue: List[str] = []
        self._active = False

        self._frame: Optional[bytes] = None
        self._robots: List[Dict[str, object]] = []
        self._phase = "STANDING BY"
        self._recipient = ""
        self._target = ""
        self._outcome: Optional[str] = None
        self._status = "Booting the fleet…"
        self._on_pad = False
        self._last_frame_wall = 0.0

    # -- lifecycle --------------------------------------------------------
    def start(self) -> None:
        """Build the engine (if needed) and launch the worker thread."""
        if self._engine is None:
            from code.apps.fleet_web.engine import MujocoFleetEngine

            self._engine = MujocoFleetEngine()
        self._thread = threading.Thread(target=self._loop, name="fleet-sim",
                                        daemon=True)
        self._thread.start()

    def stop(self, timeout: float = 10.0) -> None:
        """Signal the worker to stop, join it, and release the engine."""
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=timeout)
            self._thread = None
        if self._engine is not None:
            try:
                self._engine.close()
            except Exception:  # best-effort teardown
                traceback.print_exc()

    @property
    def stopped(self) -> bool:
        """Whether shutdown has been requested (used by the stream loop)."""
        return self._stop.is_set()

    # -- command intake ---------------------------------------------------
    def submit_command(self, text: str) -> Dict[str, object]:
        """Validate and enqueue a typed order; return a UI-friendly result.

        Args:
            text: The raw typed order.

        Returns:
            ``{"ok": bool, ...}`` — on success ``queued`` marks whether it will
            wait behind an in-flight/queued order and ``message`` is a human
            notice; on failure ``error`` is a friendly explanation.
        """
        callsigns = list(self._engine.callsigns) if self._engine else [
            "Alpha", "Bravo", "Charlie", "Delta"]
        check = validate_command(text, callsigns)
        if not check.ok:
            return {"ok": False, "error": check.error}

        with self._lock:
            ahead = (1 if self._active else 0) + len(self._queue)
            self._queue.append(text)
            busy = ahead > 0
            # Drop the finished mission's outcome so the UI never lingers on a
            # stale "✓ delivered" between this accept and the worker picking it up.
            if not busy:
                self._outcome = None
                self._phase = "DISPATCHING"
        if busy:
            note = (f"{check.recipient_label} is busy — order queued "
                    f"({ahead} ahead): fetch the {check.target_desc}.")
            self._line("system", "", note, "system")
            return {"ok": True, "queued": True, "message": note}
        return {"ok": True, "queued": False,
                "message": f"Order sent to {check.recipient_label}: "
                           f"fetch the {check.target_desc}."}

    # -- snapshots (read by Flask threads) --------------------------------
    def latest_jpeg(self) -> Optional[bytes]:
        """Return the most recent encoded BEV frame, or ``None`` before boot."""
        with self._lock:
            return self._frame

    def snapshot_state(self, after: int = 0) -> Dict[str, object]:
        """Return the full UI state, with transcript lines newer than ``after``.

        Args:
            after: Return only transcript entries with id greater than this.

        Returns:
            ``{"robots", "transcript", "mission", "next_id", "status"}``.
        """
        with self._lock:
            return {
                "robots": [dict(r) for r in self._robots],
                "transcript": self._transcript.dicts_since(after),
                "next_id": self._transcript.last_id,
                "status": self._status,
                "mission": {
                    "active": self._active,
                    "recipient": self._recipient,
                    "target": self._target,
                    "phase": self._phase,
                    "outcome": self._outcome,
                    "on_pad": self._on_pad,
                    "queued": len(self._queue),
                },
            }

    # -- worker loop ------------------------------------------------------
    def _loop(self) -> None:
        try:
            self._set(status="Booting the fleet — building four robots…")
            self._engine.reset()
            self._reset_ingest()
            self._refresh(render=True)
            self._set(status="Fleet ready — type an order or tap an example.")
            missions_run = False
            while not self._stop.is_set():
                text = self._next_command()
                if text is None:
                    self._engine.idle_step()
                    self._refresh(render=True)
                    self._pace()
                    continue
                # Lifecycle API: reuse the built fleet across missions — clear
                # the previous mission's state (continuous world), no rebuild.
                if missions_run:
                    self._engine.reset_mission()
                self._begin(text)
                outcome = self._engine.run_mission(self._on_step, self._max_steps)
                self._finish(text, outcome)
                missions_run = True
        except Exception:  # never let the worker die silently
            traceback.print_exc()
            self._set(status="Sim error — see server logs.")
        finally:
            self._set(status="Fleet stopped.")

    def _next_command(self) -> Optional[str]:
        """Pop the next queued order (marking the service busy), or ``None``."""
        with self._lock:
            if self._queue and not self._active:
                self._active = True
                self._outcome = None
                self._phase = "DISPATCHING"
                self._recipient = ""
                self._target = ""
                return self._queue.pop(0)
            return None

    def _begin(self, text: str) -> None:
        """Submit a mission and record its opening state + user echo."""
        check = validate_command(text, list(self._engine.callsigns))
        self._engine.submit(text)
        self._line("you", check.recipient_label, f"“{text}”", "user")
        self._set(active=True, recipient=check.recipient_label,
                  target=check.target_desc, outcome=None, on_pad=False,
                  status=f"On it — {check.recipient_label}: "
                         f"fetch the {check.target_desc}.")

    def _on_step(self, t: int) -> object:
        """Per-sim-step hook: ingest chatter, render, pace, honour shutdown.

        Returns ``False`` to cancel the active run promptly on shutdown (the
        runner's public stop contract), else ``None``.
        """
        if self._stop.is_set():
            return False
        if t % self._steps_per_frame == 0:
            self._ingest()
            self._refresh(render=True)
            self._pace()
        return None

    def _finish(self, text: str, outcome: str) -> None:
        """Record a mission's terminal state + a one-line summary."""
        self._ingest()
        on_pad = self._engine.object_on_pad()
        target = self._target
        if outcome == "complete":
            summary = f"✓ Delivered the {target} to the delivery pad."
        elif outcome == "stopped":
            summary = "Mission stopped."
        elif outcome == "failed":
            summary = f"✗ Could not deliver the {target}."
        else:
            summary = f"Mission ended ({outcome})."
        self._line("system", "", summary, "system")
        with self._lock:
            self._active = False
            self._outcome = outcome
            self._on_pad = on_pad
            self._phase = "DONE"
        self._refresh(render=True)
        self._set(status=summary + " Ready for the next order.")

    # -- snapshot plumbing ------------------------------------------------
    def _refresh(self, *, render: bool) -> None:
        """Recompute cached snapshots (render is done off-lock)."""
        views = [robot_view(r) for r in self._engine.robots()]
        phase = self._engine.mission_phase()
        frame = self._engine.render_jpeg() if render else None
        with self._lock:
            self._robots = views
            self._phase = phase
            if frame is not None:
                self._frame = frame

    def _ingest(self) -> None:
        """Append the current runner's new bus lines to the persistent log."""
        snaps = self._engine.bus_snaps()
        with self._lock:
            for s in snaps[self._ingested:]:
                if s.sender == "user":
                    continue  # the raw order is already echoed as a "you" line
                self._transcript.append(s.sender, s.recipient, s.text,
                                        kind_for(s.sender))
            self._ingested = len(snaps)

    def _reset_ingest(self) -> None:
        with self._lock:
            self._ingested = 0

    def _line(self, sender: str, recipient: str, text: str, kind: str) -> None:
        with self._lock:
            self._transcript.append(sender, recipient, text, kind)

    def _set(self, **fields: object) -> None:
        with self._lock:
            for key, value in fields.items():
                setattr(self, f"_{key}", value)

    def _pace(self) -> None:
        """Sleep to hold the target frame rate (never fast-forwards)."""
        period = 1.0 / self._target_fps
        now = time.perf_counter()
        elapsed = now - self._last_frame_wall
        if 0.0 < elapsed < period:
            time.sleep(period - elapsed)
        self._last_frame_wall = time.perf_counter()
