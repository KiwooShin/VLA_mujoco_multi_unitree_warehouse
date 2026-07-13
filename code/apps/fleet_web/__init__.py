"""fleet_web — live interactive web demo of the warehouse robot fleet.

A small Flask app that lets a viewer type addressed natural-language orders
("Alpha, fetch the red cube to the delivery pad", "someone bring me the blue
ball") and *watch*: a live bird's-eye stream of the whole hall with the robots
walking, per-robot status chips, and the comms transcript appending in real
time. It is the interactive counterpart of the recorded gallery videos.

This package owns ``code/apps/fleet_web/`` only. It consumes the *public* API of
:class:`code.fleet.mission.MissionRunner`, :class:`code.fleet.fleet.Fleet` and
the ``code.comms`` layer (``submit`` / ``run`` / ``bus.transcript`` /
per-robot accessors) and never edits ``code/fleet``. See :mod:`.service` for the
one background sim thread and :mod:`.app` for the Flask surface.

Public entry point::

    python -m code.apps.fleet_web [--port 7799] [--layout hero]
"""

from __future__ import annotations

__all__ = ["create_app", "FleetService"]


def __getattr__(name: str):
    # Lazy re-exports so ``import code.apps.fleet_web`` stays cheap and does not
    # drag in Flask / MuJoCo until something is actually used.
    if name == "create_app":
        from code.apps.fleet_web.app import create_app

        return create_app
    if name == "FleetService":
        from code.apps.fleet_web.service import FleetService

        return FleetService
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
