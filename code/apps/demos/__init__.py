"""demos — Production video composer for Demo Set v2 (docs/demo2_spec.md §B).

Public API::

    from code.apps.demos import (DemoComposer, DemoRecorder, FrameState,
                                 RobotFrame, Ring, Pad, PlannedPath,
                                 frame_state_from_runner)

``DemoComposer`` renders one polished 1600x900 frame per call (BEV map + comms
panel + always-on ego strip + HUD + title card); ``DemoRecorder`` drives a
mission run loop into an mp4/gif. ``frame_state_from_runner`` adapts a
``MissionRunner`` into the generic :class:`FrameState` the composer consumes.

GPU rendering: importing anything under ``code`` runs ``code/__init__`` which
pins the NVIDIA EGL ICD (``__EGL_VENDOR_LIBRARY_FILENAMES``) before MuJoCo's
first context, so offscreen renders land on the GPU. Standalone production
scripts (see :mod:`code.apps.demos.cli`) additionally call
``force_nvidia_egl()`` for the stronger guarantee. ``DemoComposer`` /
``DemoRecorder`` load lazily so the pure-logic modules (layout/effects/text)
stay importable without cv2/MuJoCo.
"""

from __future__ import annotations

from typing import Any

from code.apps.demos.models import (FrameState, Pad, PlannedPath, Ring,
                                    RobotFrame)
from code.apps.demos.runner_adapter import frame_state_from_runner

__all__ = [
    "DemoComposer", "DemoRecorder", "FrameState", "RobotFrame", "Ring", "Pad",
    "PlannedPath", "frame_state_from_runner",
]


def __getattr__(name: str) -> Any:
    """Lazily expose the cv2/MuJoCo-backed classes on first access."""
    if name == "DemoComposer":
        from code.apps.demos.composer import DemoComposer
        return DemoComposer
    if name == "DemoRecorder":
        from code.apps.demos.recorder import DemoRecorder
        return DemoRecorder
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
