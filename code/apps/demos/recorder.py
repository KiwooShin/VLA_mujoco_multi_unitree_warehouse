"""recorder.py — DemoRecorder: drive a run loop -> frames -> mp4 / gif.

Wraps a :class:`~code.fleet.mission.MissionRunner` run loop, composing a frame
every ``decimation`` control steps, and encodes h264/yuv420p/faststart via the
imageio-ffmpeg binary (the same libx264-enabled ffmpeg
``tools/build_gallery.py`` uses). :meth:`to_gif` mirrors that tool's
palettegen/paletteuse two-pass for a crisp, small GIF.
"""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path
from typing import Callable, List, Optional

import imageio_ffmpeg
import numpy as np

from code.apps.demos.composer import DemoComposer
from code.apps.demos.models import FrameState
from code.apps.demos.runner_adapter import frame_state_from_runner

_FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()

StateFn = Callable[[object, int], FrameState]


class DemoRecorder:
    """Record composed frames from a mission run into an mp4 (and optionally gif)."""

    def __init__(self, composer: DemoComposer, *, fps: int = 30,
                 decimation: int = 4) -> None:
        """Args:
            composer: The configured :class:`DemoComposer`.
            fps: Output frame rate.
            decimation: Compose one frame per this many control steps.
        """
        self.composer = composer
        self.fps = fps
        self.decimation = max(1, decimation)

    # -- capture ----------------------------------------------------------
    def record(self, runner, out_path, *, max_steps: int,
               state_fn: StateFn = frame_state_from_runner,
               hold_secs: float = 1.2) -> str:
        """Run ``runner`` to completion, composing frames, and write the mp4.

        Args:
            runner: A MissionRunner-like object exposing
                ``run(max_steps, on_step=cb)`` (``cb(runner, step)``).
            out_path: Destination ``.mp4`` path.
            max_steps: Step budget passed to ``runner.run``.
            state_fn: ``state_fn(runner, step) -> FrameState`` (defaults to the
                MissionRunner adapter).
            hold_secs: Freeze the final frame this long so the ending reads.

        Returns:
            The written mp4 path (as a string).
        """
        frames: List[np.ndarray] = []

        def on_step(r, t: int):
            if t % self.decimation == 0:
                frames.append(self.composer.compose(state_fn(r, t)))

        runner.run(max_steps, on_step=on_step)
        last_step = int(getattr(runner, "_steps", len(frames) * self.decimation))
        frames.append(self.composer.compose(state_fn(runner, last_step)))
        if frames and hold_secs > 0:
            frames.extend([frames[-1]] * int(round(self.fps * hold_secs)))
        return self.write_mp4(frames, out_path)

    # -- encode -----------------------------------------------------------
    def write_mp4(self, frames: List[np.ndarray], out_path) -> str:
        """Encode BGR frames to an h264/yuv420p/faststart mp4."""
        if not frames:
            raise ValueError("no frames to encode")
        out_path = str(out_path)
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        h, w = frames[0].shape[:2]
        writer = imageio_ffmpeg.write_frames(
            out_path, (w, h), pix_fmt_in="bgr24", pix_fmt_out="yuv420p",
            fps=self.fps, codec="libx264", macro_block_size=1,
            output_params=["-crf", "20", "-movflags", "+faststart"])
        writer.send(None)
        for f in frames:
            writer.send(np.ascontiguousarray(f, dtype=np.uint8))
        writer.close()
        return out_path

    # -- gif --------------------------------------------------------------
    @staticmethod
    def to_gif(mp4, out, *, fps: int = 12, width: int = 760,
               ss: Optional[float] = None, dur: Optional[float] = None) -> str:
        """Palette-optimised GIF from an mp4 (palettegen + paletteuse two-pass).

        Args:
            mp4: Source mp4 path.
            out: Destination gif path.
            fps: GIF frame rate.
            width: GIF width (px); height keeps aspect (``-1``).
            ss: Optional start offset (s) into the source.
            dur: Optional duration (s) of the window.

        Returns:
            The written gif path (as a string).
        """
        mp4, out = str(mp4), str(out)
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        trim: List[str] = []
        if ss is not None:
            trim += ["-ss", f"{ss}"]
        if dur is not None:
            trim += ["-t", f"{dur}"]
        scale = f"scale={width}:-1:flags=lanczos"
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tf:
            palette = tf.name
        try:
            _run([_FFMPEG, "-y", *trim, "-i", mp4, "-vf",
                  f"fps={fps},{scale},palettegen=stats_mode=diff", palette,
                  "-loglevel", "error"])
            _run([_FFMPEG, "-y", *trim, "-i", mp4, "-i", palette, "-lavfi",
                  f"fps={fps},{scale}[x];[x][1:v]paletteuse="
                  "dither=bayer:bayer_scale=3:diff_mode=rectangle", out,
                  "-loglevel", "error"])
        finally:
            Path(palette).unlink(missing_ok=True)
        return out


def _run(cmd: List[str]) -> None:
    """Run a subprocess, raising with captured stderr on failure."""
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError(
            f"ffmpeg failed ({res.returncode}): {' '.join(cmd)}\n{res.stderr[-1500:]}")
