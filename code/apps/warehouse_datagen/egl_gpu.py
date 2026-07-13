"""egl_gpu.py — Force MuJoCo EGL offscreen rendering onto the NVIDIA GPU.

Why this exists (F5 datagen requirement: rendering MUST run on GPU)
-------------------------------------------------------------------
On this host the conda env exports ``__EGL_VENDOR_LIBRARY_DIRS`` pointing at
``$CONDA/share/glvnd/egl_vendor.d`` which contains ONLY Mesa ICDs
(``50_mesa.json``, ``99_anaconda_mesa.json``). libglvnd honours that env var and
therefore never consults the *system* NVIDIA ICD
(``/usr/share/glvnd/egl_vendor.d/10_nvidia.json``), so MuJoCo silently binds the
Mesa llvmpipe SOFTWARE rasteriser:

    libEGL warning: egl: failed to create dri2 screen
    render_ego  ~160-180 ms/frame,  gpu_util 0%   (measured)

The baseline fix in ``code/sim/arena_build.py`` only *setdefault*s
``__EGL_VENDOR_LIBRARY_FILENAMES`` and does NOT clear the conda
``__EGL_VENDOR_LIBRARY_DIRS`` — with both set on this glvnd build the Mesa dir
still wins. Clearing the conda DIRS var and pinning FILENAMES to the NVIDIA ICD
restores GPU rendering:

    render_ego  ~0.88 ms/frame,  gpu_util ~78% under load   (measured, ~190x)

``force_nvidia_egl()`` MUST be called BEFORE anything imports/initialises MuJoCo
(i.e. at the very top of the CLI entrypoint), because MuJoCo reads these env
vars when it creates its first EGL context.
"""

from __future__ import annotations

import os

NVIDIA_EGL_ICD = "/usr/share/glvnd/egl_vendor.d/10_nvidia.json"


def force_nvidia_egl() -> bool:
    """Steer libglvnd/EGL to the system NVIDIA ICD for GPU offscreen rendering.

    Idempotent. Safe to call on hosts without the NVIDIA ICD (falls back to the
    existing behaviour). Sets ``MUJOCO_GL=egl`` / ``PYOPENGL_PLATFORM=egl`` too.

    Returns:
        True if the NVIDIA ICD was found and pinned, False otherwise.
    """
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    if os.path.exists(NVIDIA_EGL_ICD):
        # Drop the conda mesa-only vendor dir so the loader consults our pinned
        # ICD file instead of the Mesa ICDs it contains.
        os.environ.pop("__EGL_VENDOR_LIBRARY_DIRS", None)
        os.environ["__EGL_VENDOR_LIBRARY_FILENAMES"] = NVIDIA_EGL_ICD
        return True
    return False


def gpu_utilization() -> int:
    """Return current GPU utilization percent via nvidia-smi, or -1 on error."""
    import subprocess
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=utilization.gpu",
             "--format=csv,noheader,nounits"],
            stderr=subprocess.DEVNULL,
        ).decode().strip().split("\n")[0]
        return int(out)
    except Exception:
        return -1


def probe_render_ms(n: int = 60) -> dict:
    """Time one ego render on a tiny warehouse arena; assert it is GPU-fast.

    Returns a dict {ms_per_frame, gpu_util_peak, on_gpu}. ``on_gpu`` is a
    heuristic (< 10 ms/frame implies real GPU rendering; the CPU llvmpipe
    fallback measures ~160 ms).
    """
    force_nvidia_egl()
    import time
    import numpy as np
    import mujoco
    from code.warehouse.layout import hero_layout
    from code.warehouse.arena import warehouse_scene_cfg, build_warehouse_arena
    from code.arena import ArenaRenderer

    cfg = warehouse_scene_cfg(hero_layout(), robot="Alpha",
                              rng=np.random.default_rng(0))
    model = build_warehouse_arena(cfg)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    rend = ArenaRenderer(model)
    for _ in range(3):
        rend.render_ego(data, 0.0, render_depth=False)
    t0 = time.time()
    for _ in range(n):
        rend.render_ego(data, 0.0, render_depth=False)
    ms = (time.time() - t0) / n * 1000.0
    # sustained load to sample gpu_util
    peak = 0
    t0 = time.time()
    while time.time() - t0 < 1.5:
        for _ in range(200):
            rend.render_ego(data, 0.0, render_depth=False)
        peak = max(peak, gpu_utilization())
    rend.close()
    return {"ms_per_frame": round(ms, 2), "gpu_util_peak": peak,
            "on_gpu": bool(ms < 10.0)}


if __name__ == "__main__":
    import json
    print(json.dumps(probe_render_ms(), indent=2))
