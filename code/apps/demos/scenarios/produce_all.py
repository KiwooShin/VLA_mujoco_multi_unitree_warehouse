"""produce_all — record the six Demo Set v2 originals and build ``demo/``.

Pipeline, per scenario in :data:`~code.apps.demos.scenarios.REGISTRY`:

1. **record** the polished original to ``ops/demo2/<name>.mp4`` (DemoComposer +
   DemoRecorder, full learned stack unless the scenario overrides it);
2. **compress** it into ``demo/<name>.mp4`` (h264 / yuv420p / faststart, < 10 MB —
   the crf is bumped until it fits);
3. **gif** the story-dense window into ``demo/<name>.gif`` (palette two-pass,
   12 fps, ~760 px, < 10 MB);
4. **poster** the best single frame into ``demo/<name>_poster.png``.

Finally it writes ``demo/README.md`` (the six-demo table with stories, posters and
gif previews). ``code/apps/demos/cli.py`` remains the one-off sample renderer; this
is the batch production entrypoint.

Usage
-----
PYTHONPATH=. MUJOCO_GL=egl python -m code.apps.demos.scenarios.produce_all \\
    --only clarify_fetch --steps record,compress,gif,poster
PYTHONPATH=. MUJOCO_GL=egl python -m code.apps.demos.scenarios.produce_all --all
"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path
from typing import List, Optional

import code.apps.demos  # noqa: F401 - EGL vendor pin before MuJoCo
from code.apps.warehouse_datagen.egl_gpu import force_nvidia_egl

force_nvidia_egl()

import imageio_ffmpeg  # noqa: E402

from code.apps.demos.scenarios import REGISTRY, Scenario, get  # noqa: E402

_FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()
_REPO = Path(__file__).resolve().parents[4]
_ORIG_DIR = _REPO / "ops" / "demo2"
_DEMO_DIR = _REPO / "demo"
_SIZE_LIMIT = 10 * 1024 * 1024  # 10 MB hard cap for mp4 and gif


# --------------------------------------------------------------------------- io
def _run(cmd: List[str]) -> None:
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError(f"cmd failed ({res.returncode}): {' '.join(cmd)}\n"
                           f"{res.stderr[-1500:]}")


def _mb(path: Path) -> float:
    return path.stat().st_size / 1e6


def _duration(path: Path) -> float:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
        capture_output=True, text=True).stdout.strip()
    try:
        return float(out)
    except ValueError:
        return 0.0


# ----------------------------------------------------------------- record step
def record_original(sc: Scenario) -> Path:
    """Record the polished original to ``ops/demo2/<name>.mp4``."""
    from code.apps.demos.composer import DemoComposer
    from code.apps.demos.recorder import DemoRecorder
    _ORIG_DIR.mkdir(parents=True, exist_ok=True)
    out = _ORIG_DIR / f"{sc.name}.mp4"
    mr, layout, _cs = sc.build_runner()
    composer = DemoComposer(
        mr.fleet.viz, layout.hall_x, layout.hall_y, title=sc.title,
        description=sc.description, fps=sc.fps,
        title_card_secs=sc._title_card_secs(),
        title_fade_secs=0.3 * sc.decimation)
    recorder = DemoRecorder(composer, fps=sc.fps, decimation=sc.decimation)
    sc.submit(mr)
    try:
        path = recorder.record(mr, out, max_steps=sc.max_steps,
                               state_fn=sc.make_state_fn() or _default_state_fn())
    finally:
        composer.close()
        mr.close()
    print(f"[record] {sc.name}: {_mb(Path(path)):.2f} MB  dur {_duration(Path(path)):.1f}s")
    return Path(path)


def _default_state_fn():
    from code.apps.demos.runner_adapter import frame_state_from_runner
    return frame_state_from_runner


# --------------------------------------------------------------- compress step
def compress(sc: Scenario) -> Path:
    """Re-encode the original into ``demo/<name>.mp4`` under 10 MB (faststart)."""
    _DEMO_DIR.mkdir(parents=True, exist_ok=True)
    src = _ORIG_DIR / f"{sc.name}.mp4"
    out = _DEMO_DIR / f"{sc.name}.mp4"
    for crf in ("24", "27", "30", "33"):
        _run([_FFMPEG, "-y", "-i", str(src), "-c:v", "libx264", "-crf", crf,
              "-preset", "slow", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
              str(out), "-loglevel", "error"])
        if out.stat().st_size <= _SIZE_LIMIT:
            break
    print(f"[compress] {sc.name}.mp4: {_mb(out):.2f} MB  dur {_duration(out):.1f}s")
    return out


# -------------------------------------------------------------------- gif step
def make_gif(sc: Scenario) -> Path:
    """Palette-optimised story-dense GIF into ``demo/<name>.gif`` under 10 MB."""
    from code.apps.demos.recorder import DemoRecorder
    _DEMO_DIR.mkdir(parents=True, exist_ok=True)
    src = _DEMO_DIR / f"{sc.name}.mp4"
    out = _DEMO_DIR / f"{sc.name}.gif"
    dur_total = _duration(src)
    ss = min(sc.gif_ss, max(0.0, dur_total - 6.0))
    dur = min(sc.gif_dur, max(4.0, dur_total - ss))
    for width, fps in ((760, 12), (680, 12), (600, 10), (520, 10)):
        DemoRecorder.to_gif(src, out, fps=fps, width=width, ss=ss, dur=dur)
        if out.stat().st_size <= _SIZE_LIMIT:
            break
    print(f"[gif] {sc.name}.gif: {_mb(out):.2f} MB  ({ss:.1f}s +{dur:.1f}s)")
    return out


# ----------------------------------------------------------------- poster step
def make_poster(sc: Scenario) -> Path:
    """Extract the best single frame into ``demo/<name>_poster.png``."""
    _DEMO_DIR.mkdir(parents=True, exist_ok=True)
    src = _DEMO_DIR / f"{sc.name}.mp4"
    out = _DEMO_DIR / f"{sc.name}_poster.png"
    t = min(sc.poster_t, max(0.0, _duration(src) - 1.0))
    _run([_FFMPEG, "-y", "-ss", f"{t}", "-i", str(src), "-frames:v", "1",
          str(out), "-loglevel", "error"])
    print(f"[poster] {sc.name}_poster.png @ {t:.1f}s")
    return out


# --------------------------------------------------------------------- README
_STORIES = {
    "clarify_fetch":
        "\"Alpha, bring me the cube\" is ambiguous — the warehouse manifest holds "
        "a red, a blue and a yellow cube. Alpha does not guess: it asks the user "
        "which cube is meant, waits for the answer (\"the red one\"), then delegates "
        "a room-to-room search for the red cube and delivers it to the pad.",
    "unseen_map":
        "A warehouse the fleet has never seen — layout, doorways, shelves, object "
        "spots and delivery pad are all freshly sampled. From random start poses "
        "the owner delegates a room-to-room search for a red cube hidden in a "
        "shelf-occluded corner, receives the relative-position report, and fetches it.",
    "dual_fetch":
        "One order, two objects: \"bring the red cube and the blue ball to the "
        "delivery pad\" splits into two concurrent missions with two different "
        "owners working in parallel. Each ring is coloured for its owner; the run "
        "finishes only when both objects are on the pad.",
    "relay_multigoal":
        "A sequential relay on a second never-seen layout: one owner fetches the "
        "red cube first, then re-tasks the same teammates to search for the blue "
        "ball, delivering both in order. One TASK_COMPLETE, reported after the "
        "final leg.",
    "retask":
        "The order changes mid-mission. Alpha is fetching the red cube when the "
        "user says \"actually, bring me the yellow ball instead\". Alpha stands "
        "down its helpers, drops the abandoned approach, and re-delegates a search "
        "for the ball — the old target ring clears and a new one appears.",
    "six_robot_flagship":
        "The flagship: six humanoids (Alpha..Foxtrot) in the big 24x16 m four-room "
        "hall. A fleet-addressed \"someone bring me a ball\" is ambiguous (red, "
        "green, blue), so the allocator clarifies, then assigns the shortest-path "
        "owner, which delegates the four rooms to four searchers with the sixth "
        "teammate held back in reserve, and delivers.",
}


def write_readme() -> Path:
    """Write ``demo/README.md`` — the six-demo table with posters + gif previews."""
    _DEMO_DIR.mkdir(parents=True, exist_ok=True)
    out = _DEMO_DIR / "README.md"
    lines: List[str] = []
    lines.append("# Demo Set v2\n")
    lines.append(
        "Six production demos of the multi-room collaborative fetch fleet, each on "
        "the full learned stack (GROUND_NET perception + VLA locomotion) unless "
        "noted, with **random robot start poses**, an **always-on ego-camera strip**, "
        "a **live comms transcript**, and **clarification dialogue** for ambiguous "
        "orders. Click a poster to play the MP4; the GIF previews the story-dense "
        "window inline.\n")
    lines.append("| Demo | Preview |")
    lines.append("|------|---------|")
    for sc in REGISTRY:
        poster = f"{sc.name}_poster.png"
        mp4 = f"{sc.name}.mp4"
        gif = f"{sc.name}.gif"
        cell = (f"**{sc.title}**<br/>{_STORIES.get(sc.name, sc.description)}"
                f"<br/><br/>[![{sc.title}]({poster})]({mp4})")
        lines.append(f"| {cell} | ![{sc.title} preview]({gif}) |")
    lines.append("")
    lines.append("## Reproduce\n")
    lines.append("```bash")
    lines.append("PYTHONPATH=. MUJOCO_GL=egl \\")
    lines.append("  python -m code.apps.demos.scenarios.produce_all --all")
    lines.append("```")
    lines.append("")
    lines.append("Each demo is a deterministic "
                 "`code.apps.demos.scenarios.Scenario` (seeded layout, spawns, "
                 "objects, order and any clarify/re-task schedule); "
                 "`--only <name>` produces a single demo.\n")
    out.write_text("\n".join(lines))
    print(f"[readme] wrote {out}")
    return out


# ----------------------------------------------------------------------- main
_STEPS = {"record": record_original, "compress": compress, "gif": make_gif,
          "poster": make_poster}


def produce(sc: Scenario, steps: List[str]) -> None:
    """Run the requested pipeline steps for one scenario."""
    for step in steps:
        _STEPS[step](sc)


def main(argv: Optional[List[str]] = None) -> None:
    ap = argparse.ArgumentParser(description="Produce Demo Set v2")
    ap.add_argument("--only", default=None, help="scenario name (default: all)")
    ap.add_argument("--all", action="store_true", help="produce every scenario")
    ap.add_argument("--steps", default="record,compress,gif,poster",
                    help="comma list of: record,compress,gif,poster")
    ap.add_argument("--no-readme", action="store_true")
    args = ap.parse_args(argv)

    steps = [s.strip() for s in args.steps.split(",") if s.strip()]
    scenarios = REGISTRY if (args.all or args.only is None) else [get(args.only)]
    for sc in scenarios:
        produce(sc, steps)
    if not args.no_readme:
        write_readme()


if __name__ == "__main__":
    main()
