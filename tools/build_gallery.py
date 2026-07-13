"""build_gallery.py — Compress the Phase-5 hero videos into committed gallery assets.

Takes the raw hero recordings in ``ops/phase5/hero/`` (gitignored) and produces the
committed gallery deliverables in ``assets/gallery/``:

* ``{mission_c,mission_b,fleet_nav,allocator}.mp4`` — h264 / yuv420p / faststart,
  crf ~28, each < 8 MB;
* ``*_poster.png``  — the single best frame of each clip;
* ``hero_reel.mp4`` — the four clips concatenated behind 1 s title cards that name
  each segment, letterboxed to one canonical size, < 20 MB;
* ``mission_c.gif`` — a short (~16 s), palette-optimized GIF of the story-dense
  fetch-and-carry tail of the mission-C clip, for inline motion in the README
  (GitHub can't autoplay committed MP4s); < 10 MB.

Uses the libx264-enabled ffmpeg bundled with ``imageio-ffmpeg`` (the base ffmpeg on
this host lacks libx264).

Usage
-----
python tools/build_gallery.py
"""

from __future__ import annotations

import dataclasses
import os
import subprocess
import sys
from pathlib import Path
from typing import List, Optional

import cv2
import imageio_ffmpeg
import numpy as np

_REPO = Path(__file__).resolve().parents[1]
_HERO = _REPO / "ops" / "phase5" / "hero"
_GALLERY = _REPO / "assets" / "gallery"
_TMP = _REPO / "ops" / "phase5" / "_gallery_tmp"

_FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()
_CRF = "28"
_REEL_W, _REEL_H = 1320, 720  # canonical reel frame (matches the mission clips)


@dataclasses.dataclass(frozen=True)
class Segment:
    """One reel segment: a source clip, its gallery name, poster time, title."""

    src: str            # basename in ops/phase5/hero (no extension)
    name: str           # gallery basename
    poster_t: float     # seconds into the clip for the poster frame
    title: str          # title-card headline
    subtitle: str       # title-card sub-line


# Reel order builds toward the flagship finale.
_SEGMENTS: List[Segment] = [
    Segment("fleet_bev", "fleet_nav", 9.0,
            "Fleet Navigation", "four robots, shared aisles, live pauses"),
    Segment("mission_B", "mission_b", 22.0,
            "Peer Visibility Handoff", "a teammate reports the location"),
    Segment("mission_D", "allocator", 12.0,
            "Task Allocation", "the shortest-path robot is chosen"),
    Segment("mission_C", "mission_c", 53.0,
            "Collaborative Search and Fetch", "delegated search, then deliver"),
]


def _run(cmd: List[str]) -> None:
    """Run a subprocess, raising with captured stderr on failure."""
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError(f"cmd failed ({res.returncode}): {' '.join(cmd)}\n"
                           f"{res.stderr[-1500:]}")


def compress(seg: Segment, src_dir: Path = _HERO) -> Path:
    """H264-compress one clip into the gallery (crf 28, yuv420p, faststart)."""
    out = _GALLERY / f"{seg.name}.mp4"
    _run([_FFMPEG, "-y", "-i", str(src_dir / f"{seg.src}.mp4"),
          "-c:v", "libx264", "-crf", _CRF, "-preset", "slow",
          "-pix_fmt", "yuv420p", "-movflags", "+faststart",
          str(out), "-loglevel", "error"])
    return out


def poster(seg: Segment, src_dir: Path = _HERO) -> Path:
    """Extract the best single frame of a clip as its poster PNG."""
    out = _GALLERY / f"{seg.name}_poster.png"
    _run([_FFMPEG, "-y", "-ss", f"{seg.poster_t}",
          "-i", str(src_dir / f"{seg.src}.mp4"),
          "-frames:v", "1", str(out), "-loglevel", "error"])
    return out


def _title_card(seg: Segment) -> Path:
    """Render a 1320x720 title-card PNG (dark bg, centered headline + sub-line)."""
    img = np.full((_REEL_H, _REEL_W, 3), 22, dtype=np.uint8)
    cx = _REEL_W // 2
    (tw, _), _ = cv2.getTextSize(seg.title, cv2.FONT_HERSHEY_SIMPLEX, 1.6, 3)
    cv2.putText(img, seg.title, (cx - tw // 2, 330), cv2.FONT_HERSHEY_SIMPLEX,
                1.6, (245, 245, 245), 3, cv2.LINE_AA)
    (sw, _), _ = cv2.getTextSize(seg.subtitle, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
    cv2.putText(img, seg.subtitle, (cx - sw // 2, 390), cv2.FONT_HERSHEY_SIMPLEX,
                0.9, (150, 200, 150), 2, cv2.LINE_AA)
    cv2.line(img, (cx - 160, 355), (cx + 160, 355), (90, 90, 90), 1, cv2.LINE_AA)
    path = _TMP / f"title_{seg.name}.png"
    cv2.imwrite(str(path), img)
    return path


def build_reel(segments: List[Segment] = _SEGMENTS, src_dir: Path = _HERO,
               out_name: str = "hero_reel.mp4") -> Path:
    """Concatenate the clips behind 1 s title cards into a title-carded reel."""
    _TMP.mkdir(parents=True, exist_ok=True)
    inputs: List[str] = []
    norm: List[str] = []
    idx = 0
    for seg in segments:
        card = _title_card(seg)
        inputs += ["-loop", "1", "-t", "1", "-i", str(card)]
        norm.append(f"[{idx}:v]scale={_REEL_W}:{_REEL_H}:force_original_aspect_"
                    f"ratio=decrease,pad={_REEL_W}:{_REEL_H}:(ow-iw)/2:(oh-ih)/2,"
                    f"setsar=1,fps=30[v{idx}]")
        idx += 1
        inputs += ["-i", str(src_dir / f"{seg.src}.mp4")]
        norm.append(f"[{idx}:v]scale={_REEL_W}:{_REEL_H}:force_original_aspect_"
                    f"ratio=decrease,pad={_REEL_W}:{_REEL_H}:(ow-iw)/2:(oh-ih)/2,"
                    f"setsar=1,fps=30[v{idx}]")
        idx += 1
    concat = "".join(f"[v{i}]" for i in range(idx)) + f"concat=n={idx}:v=1:a=0[out]"
    filt = ";".join(norm) + ";" + concat
    out = _GALLERY / out_name
    _run([_FFMPEG, "-y", *inputs, "-filter_complex", filt, "-map", "[out]",
          "-c:v", "libx264", "-crf", _CRF, "-preset", "slow",
          "-pix_fmt", "yuv420p", "-movflags", "+faststart",
          str(out), "-loglevel", "error"])
    return out


# Inline-motion GIF: the story-dense tail of the mission-C clip (searchers have
# fanned out, Bravo reports the find, Alpha walks the long fetch diagonal, picks
# up the cube and carries it to the pad). Palette-optimized to stay well < 10 MB.
_GIF_SRC = "mission_C"     # basename in ops/phase5/hero
_GIF_OUT = "mission_c.gif"
_GIF_SS, _GIF_DUR = 40.0, 16.0   # window (seconds) into the source clip
_GIF_W, _GIF_FPS = 760, 12       # output width (px) and frame rate


def make_gif(src: Path = _HERO / f"{_GIF_SRC}.mp4", out_name: str = _GIF_OUT,
             ss: float = _GIF_SS, dur: float = _GIF_DUR,
             palette_name: str = "mission_c_palette.png") -> Path:
    """Render the inline-motion GIF via a two-pass palettegen/paletteuse.

    A per-clip optimized 256-colour palette (``palettegen``) plus rectangle-diff
    ``paletteuse`` keeps the flat-shaded warehouse render crisp at a fraction of
    a naive GIF's size.
    """
    palette = _TMP / palette_name
    out = _GALLERY / out_name
    scale = f"scale={_GIF_W}:-1:flags=lanczos"
    _run([_FFMPEG, "-y", "-ss", f"{ss}", "-t", f"{dur}", "-i", str(src),
          "-vf", f"fps={_GIF_FPS},{scale},palettegen=stats_mode=diff",
          str(palette), "-loglevel", "error"])
    _run([_FFMPEG, "-y", "-ss", f"{ss}", "-t", f"{dur}", "-i", str(src),
          "-i", str(palette),
          "-lavfi", f"fps={_GIF_FPS},{scale}[x];"
                    "[x][1:v]paletteuse=dither=bayer:bayer_scale=3:diff_mode=rectangle",
          str(out), "-loglevel", "error"])
    return out


# ---------------------------------------------------------------------------
# Final-demo gallery (F1-F6: rooms layout, GROUND_NET perception, VLA
# locomotion). Sources the four frame-verified originals in ``ops/final/`` and
# writes the committed ``final_*`` deliverables, regenerating ``hero_reel.mp4``
# from the FINAL clips and replacing ``mission_c.gif`` from ``final_mission_c``.
# The four cycle-1 gallery clips (fleet_nav/mission_b/allocator/mission_c) are
# left in place (they become the "hero-layout" section a later README pass owns).
# ---------------------------------------------------------------------------
_FINAL = _REPO / "ops" / "final"

_FINAL_SEGMENTS: List[Segment] = [
    Segment("fleet_bev", "final_fleet_nav", 10.0,
            "Fleet Navigation (VLA)", "four robots, cross-room, live proximity pauses"),
    Segment("mission_B", "final_scenario_b", 4.7,
            "Peer Visibility Handoff", "a teammate reports the object's relative position"),
    Segment("mission_F4", "final_generic", 6.0,
            "Fleet-Addressed Command", "'bring the object' - allocator assigns, first found wins"),
    Segment("mission_C", "final_mission_c", 33.0,
            "Collaborative Search and Fetch", "delegated room-to-room search, then deliver"),
]

# GIF window: the story-dense tail of final_mission_c — the find (F2 ring +
# F3 report), Alpha's long fetch diagonal, the GROUND_NET heatmap confirmation,
# the pickup and the start of the carry.
_FGIF_SS, _FGIF_DUR = 28.5, 16.0


def build_final() -> None:
    """Build the F1-F6 final-demo gallery from ``ops/final/`` originals."""
    _GALLERY.mkdir(parents=True, exist_ok=True)
    _TMP.mkdir(parents=True, exist_ok=True)
    for seg in _FINAL_SEGMENTS:
        clip = compress(seg, src_dir=_FINAL)
        pos = poster(seg, src_dir=_FINAL)
        print(f"[gallery] {clip.name}: {_size_mb(clip):.2f} MB  poster {pos.name}")
    reel = build_reel(_FINAL_SEGMENTS, src_dir=_FINAL, out_name="hero_reel.mp4")
    print(f"[gallery] {reel.name}: {_size_mb(reel):.2f} MB  (from the FINAL clips)")
    gif = make_gif(src=_FINAL / "mission_C.mp4", out_name="mission_c.gif",
                   ss=_FGIF_SS, dur=_FGIF_DUR,
                   palette_name="final_mission_c_palette.png")
    print(f"[gallery] {gif.name}: {_size_mb(gif):.2f} MB  (from final_mission_c)")


def _size_mb(path: Path) -> float:
    return path.stat().st_size / 1e6


def main(argv: Optional[List[str]] = None) -> None:
    import argparse

    ap = argparse.ArgumentParser(description="Build the committed gallery assets")
    ap.add_argument("--final", action="store_true",
                    help="build the F1-F6 final-demo gallery from ops/final/ "
                         "(final_* clips, reel + mission_c.gif from the finals)")
    args = ap.parse_args(argv)
    if args.final:
        build_final()
        return
    _GALLERY.mkdir(parents=True, exist_ok=True)
    _TMP.mkdir(parents=True, exist_ok=True)
    for seg in _SEGMENTS:
        clip = compress(seg)
        pos = poster(seg)
        print(f"[gallery] {clip.name}: {_size_mb(clip):.2f} MB  poster {pos.name}")
    reel = build_reel()
    print(f"[gallery] {reel.name}: {_size_mb(reel):.2f} MB")
    gif = make_gif()
    print(f"[gallery] {gif.name}: {_size_mb(gif):.2f} MB")


if __name__ == "__main__":
    main()
