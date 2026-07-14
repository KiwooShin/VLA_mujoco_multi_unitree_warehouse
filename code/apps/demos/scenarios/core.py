"""core.py — Shared framework for the Demo Set v2 scenario scripts.

A :class:`Scenario` is a deterministic, self-contained recipe for one demo: it
picks the layout (fixed ``rooms`` / ``rooms6`` or a seeded ``sample_rooms_layout``),
plants the scene objects (controlling manifest ambiguity so an ambiguous order's
CLARIFY lists exactly the intended options), seeds the random robot spawns, and
carries the natural-language order plus any scripted CLARIFY replies / mid-mission
re-task schedule. It knows how to (a) build a :class:`~code.fleet.mission.MissionRunner`,
(b) run a no-video *probe* to confirm the story lands on a seed, and (c) record a
polished original via :class:`~code.apps.demos.DemoComposer` /
:class:`~code.apps.demos.DemoRecorder`.

The framework NEVER edits the composer / recorder / mission / comms code — it only
composes their public surfaces (``submit`` / ``submit_multi`` / ``retask`` /
``mission_outcomes`` / ``known_targets_xy`` …) and controls the scene it feeds them.
"""

from __future__ import annotations

import dataclasses
import math
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from code.comms.messages import ObjectQuery, clarify_options
from code.sim.arena_build import COLORS

XY = Tuple[float, float]
ColorShape = Tuple[str, str]

_CMAP: Dict[str, Tuple[int, int, int]] = dict(COLORS)

# A varied, manifest-safe pool of distinct filler objects (never a cube or ball so
# it cannot pollute the two ambiguity queries the demos use — "the cube" / "a
# ball"). Cycled across the un-planted object spots to make each room look busy.
DEFAULT_FILLERS: Tuple[ColorShape, ...] = (
    ("orange", "cylinder"), ("green", "cone"), ("purple", "cylinder"),
    ("cyan", "cone"), ("yellow", "cylinder"), ("blue", "cone"),
    ("green", "cylinder"), ("orange", "cone"), ("purple", "cone"),
    ("cyan", "cylinder"),
)


def _obj(color: str, shape: str, x: float, y: float, size: float = 0.24) -> dict:
    """Build one scene-object dict in the runner's expected schema."""
    return {"color_name": color, "color_rgb": _CMAP[color], "shape_name": shape,
            "size": float(size), "x": float(x), "y": float(y)}


def build_objects(layout, planted: Dict[int, ColorShape],
                  filler_pool: Sequence[ColorShape] = DEFAULT_FILLERS) -> List[dict]:
    """Place objects at every ``layout.object_spot``; ``planted`` pins specific ones.

    Args:
        layout: A layout whose ``object_spots`` fix the placement grid.
        planted: ``spot_index -> (color, shape)`` for the demo's target(s) and
            any decoys that shape the CLARIFY options.
        filler_pool: Distinct ``(color, shape)`` pairs cycled over the remaining
            spots (kept clear of the ambiguity shapes by :data:`DEFAULT_FILLERS`).

    Returns:
        The full scene-object list (one per spot), in spot order.
    """
    objs: List[dict] = []
    fi = 0
    for i, (x, y) in enumerate(layout.object_spots):
        if i in planted:
            c, s = planted[i]
        else:
            c, s = filler_pool[fi % len(filler_pool)]
            fi += 1
        objs.append(_obj(c, s, x, y))
    return objs


def manifest_of(objects: Sequence[dict]) -> List[dict]:
    """The (color, shape) manifest a robot/allocator sees — positions stripped."""
    return [{"color_name": o["color_name"], "shape_name": o["shape_name"]}
            for o in objects]


def deep_spots(layout, n: int, avoid_rooms: Sequence[str] = ("loading room",)
               ) -> List[int]:
    """Pick ``n`` deep object spots, each in a DISTINCT non-``avoid`` room.

    Ranks the spots by distance from the hall centre (deepest first) and greedily
    takes the deepest spot per room, so ``n`` targets land in ``n`` different
    searchable rooms — the room-to-room-exploration story a sampled layout tells.
    """
    from code.warehouse.layout import room_of
    order = sorted(range(len(layout.object_spots)),
                   key=lambda i: -math.hypot(*layout.object_spots[i]))
    picked: List[int] = []
    used_rooms: set = set()
    for i in order:
        room = room_of(layout, layout.object_spots[i])
        if room in avoid_rooms or room in used_rooms:
            continue
        picked.append(i)
        used_rooms.add(room)
        if len(picked) == n:
            break
    return picked


@dataclasses.dataclass
class Scenario:
    """One deterministic Demo Set v2 recipe (world + order + recording knobs)."""

    name: str
    title: str
    description: str

    # -- world ---------------------------------------------------------------
    layout_kind: str = "rooms"                 # "rooms" | "rooms6" | "sampled"
    sample_seed: int = 0                       # sampled layouts only
    spawn_seed: int = 0
    planted: Dict[int, ColorShape] = dataclasses.field(default_factory=dict)
    # Layout-derived planting for sampled layouts (spot indices are only known
    # once the layout exists): ``planted_fn(layout) -> {spot_index: (color, shape)}``.
    planted_fn: Optional[Callable] = None
    filler_pool: Tuple[ColorShape, ...] = DEFAULT_FILLERS

    # -- order ---------------------------------------------------------------
    instruction: str = ""
    replies: Tuple[str, ...] = ()
    retasks: Tuple[Tuple[int, str], ...] = ()
    concurrent: bool = False                   # use submit_multi (two owners)

    # -- ambiguity precondition (asserted by the tests, no sim) --------------
    ambiguity_query: Optional[ObjectQuery] = None
    ambiguity_options: Tuple[str, ...] = ()

    # -- panel curation ------------------------------------------------------
    # Performative names to hide from the transcript PANEL (presentation only —
    # the mission still exchanges them). Used on the six-robot flagship to drop
    # the ten "can you see it? / no" visibility lines that would otherwise bury
    # the opening CLARIFY under auto-scroll (the whole exchange fires in <10
    # sim-steps, so the panel can only ever show its tail).
    hide_performatives: Tuple[str, ...] = ()

    # -- learned stack -------------------------------------------------------
    perception_mode: str = "groundnet"
    locomotion: str = "vla"

    # -- recording knobs -----------------------------------------------------
    max_steps: int = 4000
    decimation: int = 5
    fps: int = 30
    title_vid_secs: float = 2.4                # readable title-card time (video s)
    clarify_deadline_steps: int = 900
    reply_deadline_steps: int = 90
    search_deadline_steps: Optional[int] = None

    # -- gif / poster --------------------------------------------------------
    gif_ss: float = 4.0
    gif_dur: float = 18.0
    poster_t: float = 6.0

    # ------------------------------------------------------------------ world
    def make_layout(self):
        """Build this scenario's layout deterministically."""
        from code.warehouse.layout import (rooms6_layout, rooms_layout,
                                            sample_rooms_layout)
        if self.layout_kind == "rooms":
            return rooms_layout()
        if self.layout_kind == "rooms6":
            return rooms6_layout()
        if self.layout_kind == "sampled":
            return sample_rooms_layout(np.random.default_rng(self.sample_seed))
        raise ValueError(f"unknown layout_kind {self.layout_kind!r}")

    def make_objects(self, layout) -> List[dict]:
        """The scene objects for this scenario on ``layout``."""
        planted = self.planted_fn(layout) if self.planted_fn else self.planted
        return build_objects(layout, planted, self.filler_pool)

    def manifest(self, layout=None) -> List[dict]:
        """The colour/shape manifest for this scenario (positions stripped)."""
        layout = layout or self.make_layout()
        return manifest_of(self.make_objects(layout))

    def clarify_preview(self, layout=None) -> List[str]:
        """The CLARIFY options the ambiguity query yields against the manifest."""
        if self.ambiguity_query is None:
            return []
        return clarify_options(self.ambiguity_query, self.manifest(layout))

    def check_precondition(self) -> None:
        """Assert the manifest-ambiguity contract holds (raises on violation)."""
        if self.ambiguity_query is None:
            return
        got = self.clarify_preview()
        want = list(self.ambiguity_options)
        if sorted(got) != sorted(want):
            raise AssertionError(
                f"{self.name}: CLARIFY options {got} != expected {want}")

    # ---------------------------------------------------------------- runner
    def _title_card_secs(self) -> float:
        """Sim seconds that render ~``title_vid_secs`` of a readable title card."""
        # video_secs = (sim_secs / sim_dt) / decimation / fps  ->  invert.
        return self.title_vid_secs * self.fps * self.decimation * 0.02

    def build_runner(self):
        """Construct the MissionRunner + return ``(mr, layout, callsigns)``."""
        from code.fleet.mission import MissionRunner
        from code.warehouse.layout import callsigns_for_layout
        layout = self.make_layout()
        callsigns = list(callsigns_for_layout(layout))
        objects = self.make_objects(layout)
        mr = MissionRunner(
            layout=layout, objects=objects, callsigns=callsigns, seed=0,
            use_gpu=True, perception_mode=self.perception_mode,
            locomotion=self.locomotion, spawn_seed=self.spawn_seed,
            search_deadline_steps=self.search_deadline_steps or self.max_steps,
            clarify_deadline_steps=self.clarify_deadline_steps,
            reply_deadline_steps=self.reply_deadline_steps)
        return mr, layout, callsigns

    def submit(self, mr) -> None:
        """Post the order (single / concurrent) and register any re-tasks."""
        replies = list(self.replies) or None
        if self.concurrent:
            mr.submit_multi(self.instruction, replies=replies)
        else:
            mr.submit(self.instruction, replies=replies)
        for at_step, text in self.retasks:
            mr.retask(text, at_step)

    def make_state_fn(self) -> Optional[Callable]:
        """Custom FrameState builder (per-owner concurrent rings + panel curation).

        Returns ``None`` when the default adapter suffices; otherwise a closure
        that colours the concurrent goal rings by owner and/or hides the
        configured performatives from the transcript panel.
        """
        if not self.concurrent and not self.hide_performatives:
            return None
        hide = set(self.hide_performatives)
        base = _concurrent_state_fn if self.concurrent else None

        def _state_fn(mr, step: int):
            if base is not None:
                fs = base(mr, step)
            else:
                from code.apps.demos.runner_adapter import frame_state_from_runner
                fs = frame_state_from_runner(mr, step)
            if hide:
                kept = [m for m in fs.transcript
                        if getattr(getattr(m, "performative", None), "name", "")
                        not in hide]
                fs = dataclasses.replace(fs, transcript=kept)
            return fs

        return _state_fn

    # --------------------------------------------------------------- probing
    def probe(self, max_steps: Optional[int] = None, verbose: bool = True) -> dict:
        """Run the mission WITHOUT video; return a story-verification summary.

        Confirms the seed lands: the outcome, whether a CLARIFY was raised (and
        answered), how many objects reached the pad, and the completion step —
        everything needed to accept/reject a seed before the expensive record.
        """
        from code.comms.messages import Performative
        mr, _layout, _cs = self.build_runner()
        self.submit(mr)
        res = mr.run(max_steps or self.max_steps)
        tr = mr.bus.transcript
        clarifies = [m for m in tr if m.performative is Performative.CLARIFY]
        replies = [m for m in tr if m.performative is Performative.USER_REPLY]
        summary = {
            "name": self.name,
            "spawn_seed": self.spawn_seed,
            "outcome": res.outcome,
            "steps": res.steps,
            "any_fell": res.any_fell,
            "object_on_pad": res.object_on_pad,
            "clarify_questions": [m.payload.get("question") for m in clarifies],
            "n_clarify": len(clarifies),
            "n_user_reply": len(replies),
            "mission_outcomes": mr.mission_outcomes() if self.concurrent else None,
        }
        mr.close()
        if verbose:
            print(summary)
        return summary


# ---------------------------------------------------------------------------
# Concurrent ring colouring: colour + label each of the two goal rings by its
# owning mission so the "two owners in parallel" story is unmistakable on the BEV.
# ---------------------------------------------------------------------------
def _concurrent_state_fn(mr, step: int):
    """FrameState for concurrent demos: one accent-coloured, labelled ring/owner."""
    from code.apps.demos import style
    from code.apps.demos.models import Ring
    from code.apps.demos.runner_adapter import frame_state_from_runner
    fs = frame_state_from_runner(mr, step)
    missions = list(getattr(mr, "_missions", []))
    positions = mr.known_targets_xy()
    rings: List[Ring] = []
    for i, xy in enumerate(positions):
        if xy is None:
            continue
        owner = missions[i].owner if i < len(missions) else None
        label = missions[i].task.query.describe() if i < len(missions) else ""
        color = style.accent_bgr(owner) if owner else None
        rings.append(Ring((float(xy[0]), float(xy[1])), color=color, label=label))
    return dataclasses.replace(fs, rings=rings)
