# Multi-Robot Warehouse — Architecture Plan

48-hour build on top of the single-robot G1Nav baseline. Headline deliverable:
recruiter-grade demo videos of 4 named Unitree G1 humanoids collaborating in a
warehouse — occluded object search, addressed robot↔robot communication, and
optimal fetch-to-destination.

## 1. Goals (from user spec)

1. ≥4 identical Unitree G1 robots, each individually addressable by name.
2. Warehouse space with walls/shelving that block sightlines between robots.
3. Communication: user↔any-robot and robot↔robot, explicitly addressed
   (need-to-know sharing: if Bravo finds Alpha's object, Bravo tells Alpha
   only; Alpha owns reporting back to the user).
4. Canonical task: user asks robot X to fetch object O to destination D.
   X checks own view → queries fleet → delegates search → receives location
   → plans optimal path → fetches → delivers → reports.
5. Fleet-level task ("someone bring me O") → allocator picks robots by cost;
   collaborative divide-and-conquer search when nobody sees O.
6. Robot poses are exactly known (warehouse localization assumption); object
   poses are NOT — objects must be seen by a robot camera to be located.
7. Path optimality: occupancy-grid A* (shortest collision-free routes),
   fleet allocation minimizes makespan/total travel.

## 2. Warehouse scenario (demo-grade, user-specified 2026-07-12)

Design intent: *open enough that robots move freely; walled enough that
objects are usually hidden from any single viewpoint.*

- **Hall:** ~16 m × 12 m, 2.5 m perimeter walls, clean industrial look
  (light-gray floor, subtle grid texture, overhead key lights + BEV cam).
- **Shelf rows:** two double-sided shelf rows (≈4.5 m × 0.7 m × 1.8 m each)
  running lengthwise, each split by a 1.4 m mid-row gap → a "racetrack +
  crossovers" topology: three lengthwise aisles ≥2.2 m wide, multiple
  alternative routes (makes A* choices visible in demos).
- **Partitions:** one L-shaped wall forming an occluded alcove in the NE
  corner; one short freestanding wall near the SW quadrant. Objects placed
  in aisles/alcoves are invisible from most of the hall.
- **Zones:** 2 m × 2 m green delivery pad (the "destination"); four colored
  home bays along the south wall, one per robot, labeled.
- **Robots:** callsigns **Alpha, Bravo, Charlie, Delta** (readable in demos,
  trivially extensible to Echo…). Color-accented per robot (helmet/pad tint)
  so viewers track who is who; same G1 model + walk policy for all.
- **Objects:** reuse baseline object set (colored cubes/balls/cones);
  spawn presets: hero-demo layout (fixed, tuned for video) + seeded random
  layouts (for evals).
- Layout is parameterized in code (wall list → both MJCF geoms and the
  planner occupancy grid derive from the SAME source of truth, so sim and
  planning can never skew).

## 3. Baseline integration map (from codebase survey, 2026-07-12)

- **Scene build:** programmatic MjSpec in `code/sim/arena_build.py::build_arena`
  (robot XML via `MjSpec.from_file(G1_XML)` from `third_party/`, perimeter
  walls added at ~L219-239). Interior walls slot in right after that block,
  driven by a new `scene_cfg["walls"]` key. `scene_cfg` dict is the universal
  contract consumed by every rollout loop.
- **No Robot class.** All state access is literal slicing (`qpos[0:3]` pelvis
  xyz, `qpos[3:7]` quat, `qpos[7:22]` joints, `ctrl[:15]`) assuming the robot
  is the whole model, plus a hardcoded `"pelvis"` name at 10+ sites.
  → **Decision: do NOT refactor.** Federated physics: one `MjModel`/`MjData`
  per robot (robot is alone in its model, baseline code valid verbatim) + one
  shared **viz/perception model** with warehouse + N kinematically-synced
  robot bodies (mocap-style) for cross-visibility and fleet BEV video.
  Robot-robot collision handled by planner reservation + AVOID, not contact
  physics (documented limitation).
- **Cameras are procedural free cameras** positioned from qpos each frame
  (`arena_cameras.py`) — no XML duplication needed; per-robot views = same
  math at robot i's pose against the shared viz model.
- **Perception:** `HeatmapDetector` (`code/perception/detector/model.py`) is
  weights-only (shareable); but `grounding.py:110` holds a process-wide
  `GroundNetState` singleton (track hysteresis, heatmap cache) →
  must become per-robot instances. `LockGate`/`ScanSchedule`/AVOID are already
  per-instance — just construct N.
- **Nav:** `control/steer.py` is single-target proportional steering; no
  waypoint or occupancy/A* code exists anywhere. `WBCTeacher` walks from
  velocity commands (how datagen drives it) → warehouse nav = teacher-driven
  waypoint following; the distilled VLA/GROUND_NET stack provides grounding.
- **Rollout loops duplicated 4-5×** (deliberate baseline invariant). New
  warehouse rollout gets its OWN loop in `code/apps/warehouse_demo/` rather
  than threading hooks into all five.
- **Video:** BEV follow-cam constants (`apps/fancy/constants.py`, 17 m /
  -43.5°) are single-robot-tuned; fleet BEV needs a fixed wide framing for
  the 16×12 m hall. EGL GPU rendering auto-setup already in
  `arena_build.py:28-38` (GB10 vendor-race fix included).
- **Instruction parsing:** two existing resolvers (fancy `resolve_live_instruction`,
  REPL `Planner.parse`). Addressing layer ("Alpha, fetch…") sits UPSTREAM in
  `code/fleet/`: parse callsign/fleet-address → route rest to a per-robot
  resolver against that robot's own knowledge.

## 3b. External assets (not committed; README prerequisites)

`third_party/` (G1 XML + WBC ONNX) and `runs/nx6_heatmap_B/` (GROUND_NET ckpt)
are symlinked from the local dev checkout. Fresh-clone setup follows the same
README steps as the baseline. Baseline suite verified green in this repo
(1039 tests OK) with links in place.

## 4. New packages (planned)

- `code/warehouse/` — layout spec, MJCF assembly, occupancy grid export.
- `code/planner/` — occupancy-grid A*, path smoothing, waypoint following
  glue onto the existing maneuver/command interface; multi-robot reservation
  (avoid robot-robot collisions at aisle crossings).
- `code/fleet/` — Robot identity/state registry, per-robot policy+perception
  instances, message bus (typed, addressed messages, full transcript log for
  demo overlays), coordination protocols (query-visibility, delegate-search,
  report-found, task allocator).
- `code/apps/warehouse_demo/` — scripted + interactive demo entry points,
  video recording with comms-overlay captions.
- Each package gets a sibling `tests/` dir (stdlib unittest), Google-style
  docstrings, full type hints — same conventions as the baseline's 1039 tests.

## 5. Phases

| Phase | Deliverable | Gate |
|---|---|---|
| 0 | infra (watchdog, progress.md, baseline tests green in new repo) | suite passes |
| 1 | warehouse arena + walls + 1 robot A*-navigating it | nav eval ≥90% to random reachable goals |
| 2 | 4 named robots co-simulated, per-robot policy/perception | 4 robots walk simultaneously w/o falls/interference |
| 3 | message bus + protocols (addressed comms) | scripted comm scenario passes; transcript correct |
| 4 | collaborative search + fetch (mock pickup: attach within radius) | end-to-end fetch success on seeded evals |
| 5 | hero demo videos + README + gallery | watchable MP4s pushed; fresh-clone repro (VR-1 rule) |

## 5b. Demo art-pass backlog (from Phase-1a render review, 2026-07-12)

Layout is structurally correct (occlusion verified from robot-eye cams), but
for recruiter-grade footage the hero scene needs:
- Floor: replace blue checker with light-gray industrial look inside the hall.
- Shelves: wood/gray tones (current yellow collides with Charlie's bay color);
  consider taller/longer blocks — current 2×(2×1.55 m) rows read sparse in the
  16 m hall. Target "complicated but open": lengthen blocks toward ~3 m each
  and/or add scattered crate props as extra occluders.
- SW partition sits close to Alpha's bay — nudge for cleaner sightline story.
- Lighting: warmer key light; check shadows at BEV angle.

Hardening backlog (from planner adversarial review, 2026-07-12):
- Hero layout verified reachable for all spawn→object_spot pairs at inflation
  ≤0.45 m (alcove seals only at 0.50 m). `sample_layout` variants must run the
  same reachability check at the deployed inflation radius before use.
- Fetch/task layer should rasterize non-target objects as small obstacles so
  paths don't clip them (cosmetic now, matters for multi-object scenes).
- Follower arrive_radius vs steer stop_r interplay is caller-owned: nav loops
  must pass steer stop_r < follower arrive_radius (Phase 1c tunes this).

## 6. Risks / known constraints

- 4 humanoids in one MjModel: physics cost scales ~linearly; policy inference
  batches across robots on GPU. Watch step-time; GB10 is shared with other
  user jobs (currently ~95% util from unrelated processes).
- Walk policy is OOD-sensitive (>~470-step continuous rotations fall; spawn
  translational drift during large early rotations). Waypoint headings must
  respect the same bounded-scan/turn hygiene the baseline learned (NX-10/12).
- "Bring it" = mock pickup (USER-CONFIRMED 2026-07-12): no grasping policy —
  when the robot is within reach radius of the object, the object is placed
  onto the robot's hand link and kinematically re-posed to that hand's world
  pose every control step (object moves with the robot while carried);
  released onto the delivery pad at the destination. Implementation: update
  the object's geom/body pose from the hand body xpos each step in the
  robot's physics model and mirror it in the shared viz model. Pick a stable
  G1 hand/wrist body from the XML; offset so the object rests visibly in the
  hand. Document the assumption prominently in README/demos.
- VR-1 release rule: before any push that claims a working command, run that
  command end-to-end from a fresh clone.

## 7. STATUS — complete (2026-07-12)

All six phases landed; this section is the finished record. Numbers are from the
released run on the fixed hero layout (artifacts under `eval/`).

| Phase | Deliverable | Status | Evidence |
|---|---|---|---|
| 0 | Infra: watchdog, progress log, baseline suite green in new repo | ✅ 07-12 | 1039 baseline tests OK with externals symlinked |
| 1 | Warehouse arena + walls + 1 robot A\*-navigating it | ✅ 07-12 | nav **10/10** (20/20 2nd seed), 0 falls, path eff **1.00**, A\* ~22 ms |
| 2 | 4 named robots co-simulated, per-robot policy/perception | ✅ 07-12 | fleet nav **5/5** all-arrive, 0 falls; cross-visibility proven; 4-robot step 2.2–3.1 ms |
| 3 | Message bus + addressed comms protocols | ✅ 07-12 | need-to-know enforced structurally; deterministic (PYTHONHASHSEED-stable) tests |
| 4 | Collaborative search + fetch (mock pickup) | ✅ 07-12 | missions A/B/C **30/30** (10 seeds), allocator D **3/3** optimal, 0 falls; same-seed transcript byte-identical |
| 5 | Hero videos + README + gallery + fresh-clone repro | ✅ 07-12 | gallery MP4s/posters/GIF committed; flagship render ~55 s; **VR-1 rehearsal passed end-to-end** |

**Final test suite:** 1327 tests OK (7 skip without external assets).

**VR-1 fresh-clone rehearsal (Phase 5b).** Every command the README documents was
run verbatim in a fresh `git clone`, with only `third_party/` and
`runs/nx6_heatmap_B/` symlinked in (the documented external-assets step): full
suite (1327 OK), `mission_eval --seeds 1` (3/3 A-C, 3/3 D, 0 falls),
`mission_video --scenario C` (complete, on-pad, 0 falls), `fleet_video`
(4/4 arrive + cross-visibility proof), `fleet_eval --n 1` (1/1), `nav_eval`
(2/2). All exited 0 and produced their artifacts. The rehearsal also corrected a
stale README figure (cross-visibility, cited 6.6% pre-art-pass, reproduces ~3.3%
on the released scene) — exactly the class of drift VR-1 exists to catch.

## 8. STATUS addendum — F1–F6 final-demo cycle (2026-07-12 → 07-13)

After the Phase-5 release, the user specified six final-demo requirements
(`docs/final_demo_spec.md`, F1–F6). All six landed, plus two in-repo domain
fine-tunes and an interactive web demo. Numbers below are the released record.

| Feature | Deliverable | Status | Evidence / commit |
|---|---|---|---|
| F6 | Multi-room `rooms_layout` (4 named rooms, 4 open doorways, Room API = single source of truth for search regions + spoken room names) | ✅ 07-12 | 44/44 bay→spot reachability at 0.40/0.45; `8fa1782` |
| F5-p1 | Warehouse-domain DART datagen (`code/apps/warehouse_datagen`) | ✅ 07-12 | 200 eps / 125,683 fr, 98.5% ep success; EGL GPU fix (160 ms→0.88 ms/frame); `821b06f` |
| Cyc-2a | Real GROUND_NET detector in the mission loop (per-robot state, oracle occlusion gate + learned confirmer) | ✅ 07-12 | groundnet missions 12/12, 0 falls; measured domain-shift confidence collapse → fine-tune queued; `72eeb54` |
| Cyc-2b | Interactive live web demo (`code/apps/fleet_web`) | ✅ 07-12 | MJPEG BEV + live transcript + command box; `ad0e206` |
| F1–F4 | Comm-emphasized ego insets (F1), deferred target ring at reported position (F2), fixed relative-position report sentence (F3), generic fleet commands (F4); room-aware search through doorways; `MissionRunner` lifecycle API | ✅ 07-12 | rooms C 9/9 + D 3/3, hero 6/6 + 3/3, 0 falls; `4d7b43b` |
| GN-FT | GROUND_NET **warehouse fine-tune** (11,630-frame det set, NX-6 recipe, from scratch, best ep 12) | ✅ 07-13 | detection **8.8% → 100%**, xy p90 0.037 m, confirm confidence p50 **0.243 → 0.894**, 0 through-wall hallucinations; ckpt auto-resolution (`GROUND_NET_CKPT` env → warehouse_ft → baseline); `8c65b91` |
| F5 | **VLA locomotion backend** — 20-epoch fine-tune of the deployed distilled walk policy on the warehouse DART set; teacher `\|` vla switch on the nav engine (teacher = datagen + standing balance only) | ✅ 07-13 | val_action 0.0886 → 0.0612; closed-loop select over 5 ckpts (ep19); gate **30/30** (hero 20/20 ×2 seeds + rooms 10/10), 0 falls, eff ≥ 0.994, 1.6 ms/step; `3bb34cf` |
| Integ | Fleet on VLA locomotion + final demo video set | ✅ 07-13 | **Full learned stack (rooms + fine-tuned detector + VLA): C 12/12, D 3/3, 0 falls across 24 missions**, 1.7 ms/step; hero 6/6 + 3/3; interaction fix (VLA walks, WBC balances held stands); four `final_*` gallery videos + rebuilt reel/GIF; `07d96c9` |

**Final test suite:** 1450 tests OK (7 skip with all assets present; a fresh clone
without the two trained fine-tunes runs 1449 OK, 11 skip).

**VR-1 re-rehearsal (final release, 2026-07-13).** Fresh `git clone` with only
`third_party/` + `runs/nx6_heatmap_B/` symlinked (the two trained fine-tunes
deliberately absent) verified the documented fallback story end-to-end: full suite
(1449 OK, 11 skip), `mission_eval --layout rooms --seeds 1` (oracle+teacher, C 2/2
+ D 3/3, 0 falls), `mission_video --scenario C --layout rooms` (complete, on-pad,
0 falls), `fleet_web` (served, one addressed command completed, clean kill). Then
symlinking the two trained-ckpt dirs (simulating "after training") the full learned
stack ran: `mission_eval --layout rooms --perception groundnet --locomotion vla
--seeds 1`. All exited 0. Confirms fresh clones run every demo in the
teacher/oracle fallback and upgrade cleanly once the checkpoints exist.
