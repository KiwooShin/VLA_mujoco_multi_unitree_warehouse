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

## 6. Risks / known constraints

- 4 humanoids in one MjModel: physics cost scales ~linearly; policy inference
  batches across robots on GPU. Watch step-time; GB10 is shared with other
  user jobs (currently ~95% util from unrelated processes).
- Walk policy is OOD-sensitive (>~470-step continuous rotations fall; spawn
  translational drift during large early rotations). Waypoint headings must
  respect the same bounded-scan/turn hygiene the baseline learned (NX-10/12).
- "Bring it" = mock pickup (weld/attach object when within reach radius,
  release on pad) — no manipulation policy in the baseline; document clearly.
- VR-1 release rule: before any push that claims a working command, run that
  command end-to-end from a fresh clone.
