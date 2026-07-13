# Progress Log — VLA_mujoco_multi_unitree_warehouse

48-hour autonomous build (2026-07-12 16:20 → 2026-07-14 ~16:30).
Goal: multi-robot (4+ named Unitree G1 humanoids) warehouse sim with
view-blocking walls, user↔robot and robot↔robot communication, collaborative
object search-and-fetch, and optimal path planning — with recruiter-grade demo
videos as the headline deliverable.

Reporting cadence: one entry every ~2 hours (what was done / next-2h plan /
analysis / performance table).

---

## 2026-07-12 16:20 — Session start (entry 1)

### Done
- Cloned empty remote `KiwooShin/VLA_mujoco_multi_unitree_warehouse`; rsync'd
  the full single-robot baseline from `VLA_mujoco_unitree` (verified
  byte-identical, 26 MB). Baseline = refactored 11-package `code/` tree,
  1039 unittests, distilled G1 walk policy, GROUND_NET detector, two-camera
  perception, scan/lock/avoid stack.
- Verified env: conda `g1nav` (py3.10), mujoco 3.9.0, torch 2.7.1+cu128,
  CUDA available on GB10.
- GPU/process watchdog live: `tools/watchdog.sh` → `ops/watchdog.log`,
  5-min cadence, ALERT lines for halted (D/Z), stalled (frozen cputime), and
  CPU-instead-of-GPU (>90% CPU while GPU <5%) processes.
- Launched codebase-mapping sub-agent over the baseline to produce the
  architecture map (arena construction, robot abstraction, rollout loops,
  perception singletons, apps/resolver, video/eval plumbing).

### Next 2 hours
- Architecture plan (`docs/multi_plan.md`): warehouse arena spec (walls,
  shelves, spawn points), N-robot MJCF namespacing, per-robot policy/percep
  instances, occupancy-grid + A* planner, message-bus protocol
  (addressed user↔robot / robot↔robot), task allocator.
- Phase 1 kickoff via Opus execution sub-agent: warehouse arena + walls +
  single robot navigating it with A* waypoints (the multi-robot layer lands
  on top of a working warehouse).
- Initial + phase-1 commits pushed to remote.

### Analysis
- Baseline is single-robot everywhere: one MJCF robot tree, global camera
  names, singleton lock/scan state. Multi-robot needs disciplined namespacing
  (`r1_`, `r2_`, …) and per-robot state objects — mapping agent is confirming
  exact touchpoints.
- GB10 nvidia-smi reports memory as [N/A] (unified memory) — watchdog keys
  off utilization + compute-app list instead of memory.

### Performance
| Metric | Value | Notes |
|---|---|---|
| Baseline tests passing | pending first run in new repo | 1039 expected |
| Warehouse nav success | — | not built yet |
| Multi-robot fetch success | — | not built yet |

---

## 2026-07-12 17:50 — Phase 1 complete (entry 2)

### Done (last ~1.5 h)
- Baseline verified in new repo: 1039 tests green after symlinking gitignored
  externals (third_party/ G1+WBC, GROUND_NET ckpt) from the dev checkout.
- **Phase 1a (warehouse)**: code/warehouse/ — hero 16×12 m layout (2 split
  shelf rows → 3 aisles ≥3.3 m + crossover, NE occluded alcove, SW partition,
  delivery pad, 4 named home bays), occupancy rasterizer sharing the wall list
  with the MJCF (single source of truth), demo-ready arena builder. 63 tests.
- **Phase 1b (planner)**: code/planner/ — 8-connected A* (no corner cutting,
  deterministic), supercover line-of-sight smoothing, pure-pursuit
  WaypointFollower. 49 tests; ~22 ms/plan on the 160×120 hall grid.
- Planner adversarially reviewed (27-agent workflow): 0 confirmed defects of
  12 raw findings; I empirically re-checked the 2 scariest rejections
  (alcove reachable ≤0.45 m inflation; radius interplay is caller-owned).
- **Phase 1c (integration)**: code/apps/warehouse_demo/ — WBCTeacher walks
  steer()-commanded A* paths. Key reliability fix: stamp non-goal objects
  into the planning grid (the only observed fall mode was object collision).
  OOD-turn guard added (never triggered; max observed turn run 41 steps
  vs ~470 OOD limit). 29 tests. BEV videos with path overlay + ego PiP in
  ops/phase1c/ (verified frames myself: readable fleet-scale framing).
- Contract fixes: scene_cfg now carries exact hall_x/hall_y; consumer prefers
  them. Suite total **1181 OK**.
- User locked in mock-pickup spec (object snaps to hand link when close,
  kinematically carried, released at pad) — recorded in docs/multi_plan.md.

### Next 2 hours
- **Phase 2**: code/fleet/ — 4 named robots co-simulated (federated physics:
  one MjModel per robot; shared kinematic viz model for cross-visibility +
  fleet BEV), RobotUnit stepwise API refactor of nav_rollout, 4-robot
  simultaneous-nav eval + video. Then start Phase 3 message bus.

### Analysis
- Nav is robust: 10/10 (and 20/20 on 2nd seed), zero falls/wall hits, path
  efficiency 1.00, min clearance 0.24 m at 0.40 m inflation. Step cost:
  0.52 ms/step headless, 18 ms/step with BEV+ego rendering → 4 robots
  headless ≈2 ms/step is trivially real-time; rendering dominates videos.
- onnxruntime here has no CUDAExecutionProvider → WBC teacher ONNX runs on
  CPU at 0.5 ms/step (fine; not a GPU-waste alert — flagged for awareness).
  GROUND_NET + rendering are the GPU consumers.
- Watchdog: 0 ALERTs; unrelated user GPU jobs ended ~16:58 (GPU now ours).

### Performance
| Metric | Value | Notes |
|---|---|---|
| Test suite | 1181 OK (7 skip) | baseline 1039 + 142 new |
| Nav success (hero, bay→occluded spot) | 10/10 (20/20 seed B) | 0 falls, 0 wall hits |
| Path efficiency (planned/walked) | 1.00 | pelvis tracks smoothed A* tightly |
| Plan time | ~22 ms | 160×120 grid, 0.1 m |
| Sim step | 0.52 ms headless / 18 ms rendered | 50 Hz control budget 20 ms |
