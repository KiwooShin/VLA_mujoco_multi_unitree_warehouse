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

---

## 2026-07-12 19:30 — Phases 2, 3, 4 complete (entry 3)

### Done (last ~1.7 h)
- **Phase 2 (fleet co-sim)**: code/fleet/ — RobotUnit stepwise engine
  (nav_core refactor, no duplicated loop), Fleet with deadlock-free
  proximity-pause (1.0/1.2 m hysteresis, callsign priority), shared viz
  model (prefixed MjSpec attach, kinematic qpos sync, per-robot tint).
  Cross-visibility PROVEN (6.6% ego pixel diff; robots see each other).
  fleet_eval 5/5 all-arrive, 0 falls. 4-robot stepping 1.8 ms headless.
- **Phase 3 (comms)**: code/comms/ — typed performatives, FIFO bus with
  human-readable transcript, RobotProtocol state machine with need-to-know
  enforced structurally (helpers can't message user), addressing parser.
  41 deterministic tests (PYTHONHASHSEED-stable).
- **Phase 4 (missions)**: code/fleet/ mission layer — geometric visibility
  oracle (FOV+range+wall LOS), region search (N/M/S thirds, reachable
  patrols), mock pickup/carry on right_wrist_yaw_link (user-confirmed
  mechanic), A*-length-optimal allocator, MissionRunner, flagship video
  with live comms panel. Canonical scenario runs END-TO-END on video.
- Watchdog caught a real perf issue (117% CPU, GPU 0): root cause was
  unbounded BEV trail re-projection (86.8% of step cost), NOT sim —
  fixed (video 38.2→7.1 ms/step; flagship render 11.5 min→55 s).
- Pushed: 95a47c1 (Phase 2), 3f2fda2 (comms), 55bb8bc (Phase 4).

### Next 2 hours
- 10-seed mission reliability eval (running in background).
- **Phase 5a**: warehouse art pass (floor/shelf/lighting per §5b backlog,
  cosmetics only — geometry frozen) + re-record hero videos + compressed
  gallery assets (assets/gallery/).
- **Phase 5b**: recruiter-facing README rewrite + fresh-clone VR-1
  rehearsal + final push.

### Analysis
- All 4 mission classes green first try: A (owner sees) 4/4, B (peer sees)
  4/4, C (delegated search) 4/4, D allocator picks true A*-argmin robot 3/3;
  0 falls across ~24 missions. Determinism: same-seed scenario-C double run
  is byte-identical (6,700 steps).
- Worst-case mission (alcove object) completes at 6,700/9,000 step budget —
  bounded, no timeout risk at current warehouse scale.
- Protocol pump + visibility oracle cost is negligible (0.012 ms/robot-step);
  physics+viz sync 2.64 ms/step for 4 robots → headless missions 2.2 ms/step.

### Performance
| Metric | Value | Notes |
|---|---|---|
| Test suite | 1305 OK (7 skip) | +48 mission tests |
| Fleet nav (4 robots simultaneous) | 5/5, 0 falls | mean makespan 1257 steps |
| Missions A/B/C | 12/12 | seeds=4 per class |
| Allocator correctness (D) | 3/3 | vs independent A* argmin |
| Falls (all missions) | 0 | ~24 missions |
| Headless mission step | 2.2 ms | 4 robots + protocol |
| Flagship video render | 55 s | was 11.5 min pre-fix |

---

## 2026-07-12 20:50 — Release: all 5 phases complete (entry 4)

### Done (last ~1.3 h)
- **10-seed mission eval**: A/B/C 30/30, allocator 3/3, 0 falls, 177 s.
- **Adversarial review round 2** (mission layer, 25 agents): 0 findings
  survived the "matters for current evals" gate, but 7 mechanisms were
  code-confirmed by reviewers → all 7 fixed (+22 regression tests):
  TASK_FAILED on owner fall/unreachable goal (no more mission hangs),
  pickup/search booleans honored (no false TASK_COMPLETE), fallen carrier
  drops the object, fleet requests queue instead of vanishing, mid-mission
  submit() raises, DEFAULT_REGIONS aligned, NaN-pose allocator guard.
  Eval unchanged after fixes (12/12, 3/3, 0 falls); determinism preserved.
- **Art pass**: root-caused the blue floor (G1 XML checker texture beats
  geom_rgba) — both model variants now render identical clean industrial
  look; two-light rig; desaturated pads. Object palette/geometry untouched.
- **Hero gallery committed**: assets/gallery/ — mission_c/b, fleet_nav,
  allocator MP4s + posters + hero_reel (1:56) + mission_c.gif (1.8 MB,
  inline README motion); stale single-robot GIFs removed.
- **README rewritten** (247 lines, recruiter-first) with honest-assumptions
  box; docs/multi_plan.md STATUS section appended.
- **Fresh-clone VR-1 rehearsal PASSED**: every documented command run
  verbatim in a clean clone — suite 1327 OK / mission_eval / mission_video /
  fleet_video / fleet_eval / nav_eval all exit 0 with artifacts. One stale
  README figure caught and corrected (cross-visibility 6.6%→3.3% after the
  art pass changed ego pixel-diff).

### Next 2 hours
- Cycle 2 kickoff: (a) real learned perception in the mission loop —
  per-robot GROUND_NET detector confirmation at close range alongside the
  geometric oracle (fixes the grounding singleton; strengthens the VLA
  story); (b) interactive live fleet demo (web UI: type addressed commands,
  watch fleet + transcript live) — design brief first, then Opus agents.

### Analysis
- The two-vote adversarial gate is well-calibrated for "ship now" questions
  but under-weights latent robustness for interactive use; reading the
  verifier journals (mechanism-confirmed vs impact-refuted) was the right
  orchestrator move — all 7 "rejected" mechanisms were real and cheap to fix.
- Release state is reproducible by construction: external assets are two
  documented symlinks; all headline numbers regenerate from committed CLIs.

### Performance
| Metric | Value | Notes |
|---|---|---|
| Test suite | 1327 OK (7 skip) | +22 robustness tests |
| Missions (10 seeds × A/B/C) | 30/30 | 0 falls |
| Allocator optimality | 3/3 | vs A* argmin ground truth |
| Fresh-clone rehearsal | 6/6 commands pass | VR-1 rule |
| Gallery footprint | ~7 MB committed | reel 2.8 MB + GIF 1.8 MB |
