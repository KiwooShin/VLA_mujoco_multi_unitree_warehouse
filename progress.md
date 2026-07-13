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

---

## 2026-07-12 22:30 — Final-demo features F1-F4/F6 + learned stack (entry 5)

User specified final-demo requirements F1-F6 (docs/final_demo_spec.md) at
~21:10; five of six are DONE and pushed; F5 (VLA locomotion) is training.

### Done (last ~1.7 h)
- **Cycle 2b (web demo, ad0e206)**: code/apps/fleet_web — live browser
  dashboard (MJPEG BEV, status chips, live transcript, command box with
  friendly validation, sequential queue). Live rehearsal: addressed fetch
  27 s; queued fleet order allocated to path-shortest robot. 42→43 tests.
- **F6 (rooms layout, 8fa1782)**: 20×14 m, four named rooms (loading/
  storage A/storage B/back room), four 2.0 m doorways in a cycle, solid
  A|B divider; Room/room_of API = single source of truth for search
  regions AND spoken room names. 44/44 reachability at 0.40/0.45. 22 tests.
- **F5-p1 (datagen, 821b06f)**: 200-ep/125k-frame warehouse DART dataset,
  0.985 success, action distribution matches original; validated via both
  baseline loaders + real-trainer overfit gate. Found+fixed conda EGL
  vendor shadowing (llvmpipe 160 ms/frame → 0.88 ms GPU, 190×).
- **Cycle 2a (GROUND_NET, 72eeb54)**: real detector in the mission loop —
  per-robot GroundNetState (singleton cross-contamination regression-
  tested), shared weights, geometry-consistency confirm gate, heatmap
  insets on video. groundnet missions 12/12, 0 falls, 1567 confirmations.
  HONEST finding: warehouse domain shift collapses detector confidence
  (geometry stays 0.03-0.13 m accurate) → warehouse det fine-tune queued.
- **F1-F4 batch (4d7b43b)**: comm-emphasized ego insets; deferred target
  ring at REPORTED position; exact relative-position sentence ("I am robot
  Bravo, currently in storage A at position (-4.0, 2.8). The object is
  located -5.0 m and 3.2 m away from me."); generic fleet commands
  ("someone bring the object"); room-aware search through doorways
  (rooms eval C 9/9, D 3/3, 0 falls); MissionRunner lifecycle API
  (fleet_web now reuses one runner). Frame-verified rooms scenario-C and
  generic-command draft videos in ops/f14/.
- **F5-p2 (in flight)**: fine-tune from deployed baseline ckpt on the
  warehouse DART set — training on GPU (50-65% util), epoch 8+/20,
  checkpoints saving; closed-loop selection + teacher|vla backend next.

### Next 2 hours
- VLA training completes → closed-loop checkpoint selection → vla backend
  gate (hero + rooms, ≥90%, 0 falls) → commit.
- GROUND_NET warehouse fine-tune (det dataset gen + train) once the GPU
  frees; re-run perception_eval (expect confidence recovery).
- Then final demo videos (rooms + VLA locomotion + all F-features).

### Performance
| Metric | Value | Notes |
|---|---|---|
| Rooms missions (search class C) | 9/9 | room-to-room exploration |
| Rooms allocator (D) | 3/3 | A* argmin verified |
| Hero regression after F1-F4 | 6/6 + 3/3 | unchanged |
| GROUND_NET in-loop | 12/12, 1567 confirms | geometry gate, 0 through-wall FP in-mission |
| Warehouse DART dataset | 200 eps / 125,683 fr | 98.5% ep success |
| Datagen render | 0.88 ms/frame GPU | was 160 ms on llvmpipe |
| Suites | comms 52, fleet 120, fleet_web 43, warehouse 86 | + warehouse_demo 39 |

---

## 2026-07-13 01:45 — Full learned stack + final demo set (entry 6)

### Done (last ~3.2 h)
- **GROUND_NET warehouse fine-tune (8c65b91)**: new 11,630-frame warehouse
  detector dataset (NX-6-recipe-compatible, hero+rooms scenes, wall-occluded
  negatives from segmentation); retrained detector. Warehouse detection
  rate 0.070 → **1.000**, xy p90 0.039 m, confidence p50 0.243 → 0.894.
  The raw "through-wall FP" metric was decomposed frame-by-frame with
  segmentation renders: 0 true hallucinations (the point-LOS oracle
  mislabels partially visible objects as hidden). Ckpt auto-resolution:
  GROUND_NET_CKPT env → warehouse_ft → original.
- **F5 VLA locomotion (3bb34cf)**: 20-epoch fine-tune from the deployed
  baseline policy on the warehouse DART set (val_action 0.0886→0.0612).
  Closed-loop selection across 5 checkpoints (all 10/10; ep19 chosen).
  Gate PASSED round 1: hero 20/20 + rooms 10/10, 0 falls, eff ≥0.994,
  1.6 ms/step GPU inference. Deploy path replicates the baseline
  velocity-injection recipe bit-for-bit in distribution.
- **Final integration (07d96c9)**: locomotion=teacher|vla threaded through
  the fleet (one shared policy, per-unit proprio windows). Interaction
  found+fixed: VLA cannot balance a post-walk stand → held steps run WBC
  balance, walking stays VLA (same precedent as the settle phase).
  **Full-stack headline (rooms + fine-tuned detector + VLA): C 12/12,
  D 3/3, 0 falls across 24 missions.** Four frame-verified FINAL videos
  (flagship exploration, generic fleet command, 4-robot cross-room nav,
  peer-sighting) compressed into assets/gallery/final_*.mp4 + rebuilt
  hero_reel + mission_c.gif.
- Known edge (documented, not scored): one unscored D-mission delivery
  failed on a 1.7 m detector localization outlier — owner walked to the
  reported spot, out of pickup range. Future work: close-range detector
  re-confirmation updating the goal.

### Next
- Final release agent (running): README refresh around the learned stack,
  gallery reorg, docs status, fresh-clone re-rehearsal incl. trained-ckpt
  absence fallback story. Then commit + push; remaining session time goes
  to polish/backlog (interactive rooms web demo default, oracle
  partial-visibility upgrade, randomized-layout generalization eval).

### Performance (headline, full learned stack)
| Metric | Value | Notes |
|---|---|---|
| Rooms missions (C, delegated exploration) | 12/12 | groundnet + VLA, seeds 4 |
| Allocator (D) | 3/3 | A* argmin verified |
| Falls (all 24 full-stack missions) | 0 | |
| Detector (warehouse) | det 1.000, conf p50 0.894 | was 0.070 / 0.243 |
| VLA nav gate | 30/30, 0 falls | hero 2 seeds + rooms |
| Policy inference | 1.6-1.7 ms/step | GPU |
| Test suite | 1450 OK (7 skip) | |

---

## 2026-07-13 04:20 — Release refresh + hardening + generalization (entry 7)

### Done (last ~2.5 h)
- **Final release refresh (c4b6b80)**: README rewritten around the learned
  stack (416 lines) — final videos first, real transcript with the exact
  relative-position sentence, updated honest-assumptions box, full-stack
  results, NEW training section with reproducible commands, ckpt
  auto-resolution + fresh-clone fallback documented. Fresh-clone VR-1
  re-rehearsal: every documented command exit 0 BOTH without trained ckpts
  (teacher/oracle fallback) and with them (full stack).
- **Perception precision (347a0d3)**: (1) D-14 fixed — approach
  confirmations refine the in-flight goal + pickup retry reconfirms from
  close range; the exact previously-failing mission now completes
  (same seed). Rooms full stack now 12/12 + 3/3 with ZERO residual
  failures. (2) Visibility oracle samples object extent (edge-visible
  counts); the 17 "through-wall FPs" root-caused as physically-invalid
  camera poses (inside shelves) — valid-pose gate added; genuine
  occlusion FP rate: 0.000. Determinism byte-identical. +19 tests.
- **Generalization eval (in flight)**: sample_rooms_layout() +
  reachability-gated sample_layout (found+fixed: ~30% of ungated hero
  draws sealed the alcove); gen_eval CLI sweeping 8+8 randomized layouts
  × 3 seeds × both stacks (~200 missions). Rooms family COMPLETE (rc=0);
  hero family running. +20 tests; suite 1489 OK.

### Next
- Generalization tables → README summary table + commit.
- Remaining backlog if time: space-time reservations, 6-8 robots,
  rooms-default web demo.

### Performance
| Metric | Value | Notes |
|---|---|---|
| Full stack after fixes | 12/12 C + 3/3 D, 0 fails | rooms, seeds 4 |
| Recovered mission | D seed=102: failed → complete | same-seed proof |
| Occlusion FP (valid poses) | 0.000 | was 0.262 raw / mislabeled |
| Test suite | 1489 OK (7 skip) | +39 since release |
