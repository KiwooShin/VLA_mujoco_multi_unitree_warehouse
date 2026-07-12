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
