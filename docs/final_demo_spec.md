# Final Demo Spec (user requirements, 2026-07-12 ~21:10)

Five features for the FINAL demo videos, on top of the released gallery.

## F1 — Communicating robots' camera views, emphasized
Show robot ego views as insets; when two robots are exchanging messages
(e.g., Alpha↔Charlie QUERY/REPORT), BOTH their camera insets are shown with
an emphasized border (each robot's accent color, e.g. glow/thick outline)
for the duration of the exchange. Non-communicating robots' views are
hidden or de-emphasized.

## F2 — Target symbol only after it is known
No target ring/marker on the object at video start. The ring appears only
once a robot has actually located the object (REPORT_VISIBILITY yes /
REPORT_FOUND), drawn at the REPORTED position, from that moment onward.
Timeline: ask → find → THEN symbol.

## F3 — Fixed relative-position report format
Position reports always follow:
"I am robot {NAME}, currently in {ROOM} at position ({x}, {y}).
 The object is located {DX} m and {DY} m away from me."
Relative offsets (DX, DY) from the reporting robot, because each robot
knows its own pose exactly; receiver reconstructs absolute = reporter pose
+ offset. ROOM = named region (north/middle/south area, NE alcove, bays).
Applies to REPORT_VISIBILITY(visible=True) and REPORT_FOUND payloads +
transcript formatting + receiver-side computation.

## F4 — Fleet-addressed simple command demo
Demo where the user addresses the ENTIRE fleet with a minimal command,
e.g. "bring the object to the destination" (no color/shape, no robot
name): resolver accepts generic object references, allocator/fleet plans
optimally (search if hidden), mission completes, milestones reported.

## F5 — VLA locomotion (no direct WBC in demos)
Locomotion in final demos must come from the TRAINED VLA policy lineage of
/home/kiwoos/work/VLA_mujoco_unitree (distilled GroundedNav: ego RGB +
proprio + velocity conditioning → joint targets), not from driving the
Unitree whole-body controller directly. Plan:
1. Collect warehouse-domain DART data (teacher-driven rollouts along A*
   paths in the warehouse arena — the teacher's role is now confined to
   DATA COLLECTION, per the baseline recipe).
2. Fine-tune the distilled policy from the baseline checkpoint on that
   data (visual domain gap: gray floor/shelves vs playground).
3. Closed-loop validate (warehouse nav eval, policy-locomotion mode);
   gate: match teacher-mode success within a small margin, 0-fall bar.
4. Add locomotion backend switch (teacher | vla) to the nav engine;
   missions/videos run "vla" once validated; teacher mode stays for
   datagen and as a documented fallback.

## F6 — Multi-room warehouse (user, 2026-07-12 ~21:15)
Final demo takes place in a MULTI-ROOM warehouse: rooms connected by open
doorways, so objects are only findable by exploring room to room.
Design (orchestrator defaults, adjustable): `rooms_layout()` — 20×14 m
shell, FOUR named rooms: "loading room" (south strip, 4 robot bays),
"storage A" (west, shelf blocks), "storage B" (east, shelves + delivery
pad), "back room" (north). Open doorways (≥1.8 m) connect adjacent rooms;
no doors swing (always open). Rooms ARE the search regions AND the F3
room names (one source of truth: layout.rooms → search partition →
"currently in {room}" reports → COMMAND_SEARCH region strings).
Hero single-hall layout retained for regression/evals. Reachability
validated at inflation 0.40/0.45 for every bay→spot pair (alcove-sealing
lesson). Room-aware patrols route through doorways.

## Sequencing
- F5 datagen starts immediately (no file conflicts with Cycle-2 agents).
- F1-F4 batch after Cycle-2a/2b land (same files: mission_video, comms,
  actions). F3 changes code/comms payloads/formatting + tests.
- Final videos re-recorded once F1-F5 are all in.
