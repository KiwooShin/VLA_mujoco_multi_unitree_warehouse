# Demo Set v2 — Production Spec (user requirements, 2026-07-13 ~19:00)

User verdict on v1: not fancy enough — tasks too simple, single-room feel,
camera views missing in some videos, fixed start locations, ambiguous orders
("the object") executed without clarification.

## Hard requirements (all six demos)
1. Complicated tasks + complicated environments (multi-room, randomized,
   cluttered; different layout per demo where possible).
2. Robots start at RANDOM free locations (reachability-validated), not bays.
3. Every robot's camera view visible AT ALL TIMES (persistent ego strip;
   comm-emphasis glow overlays on top of the always-on views).
4. Communication text visible AT ALL TIMES (persistent transcript panel in
   every demo, including pure-navigation ones).
5. Ambiguous orders trigger CLARIFICATION: the robot/allocator asks the user
   which object is meant (from the warehouse object manifest — types/colors
   known, positions unknown), waits for the (scripted) user reply, then
   proceeds. New CLARIFY message flow, visible in the panel.
6. Output: repo `demo/` directory, one .mp4 AND one .gif per demo
   (mp4 <10 MB, gif <10 MB, both watchable quality), plus a poster PNG.
7. Audience: top-level AI-company recruiters/hiring managers — production
   polish: title card per demo (name + one-line description), clean HUD
   (mission phase, sim time), consistent typography, no debug artifacts.

## The six demos
| # | Name | Scenario | Environment |
|---|---|---|---|
| 1 | clarify_fetch | "Bring me the cube" with 3 cubes in the manifest → robot asks "which cube: red, blue, or yellow?" → user: "the red one" → delegated search + fetch | rooms, random spawns |
| 2 | unseen_map | Delegated exploration on a NEVER-SEEN sampled rooms layout, random spawns, occluded target | sample_rooms_layout(seed) |
| 3 | dual_fetch | "Bring the red cube and the blue ball to the pad" → TWO concurrent missions, two owners working in parallel, both delivered | rooms, random spawns |
| 4 | relay_multigoal | One robot fetches two objects sequentially (helpers re-used across legs) | rooms variant, random spawns |
| 5 | retask | Mid-mission order change: user redirects to a different object; owner aborts, re-clarifies if needed, completes the new task | rooms, random spawns |
| 6 | six_robot_flagship | Six robots, big hall, random spawns, ambiguous "bring me a ball" → clarify → search with reserve searchers → fetch | rooms6 |

## New capabilities (build order)
A. comms/fleet: CLARIFY dialogue (question to user + scripted user reply
   injection at runner level), random-spawn support (seeded free-pose
   sampling, reachability-checked), concurrent missions (2 owners),
   sequential multi-goal, mid-mission re-task. All behind new paths;
   existing tests/evals stay green.
B. video: new demo composer — 1600x900 canvas: BEV center-left, comms panel
   right, ALWAYS-ON ego strip bottom (one tile per robot, name + state,
   comm-glow borders), title card (2 s), HUD. Works for 4 and 6 robots.
C. production: 6 scenario scripts (deterministic seeds), record originals to
   ops/demo2/, frame-verify each beat, compress mp4+gif+poster into demo/,
   README gallery section update.
