"""fleet — Multi-robot warehouse co-simulation + collaborative missions.

Architecture (docs/multi_plan.md sec 3, federated physics + shared viz model):

  * ``robot_unit`` — :class:`~code.fleet.robot_unit.RobotUnit`: one named G1 with
    its OWN warehouse ``MjModel`` / ``MjData`` / ``WBCTeacher`` (the robot is
    alone in its model, so all baseline single-robot code is valid verbatim).
  * ``viz`` — :class:`~code.fleet.viz.FleetViz`: ONE shared kinematic model with
    the warehouse + all robots attached under name prefixes, never stepped —
    each frame every robot's physics qpos is copied into its prefixed slice and
    ``mj_forward`` refreshes kinematics. This is the model the fleet BEV video
    and the cross-visibility ego renders draw from.
  * ``fleet`` — :class:`~code.fleet.fleet.Fleet`: constructs the units + the viz
    model, steps them together, and applies the mutual-proximity pause (no
    inter-robot contact physics; the lower-priority robot yields).
  * ``fleet_eval`` / ``fleet_video`` — CLIs for the multi-robot eval + video.

Phase 4 (collaborative search + fetch-to-destination missions):

  * ``visibility`` — deterministic geometric visibility oracle (FOV + range +
    wall line-of-sight) standing in for the detector behind ``can_see``.
  * ``search`` — north/middle/south region partition + reachable patrol
    waypoints + :class:`~code.fleet.search.SearchController`.
  * ``carry`` — mock pickup: the object snaps to the carrier's right wrist and
    is kinematically re-posed to the hand every step until released on the pad.
  * ``actions`` — :class:`~code.fleet.actions.FleetRobotActions`, the thin
    bridge implementing :class:`code.comms.protocol.RobotActions`.
  * ``allocator`` — path-length-optimal (A*) robot choice for fleet-addressed
    requests.
  * ``mission`` — :class:`~code.fleet.mission.MissionRunner`: fleet + bus +
    protocols + carry in one closed loop, from text order to TASK_COMPLETE.
  * ``mission_video`` / ``mission_eval`` — flagship demo video + seeded evals.
"""
