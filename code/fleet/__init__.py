"""fleet — Phase 2 multi-robot warehouse co-simulation.

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
"""
