"""warehouse_demo — Phase 1c single-robot A* navigation across the warehouse.

Submodules:
  * ``nav_rollout`` — ``run_nav_rollout`` (teacher-driven A* walk) + ``NavResult``.
  * ``bev`` — fixed wide-BEV camera framing + path-overlay projection math.
  * ``nav_eval`` — CLI: N random spawn->occluded-goal episodes + JSON/summary.
  * ``nav_video`` — CLI: record annotated BEV MP4s of a few episodes.
"""
