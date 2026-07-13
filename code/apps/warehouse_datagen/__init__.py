"""warehouse_datagen — F5 phase-1: warehouse-domain DART data collection.

Generates teacher-driven DART rollouts along A*-planned warehouse routes (plus a
matched share of baseline-style direct-steer primitive segments), recording the
EXACT baseline DART parquet schema (loadable by the baseline
``PhaseParquetDataset``) plus GPU-rendered ego RGB mp4s for visual-domain
fine-tuning of the distilled GroundedNav walk policy.

Modules:
  egl_gpu             — force MuJoCo EGL offscreen rendering onto the NVIDIA GPU.
  scene               — seeded warehouse episode-plan sampler (pure logic).
  rollout             — single warehouse DART episode (teacher + ego rendering).
  gen_warehouse_dart  — CLI, orchestration, meta assembly, idempotent chunking.
  sanity              — schema conformance + contact sheet + distribution stats.
"""
