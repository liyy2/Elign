# GEOM Experiment Leaderboard

This file tracks iterative GEOM RL experiments focused on **RDKit validity × RDKit uniqueness**.

## Summary

| Attempt | Date | Description | AtomStab | MolStab | Valid | Uniq | V×U | Duration | Status |
|---------|------|-------------|----------|---------|-------|------|-----|----------|--------|
| 001 | 2026-01-08 | Force-only (suffix-only, 2h budget) | 0.809 | 0.000 | 1.000 | 0.836 | 0.836 | 0.082h | completed |
| 002 | 2026-01-08 | Force-only (wrapper interrupted before checkpoints) | -- | -- | -- | -- | -- | 0.001h | early_stop |
| 003 | 2026-01-08 | Force-only (wrapper interrupted; no checkpoints) | 0.835 | 0.000 | 1.000 | 0.734 | 0.734 | 0.128h | early_stop |
| 004 | 2026-01-08 | Force-only (2h budget; interrupted) | 0.764 | 0.000 | 1.000 | 0.867 | 0.867 | 0.068h | early_stop |
| 005 | 2026-01-08 | Force-only (2h budget) | 0.820 | 0.000 | 0.992 | 0.748 | 0.742 | 0.093h | completed |
| 006 | 2026-01-08 | Force-only (6h budget; interrupted) | 0.705 | 0.000 | 0.852 | 0.899 | 0.766 | 0.181h | early_stop |
| 007 | 2026-01-08 | Energy+force (older weights; interrupted) | 0.804 | 0.000 | 0.992 | 0.606 | 0.602 | 0.040h | early_stop |
| 008 | 2026-01-08 | Energy+force (gentle energy; interrupted) | 0.726 | 0.000 | 0.992 | 0.890 | 0.883 | 0.142h | early_stop |
| 009 | 2026-01-08 | Energy+force resume (allocator crash) | 0.724 | 0.000 | 0.992 | 0.882 | 0.875 | 0.916h | failed |
| 010 | 2026-01-08 | Energy+force resume + dup0.10 + KL0.02 | -- | -- | -- | -- | -- | 1.500h | completed |

## Best Run So Far

Current best is **attempt 008** with V×U=0.883.

## Metrics Definitions

- **AtomStab**: atom stability rate (training rollouts).
- **MolStab**: molecule stability rate (training rollouts).
- **Valid**: RDKit validity fraction (training rollouts).
- **Uniq**: RDKit uniqueness among valid molecules (training rollouts).
- **V×U**: product of validity and uniqueness (primary objective).
- **Duration**: wall-clock time inferred from log timestamps + file mtime.
- **Status**: `completed`, `early_stop`, `failed`, `running`.

## Notes

- Compute budget per attempt: **≤6 hours** (unless explicitly shorter).
- These metrics come from the trainer's rollout batch (often small), so expect noise.
