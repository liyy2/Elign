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
| 010 | 2026-01-08 | Baseline force-only (suffix-only) on GEOM, small pilot batch | -- | -- | -- | -- | -- | 0.008h | failed |
| 011 | 2026-01-08 | Baseline force-only (suffix-only) on GEOM, small pilot batch | 0.769 | 0.000 | 1.000 | 0.984 | 0.984 | 0.075h | completed |
| 012 | 2026-01-08 | Resume + atom stability shaping (w=1.0) | -- | -- | -- | -- | -- | 0.008h | completed |
| 013 | 2026-01-08 | Resume + atom stability (w=1.0) + valence under/over (w=0.02) | -- | -- | -- | -- | -- | 0.008h | completed |
| 014 | 2026-01-08 | Resume(attempt_011 best) + atom stability shaping (w=1.0) | 0.737 | 0.000 | 0.891 | 0.807 | 0.719 | 0.109h | completed |
| 015 | 2026-01-08 | Resume(prev) + atom stability (w=1.0) + valence under/over (w=0.02) | 0.670 | 0.000 | 0.984 | 0.984 | 0.969 | 0.109h | completed |
| 016 | 2026-01-08 | Resume(prev) + stability/valence + focus smaller node counts (20-80) | 0.740 | 0.000 | 1.000 | 1.000 | 1.000 | 0.109h | completed |
| 017 | 2026-01-08 | Resume(attempt_016 epoch50) + optimize V×U×MolStab with stronger stability shaping | 0.720 | 0.000 | 1.000 | 0.969 | 0.969 | 0.109h | completed |
| 018 | 2026-01-08 | Resume(attempt_016 epoch50) + batch128 + stability_weight + combined checkpoint metric | 0.674 | 0.000 | 0.984 | 0.810 | 0.797 | 0.117h | completed |
| 019 | 2026-01-08 | Resume(attempt_016 epoch50) + skip_prefix=800 + combined checkpoint metric | 0.705 | 0.016 | 0.953 | 1.000 | 0.953 | 0.109h | completed |
| 020 | 2026-01-08 | Resume(attempt_019 best) + skip_prefix=800 + checkpoint metric=V×U×MolStab (6h, no early-stop) | 0.986 | 0.641 | 1.000 | 0.781 | 0.781 | 6.168h | completed |
| 021 | 2026-01-09 | Resume(attempt_020 best) + KL(0.02) + mol_stability_w(0.5) + dup_penalty(0.10) + lr=4e-6 | 0.955 | 0.219 | 0.953 | 0.754 | 0.719 | 0.167h | failed |
| 022 | 2026-01-09 | Resume(attempt_020 best) + KL(0.02, ref=resume) + dup_penalty(0.10) + energy_adv=0 | 0.970 | 0.312 | 0.969 | 0.742 | 0.719 | 0.500h | failed |
| 023 | 2026-01-09 | Resume(attempt_020 best) + energy+force(stable-gated) + KL(0.01, ref=resume) + dup_penalty(0.10) + lr=4e-6 | 0.971 | 0.500 | 1.000 | 0.891 | 0.891 | 0.448h | early_stop |
| 024 | 2026-01-09 | Resume(attempt_020 best) + energy(stable-gated) + force(ungated) + KL(0.02) + lr=3e-6 + dup_penalty(0.10) | -- | -- | -- | -- | -- | 0.037h | early_stop |
| 025 | 2026-01-09 | Resume(attempt_020 best) + energy(stable-gated) + force(ungated) + KL(0.02) + lr=3e-6 + dup_penalty(0.10) | 0.974 | 0.453 | 0.953 | 0.836 | 0.797 | 0.202h | early_stop |
| 026 | 2026-01-09 | Resume(attempt_020 best) + energy(stable-gated) + force(ungated) + stab_w(1.0) + KL(0.02) + lr=3e-6 + dup_penalty(0.10) | 0.981 | 0.594 | 1.000 | 0.875 | 0.875 | 0.342h | early_stop |
| 027 | 2026-01-09 | Resume(attempt_026 best) + stab_w(1.0) + KL(0.04) + lr=2e-6 + dup_penalty(0.10) | 0.990 | 0.703 | 1.000 | 0.781 | 0.781 | 1.859h | early_stop |
| 028 | 2026-01-09 | Resume(attempt_027 best) + dup_penalty(0.20) + stab_w(1.0) + KL(0.04) + lr=2e-6 | 0.976 | 0.562 | 0.969 | 0.887 | 0.859 | 0.072h | early_stop |
| 029 | 2026-01-09 | Resume(attempt_027 best) + dup_penalty(0.20) + stab_w(1.0) + KL(0.04) + lr=2e-6 | 0.987 | 0.734 | 1.000 | 0.469 | 0.469 | 0.251h | early_stop |
| 030 | 2026-01-09 | Resume(prev_best) + dup_penalty(0.30) + stab_w(1.0) + KL(0.04) + lr=2e-6 | 0.978 | 0.594 | 1.000 | 0.688 | 0.688 | 0.211h | completed |
| 031 | 2026-01-09 | Resume(prev_best) + dup_penalty(0.30) + stab_w(1.0) + KL(0.04) + lr=3e-6 | 0.977 | 0.562 | 1.000 | 0.750 | 0.750 | 0.119h | completed |
| 032 | 2026-01-09 | Resume(attempt_027 best) + MLFF(real) + no_force_gate + clip10 + dup(0.20) + stab_w(2.0) + KL(0.04) + lr=2e-6 | -- | -- | -- | -- | -- | 0.003h | failed |
| 033 | 2026-01-09 | Resume(attempt_027 best) + MLFF(real) + no_force_gate + clip10 + dup(0.20) + stab_w(2.0) + KL(0.04) + lr=2e-6 | -- | -- | -- | -- | -- | 0.003h | failed |
| 034 | 2026-01-09 | Resume(attempt_027 best) + MLFF(real) + no_force_gate + clip10 + dup(0.20) + stab_w(2.0) + KL(0.04) + lr=2e-6 | 0.990 | 0.750 | 0.969 | 0.855 | 0.828 | 0.946h | completed |
| 035 | 2026-01-09 | Resume(prev_best) + no_force_gate + clip10 + dup(0.20) + stab_w(3.0) + KL(0.04) + lr=2e-6 | 0.970 | 0.510 | 0.906 | 0.776 | 0.703 | 0.251h | early_stop |
| 036 | 2026-01-09 | Resume(prev_best) + no_force_gate + clip10 + dup(0.20) + stab_w(3.0) + KL(0.06) + lr=1e-6 | 0.978 | 0.529 | 0.969 | 0.790 | 0.766 | 0.103h | completed |
| 037 | 2026-01-09 | Resume(attempt_034 best) + fixed_KL + clip10 + dup(0.30) + stab_w(2.5) + KL(0.04) + lr=2e-6 | 0.996 | 0.860 | 0.984 | 0.778 | 0.766 | 0.739h | completed |
| 038 | 2026-01-09 | Resume(attempt_037 best) + lr=1e-6 + dup(0.35) + stab_w(4.0) + atom_w(10) + valence_w(0.10) + fixed_KL(0.04) | 0.985 | 0.600 | 1.000 | 0.703 | 0.703 | 0.117h | completed |
| 039 | 2026-01-09 | Resume(attempt_037 best) + lr=1e-6 + dup(0.35) + stab_w(4.0) + atom_w(10) + valence_w(0.10) + fixed_KL(0.04) | 0.981 | 0.578 | 0.953 | 0.869 | 0.828 | 0.306h | completed |
| 040 | 2026-01-10 | Resume(attempt_039 best) + lr=1e-6 + dup(0.45) + stab_w(4.0) + atom_w(10) + valence_w(0.10) + fixed_KL(0.04) | 0.995 | 0.844 | 0.984 | 0.778 | 0.766 | 1.624h | completed |
| 041 | 2026-01-10 | Resume(attempt_040 epoch1049) + each_prompt_sample=32 + grad_accum=8 + lr=1e-6 + dup(0.45) + stab_w(4.0) + atom_w(10) + valence_w(0.10) + fixed_KL(0.04) | 0.985 | 0.625 | 0.914 | 0.872 | 0.797 | 0.443h | completed |
| 042 | 2026-01-10 | Resume(attempt_040 epoch1049) + each_prompt_sample=16 + no_filtering + force_only_if_stable + dup(0.55) + stab_w(4.0) + atom_w(10) + valence_w(0.10) + fixed_KL(0.04) | -- | -- | -- | -- | -- | 0.003h | failed |
| 043 | 2026-01-10 | Resume(attempt_040 epoch1049) + each_prompt_sample=16 + no_filtering + force_only_if_stable + dup(0.55) + stab_w(4.0) + atom_w(10) + valence_w(0.10) + fixed_KL(0.04) | 0.983 | 0.656 | 0.922 | 0.898 | 0.828 | 0.440h | completed |
| 044 | 2026-01-10 | Resume(attempt_040 epoch1049) + lr=1e-6 + dup(0.55) + stab_w(4.0) + atom_w(10) + valence_w(0.10) + fixed_KL(0.04) | 0.991 | 0.750 | 0.984 | 0.810 | 0.797 | 0.752h | early_stop |
| 045 | 2026-01-10 | Resume(attempt_044 epoch1089) + lr=1e-6 + dup(0.60) + invalid(0.45) + stab_w(4.0) + atom_w(10) + valence_w(0.10) + fixed_KL(0.04) | 0.986 | 0.641 | 0.984 | 0.905 | 0.891 | 0.365h | completed |
| 046 | 2026-01-10 | Resume(attempt_045 epoch1092) + lr=1e-6 + dup(0.60) + invalid(0.50) + soft_valence(0.02) + stab_w(4.0) + atom_w(10) + valence_w(0.10) + fixed_KL(0.04) | -- | -- | -- | -- | -- | 0.003h | failed |
| 047 | 2026-01-10 | Resume(attempt_045 epoch1092) + lr=1e-6 + dup(0.60) + invalid(0.50) + soft_valence(0.02) + stab_w(4.0) + atom_w(10) + valence_w(0.10) + fixed_KL(0.04) | 0.991 | 0.766 | 0.953 | 0.738 | 0.703 | 0.054h | completed |
| 048 | 2026-01-10 | Resume(attempt_045 epoch1092) + lr=1e-6 + dup(0.60) + invalid(0.50) + soft_valence(0.02) + stab_w(4.0) + atom_w(10) + valence_w(0.10) + fixed_KL(0.04) | 0.992 | 0.781 | 0.969 | 0.823 | 0.797 | 0.070h | completed |
| 049 | 2026-01-10 | Resume(attempt_045 epoch1092) + lr=1e-6 + dup(0.60) + invalid(0.45) + soft_valence(0.02) + stab_w(4.0) + atom_w(10) + valence_w(0.10) + fixed_KL(0.04) | 0.991 | 0.766 | 0.953 | 0.738 | 0.703 | 0.103h | completed |
| 050 | 2026-01-10 | Resume(attempt_045 epoch1092) + lr=1e-6 + dup(0.65) + invalid(0.45) + soft_valence(0.02) + stab_w(4.0) + atom_w(10) + valence_w(0.10) + fixed_KL(0.04) | 0.992 | 0.781 | 0.938 | 0.767 | 0.719 | 0.293h | failed |
| 051 | 2026-01-10 | Resume(attempt_045 epoch1092) + skip700 + lr=1e-6 + dup(0.60) + invalid(0.45) + soft_valence(0.02) + stab_w(4.0) + atom_w(10) + valence_w(0.10) + fixed_KL(0.04) | 0.994 | 0.828 | 1.000 | 1.000 | 1.000 | 1.168h | early_stop |
| 052 | 2026-01-10 | Resume(attempt_051 checkpoint_best epoch1146) + lr=5e-7 + clip=0.15 + KL=0.06 + force_only_if_stable + stab_w=5.0 + val_over=0.15/0.03 | -- | -- | -- | -- | -- | 0.001h | failed |
| 053 | 2026-01-10 | Resume(attempt_051 checkpoint_best epoch1146) + lr=5e-7 + clip=0.15 + KL=0.06 + force_only_if_stable + stab_w=5.0 + val_over=0.15/0.03 | 0.994 | 0.828 | 1.000 | 1.000 | 1.000 | 1.571h | completed |
| 054 | 2026-01-10 | Resume(attempt_051 checkpoint_best epoch1146) + lr=3e-7 + clip=0.15 + KL=0.06 + force_only_if_stable + stab_w=6.0 + val_over=0.2/0.05 | 0.994 | 0.828 | 1.000 | 1.000 | 1.000 | 0.668h | early_stop |
| 055 | 2026-01-10 | Resume(attempt_051 checkpoint_best epoch1146) + no_force_gate + lr=3e-7 + clip=0.15 + KL=0.06 + stab_w=6.0 + val_over=0.2/0.05 | 0.994 | 0.828 | 1.000 | 1.000 | 1.000 | 0.418h | early_stop |
| 056 | 2026-01-10 | Resume(attempt_051 checkpoint_best epoch1146) + lr=3e-7 + stabguard + underbond_w=0.15 soft=0.05 | 0.994 | 0.828 | 1.000 | 1.000 | 1.000 | 0.251h | early_stop |
| 057 | 2026-01-10 | Resume(attempt_051 checkpoint_best epoch1146) + each32 + shaping(adaptive 50/5) + force_align_w=0.05 | 0.994 | 0.828 | 1.000 | 1.000 | 1.000 | 0.367h | failed |
| 058 | 2026-01-10 | Resume(attempt_051 checkpoint_best epoch1146) + each16 + shaping(energy-only adaptive 50/5) + force_align_w=0.02 | 0.994 | 0.828 | 1.000 | 1.000 | 1.000 | 0.418h | early_stop |
| 059 | 2026-01-10 | Resume(attempt_051 checkpoint_best epoch1146) + each24 + shaping(energy-only adaptive 50/5) + force_align_w=0.02 + energy_adv_w=0.05 | 0.994 | 0.828 | 1.000 | 1.000 | 1.000 | 0.439h | completed |
| 060 | 2026-01-10 | Resume(attempt_051 checkpoint_best epoch1146) + each24 + shaping(energy-only adaptive 50/5) + force_align_w=0.02 + energy_adv_w=0.02 + val_over_w=0.20 + dup=0.65 | 0.994 | 0.828 | 1.000 | 1.000 | 1.000 | 0.501h | early_stop |
| 061 | 2026-01-10 | Resume(attempt_060 checkpoint_best epoch1147) + stab_w=6 + soft_valence(0.05) + underbond_w=0.15 + no_force_gate + energy_w=0.10 + energy_adv_w=0.05 + dup=0.60 + lr=3e-7 | 0.994 | 0.828 | 1.000 | 1.000 | 1.000 | 0.235h | completed |
| 062 | 2026-01-10 | Resume(attempt_060 checkpoint_best epoch1147) + force_gate + val_over_w=0.30 + val_over_soft=0.08 + stab_w=6 + dup=0.50 | 0.994 | 0.828 | 1.000 | 1.000 | 1.000 | 0.402h | early_stop |
| 063 | 2026-01-10 | Resume(attempt_062 checkpoint_best) + force_gate + overbond_w=0.40 + overbond_soft=0.12 + dup=0.45 + lr=3e-7 | 0.994 | 0.828 | 1.000 | 1.000 | 1.000 | 0.368h | early_stop |
| 064 | 2026-01-10 | Resume(prev_best) + force_gate + overbond_w=0.50 + overbond_soft=0.15 + dup=0.40 + lr=3e-7 | 0.994 | 0.828 | 1.000 | 1.000 | 1.000 | 0.468h | early_stop |
| 065 | 2026-01-10 | Resume(attempt_064 best) + stronger stability/valence + smaller PPO step (lr=2e-7, kl=0.10, clip=0.10) | 0.994 | 0.828 | 1.000 | 1.000 | 1.000 | 0.085h | early_stop |
| 066 | 2026-01-10 | Resume(attempt_064 best) + stronger stability/valence + smaller PPO step + save_interval=1 | 0.994 | 0.828 | 1.000 | 1.000 | 1.000 | 0.135h | early_stop |
| 067 | 2026-01-10 | Resume(attempt_064 best) + terminal-only rewards + smoothed stop checks + tighter PPO step | 0.994 | 0.828 | 1.000 | 1.000 | 1.000 | 0.502h | early_stop |
| 068 | 2026-01-10 | Resume(attempt_067 best) + force-gated stability + stronger overbond penalties + tighter PPO | -- | -- | -- | -- | -- | 0.003h | failed |
| 069 | 2026-01-10 | Resume(attempt_067 best) + force-gated stability + stronger overbond penalties + tighter PPO | 0.994 | 0.828 | 1.000 | 1.000 | 1.000 | 0.185h | early_stop |
| 070 | 2026-01-10 | Resume(attempt_067 best) + stronger stability/valence + conservative PPO step + save_interval=1 | 0.994 | 0.828 | 1.000 | 1.000 | 1.000 | 0.202h | early_stop |

## Best Run So Far

Current best is **attempt 016** with V×U=1.000.

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
