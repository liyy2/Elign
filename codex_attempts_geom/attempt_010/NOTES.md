# Attempt 010: Energy+force resume + dup0.10 + KL0.02

**Date**: 2026-01-08  
**Status**: completed  
**Duration**: 1.500h  

## Run

- Log: `/home/yl2428/e3_diffusion_for_molecules-main/outputs/verl/geom_vxu/codex_geom_vxu_attempt010_energy_resume_dup10_kl002_20260108_165726.log`
- Run dir: `/home/yl2428/e3_diffusion_for_molecules-main/outputs/verl/geom_vxu/codex_geom_vxu_attempt010_energy_resume_dup10_kl002_20260108_165726`

## Best Metrics (checkpoint_best)

- Best epoch: None
- Atom stability: --
- Mol stability: --
- RDKit validity: --
- RDKit uniqueness: --
- Validity × Uniqueness: --

## Config

- Config: `ddpo_geom_energy_force_vxu_suffix`
- Resume from: `outputs/verl/geom_vxu/geom_vxu_energy2_20260108_122126/checkpoint_best.pth`
- Overrides: `['filters.duplicate_penalty_scale=0.10', 'train.kl_penalty_weight=0.02']`
