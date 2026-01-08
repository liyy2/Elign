# Attempt 003: Force-only (wrapper interrupted; no checkpoints)

**Date**: 2026-01-08  
**Status**: early_stop  
**Duration**: 0.128h  

## Run

- Log: `outputs/verl/geom_vxu/geom_vxu_20260108_113759.log`
- Intended run dir: `outputs/verl/geom_vxu/geom_vxu_20260108_113759_phase1_force` (empty)

## Best Metrics (training rollout)

- Best epoch: 1
- Atom stability: 0.8345
- Mol stability: 0.0000
- RDKit validity: 1.0000
- RDKit uniqueness: 0.7344
- Validity × Uniqueness: 0.7344

## Notes

- Wrapper (`run_geom_vxu_6h.py`) was interrupted while sleeping; some metrics printed but no checkpoints/configs were written to the requested `save_path`.

