# Attempt 002: Force-only (wrapper interrupted before checkpoints)

**Date**: 2026-01-08  
**Status**: early_stop  
**Duration**: 0.001h  

## Run

- Log: `outputs/verl/geom_vxu/geom_vxu_20260108_113710.log`
- Intended run dir: `outputs/verl/geom_vxu/geom_vxu_20260108_113710_phase1_force` (empty)

## Notes

- Wrapper (`run_geom_vxu_6h.py`) was interrupted while sleeping; training started but exited before emitting epoch metrics or writing checkpoints/configs.

