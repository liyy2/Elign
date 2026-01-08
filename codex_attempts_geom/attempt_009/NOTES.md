# Attempt 009: Energy+force resume (allocator crash)

**Date**: 2026-01-08  
**Status**: failed  
**Duration**: 0.916h  

## Run

- Log: `outputs/verl/geom_vxu/geom_vxu_energy2_20260108_122126_resume.log`
- Run dir: `outputs/verl/geom_vxu/geom_vxu_energy2_20260108_122126`

## Best Metrics (training rollout, before crash)

- Best epoch: 28
- Atom stability: 0.7239
- Mol stability: 0.0000
- RDKit validity: 0.9922
- RDKit uniqueness: 0.8819
- Validity × Uniqueness: 0.8750

## Failure

- Crash: PyTorch CUDA allocator internal assert (`CUDACachingAllocator.cpp:432`).

