# Ala2 Scratch Module

This module is a standalone implementation path for finite-temperature alanine dipeptide experiments.

It is intentionally separate from the existing QM9/GEOM diffusion and RL stack.

## Scope

- Fixed-topology alanine dipeptide only
- Fixed atom count and atom identities
- Coordinate-only diffusion
- Terminal MLFF energy reward only
- W&B logging in the first pass
- Slurm-first execution

## Milestones

### 1. Data

Successful data loading and processing means:

- source trajectory files are parsed
- topology metadata is inferred
- processed arrays are written
- train/val/test splits are written
- a smoke-load path can build at least one batch with finite coordinates

### 2. Pretrain

Successful pretraining means:

- the job runs stably under Slurm
- losses remain finite
- checkpoints are written
- W&B receives training and evaluation metrics
- the base checkpoint can be evaluated on held-out Ala2 data

The base model does not need to be strong. It only needs to be healthy enough for post-training.

### 3. Post-Train

Successful post-training means:

- the pretrained checkpoint loads
- terminal MLFF energy reward is finite
- the optimization loop remains stable
- W&B logs reward, energy, and evaluation metrics
- at least one post-trained checkpoint can be compared against the base checkpoint

### 4. Scientific Win

The target outcome is that a post-trained checkpoint improves over the base checkpoint on at least one held-out equilibrium metric, preferably `phi_psi_jsd`.

## Main Entry Points

- `ala2_scratch/prepare_data.py`
- `ala2_scratch/pretrain.py`
- `ala2_scratch/posttrain.py`
- `ala2_scratch/eval.py`

## Slurm

- `ala2_scratch/slurm/pretrain_ala2.sbatch`
- `ala2_scratch/slurm/posttrain_ala2.sbatch`

## Notes

- This module should reuse the EGNN backbone only.
- It should not depend on the existing DDPO trainer or RDKit-based evaluation path.
