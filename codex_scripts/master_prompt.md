# Experiment Configuration

## Task
**Type:** RL Post-Training (PPO/DDPO)  
**Dataset:** GEOM-drug 
**Base Model:** E(3) Equivariant Diffusion Model
Always run conda activate edm when you run.
## Run-Specific Objective
<!-- Edit this section for each experiment -->
help me improve the experiment using dl-monitor skill on GEOM dataset. Implement some changes based on my prior experiment
Optimize molecular validity and stability using MLFF energy/force rewards.
Target: Improve validity×uniqueness while maintaining atom/molecule stability. keep iterating until siginificant progress. a good rule of thumb is that the validity should reach 98% for 1024 tests.
Some ideas: You can start a train with only terminal force i think this will give you a much better stability.
W&B: set `WANDB_API_KEY` (40+ chars) in your environment or run `wandb login` (do not paste keys into this file).

## Key Metrics
- validity
- uniqueness  
- atom_stability
- molecule_stability
- validity × uniqueness
- reward/mean

For small run could evaluate with little samples, but when you made progress, always use 1024 samples to report the results. (previous results might not be evaluated on 1024 samples, maintained a new leaderboard for the 1024 samples in @New_leaderboard_1024.md). Try to train longer since RL is quite noisy. 
It would also be nice to visualize use wandb. When you are confident about a run, and you should train longer (>4hr). You should use slurm to launch as many jobs as you can. Also take good advantage of the local resource (i.e. for testing new ideas) (it's an h100 node with a single card)

## Alert Levels
- none: healthy - Averaged Reward across runs (EMA smoothed potentially) should be increasing
- info: checkpoint saved, minor notes
- warning: plateau starting, high KL
- critical: NaN, OOM, crash

## Recommendations
Provide actionable suggestions for next steps or next run:
- If plateauing: suggest lr adjustment, different reward weights
- If unstable: suggest lower lr, higher KL penalty
- If healthy: suggest continuing or what to monitor
- If crashed: suggest fix or parameter change for retry


## INJECT PROMPTs
