# ELIGN post-training

ELIGN post-trains a pretrained **equivariant diffusion model** by treating the reverse diffusion process as a **policy** and optimizing it with reinforcement learning.

At a high level:

1) Load a pretrained diffusion backbone (EDM-style; see `edm_source/`).
2) Sample rollouts from the diffusion policy (optionally with grouped/shared noise).
3) Score sampled molecules using physics-inspired reward signals (ML force fields).
4) Update the diffusion policy with **FED-GRPO**.

This document describes the runtime pipeline and how it maps to the code.

## Key scripts

- Training entrypoint: `run_elign.py`
- Evaluation sampling: `eval_elign_rollout.py`
- Evaluation metrics: `compute_elign_metrics.py`

## Key modules

**Dataloader (prompt grouping)**

- `elign/dataloader/dataloader.py` (`EDMDataLoader`)

Produces `DataProto` batches that include:

- `group_index`: which rollout samples belong to the same prompt group
- `nodesxsample`: node count per sample
- optionally `context`: conditioning vector (for conditional models)

**Rollout (diffusion sampling)**

- `elign/worker/rollout/edm_rollout.py` (`EDMRollout`)

Runs the diffusion sampler to produce:

- `latents`, `next_latents`, `timesteps`: diffusion trajectories
- behavior-policy quantities needed for policy gradient (e.g. `log_prob_old`, `mu_old`)

**Reward (forces / energies / stability)**

- `elign/worker/reward/force.py` (`UMAForceReward`)
- `elign/worker/reward/dummy.py` (`DummyReward`, for smoke tests)
- `elign/worker/reward/scheduler.py` (`RewardScheduler`, for shaping schedules)

`UMAForceReward` computes:

- a force-based term (RMS or max aggregation over per-atom forces)
- an optional energy term (with configurable transform/clipping and optional stability gating)
- optional stability / valence penalties

It can optionally emit per-timestep reward traces (`*_rewards_ts`) when shaping is enabled.

**Filtering / penalties**

- `elign/worker/filter/filter.py` (`Filter`)

Adds training-time penalties (novelty, invalidity, duplicates) and logs RDKit validity/uniqueness.
Penalties are injected into all reward channels so they affect learning even when using split force/energy advantages.

**Trainer / optimizer loop**

- `elign/trainer/fed_grpo_trainer.py` (`FedGrpoTrainer`)
- `elign/worker/actor/edm_actor.py` (`EDMActor`)

The trainer orchestrates:

1) rollout generation
2) reward computation
3) filtering / penalty injection
4) advantage computation (FED-GRPO)
5) policy update steps (PPO-style clipped objective + optional KL penalty)
6) checkpointing and logging

## Running ELIGN

ELIGN training uses Hydra configs under `elign/trainer/config/`.

### Minimal smoke test (no MLFF)

```bash
python run_elign.py \
  --config-name fed_grpo_config \
  reward.type=dummy \
  wandb.enabled=false \
  dataloader.epoches=2 \
  dataloader.sample_group_size=2 \
  dataloader.each_prompt_sample=4 \
  model.time_step=50
```

### Typical run directory contents

ELIGN writes training artifacts under `save_path` (recommended: `outputs/elign/...`):

- `config.yaml`: the resolved Hydra config used for the run
- `checkpoint_latest.pth`: latest training checkpoint (model + optimizer state)
- `checkpoint_best.pth`: best checkpoint according to `best_checkpoint_metric`
- `generative_model_ema.npy`: model weights snapshot saved when a new best checkpoint is reached

### Distributed training

`run_elign.py` supports `torchrun` / `torch.distributed` (DDP) via environment variables.

Example (single node, 1 GPU):

```bash
torchrun --standalone --nproc_per_node=1 run_elign.py \
  --config-name fed_grpo_qm9_energy_force_group4x6 \
  save_path=outputs/elign/qm9/my_run \
  wandb.enabled=false
```

## Practical notes

### Grouping and `group_index`

FED-GRPO relies on grouping multiple samples per prompt. The grouping is controlled by:

- `dataloader.sample_group_size`
- `dataloader.each_prompt_sample`

Total rollouts per dataloader batch is:

```
batch_size = sample_group_size * each_prompt_sample
```

### MLFF cost

If reward shaping is enabled, reward computation may evaluate many intermediate diffusion steps.
This can multiply MLFF calls significantly. Start with terminal-only rewards and add shaping only when needed.

### Progress bars

Some ELIGN components enable tqdm progress bars only when `ELIGN_TQDM=1` is set in the environment.

