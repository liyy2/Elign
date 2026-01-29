# Config reference (Hydra)

ELIGN post-training is configured via Hydra YAML files under:

- `elign/trainer/config/`

The default config is:

- `elign/trainer/config/fed_grpo_config.yaml`

You can override any key via CLI, e.g.:

```bash
python run_elign.py --config-name fed_grpo_config train.learning_rate=2e-6 reward.use_energy=true
```

## Top-level keys

- `seed`: base seed (rank-aware under `torchrun`)
- `save_path`: where training artifacts are written (recommended: `outputs/elign/...`)
- `resume`: resume from an optimizer/model checkpoint
- `checkpoint_path`: path to a checkpoint when `resume=true`
- `best_checkpoint_metric` / `best_checkpoint_mode`: controls which checkpoint is considered “best”

## `dataloader`

Controls how many samples are generated per update iteration:

- `sample_group_size`: number of distinct prompts in a batch
- `each_prompt_sample`: number of rollouts per prompt (group size)
- `micro_batch_size`: maximum number of prompts sent to the sampler at once (memory control)
- `epoches`: number of rollout/update iterations to run
- `smiles_path`: QM9 SMILES pickle used for novelty penalties (when enabled)

Derived quantity:

```
rollouts_per_iter = sample_group_size * each_prompt_sample
```

## `model`

Backbone diffusion model pointers:

- `config`: path to the EDM `args.pickle`
- `model_path`: path to pretrained diffusion weights (e.g., `generative_model_ema.npy`)
- `time_step`: number of diffusion steps used during sampling/training
- `share_initial_noise`: whether to share the initial noise across samples in a prompt group

## `reward`

Reward implementation and shaping:

- `type`: `uma` or `dummy`
- `mlff_model`: UMA model name (e.g. `uma-s-1p1`)
- `use_energy`: include an energy term
- `force_aggregation`: `rms` or `max`
- `force_clip_threshold`: optional per-atom force clipping threshold
- `energy_transform_offset` / `energy_transform_scale` / `energy_transform_clip`: energy shaping
- `energy_only_if_stable`: gate energy reward to stable molecules
- `force_only_if_stable`: gate force reward to stable molecules
- `stability_weight`: optional +w/-w stability bonus
- `valence_underbond_weight` / `valence_overbond_weight`: optional chemistry penalties

**FED-GRPO mixing weights**

- `force_adv_weight`
- `energy_adv_weight`

These are applied to **normalized** group-relative advantages (not raw rewards). They are the main knobs
for balancing force vs energy influence during learning.

### `reward.shaping`

If `reward.shaping.enabled=true`, ELIGN computes additional reward traces across selected diffusion steps:

- `mode`:
  - `delta`: legacy per-step PBRS deltas `r_shape` (Eq. 3)
  - `pbrs_return_to_go`: paper-aligned PBRS return-to-go `G_t^(E) = γ^t Ψ_0 - Ψ_t` (Eq. 4)
- `mlff_batch_size`: MLFF evaluation batch size (controls memory)
- `terminal_weight`: weight applied to the terminal step when aggregating shaped advantages
- `scheduler`: how diffusion steps are selected for shaping (uniform or adaptive)

Start with terminal-only rewards (`enabled=false`) and enable shaping only after you have a stable baseline.

## `filters`

Training-time filtering and penalties:

- `enable_filtering`: deduplicate within-batch by canonical SMILES (largest fragment)
- `enable_penalty`: novelty penalty vs the QM9 SMILES set
- `penalty_scale`: novelty penalty magnitude
- `invalid_penalty_scale`: optional penalty for RDKit-invalid molecules
- `duplicate_penalty_scale`: optional anti-collapse penalty for duplicates within a rollout batch

## `train`

Optimization hyperparameters:

- `learning_rate`
- `clip_range`: PPO clip range
- `kl_penalty_weight`: optional KL-to-reference penalty
- `train_micro_batch_size`: batch size per optimizer step
- `gradient_accumulation_steps`: accumulate gradients across multiple micro-batches
- `max_grad_norm`
- `epoch_per_rollout`: number of optimizer epochs per rollout batch

Optional time-based stopping:

- `max_time_hours`

## `wandb`

- `enabled`
- `wandb_project`
- `wandb_name`

Set `WANDB_MODE=offline` for fully offline runs.
