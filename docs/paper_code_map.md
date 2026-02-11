# Elign paper ↔ code map

This repo contains an implementation of **Elign** (post-training for E(3)-equivariant diffusion models) and its RL optimizer **FED-GRPO**.

Paper: [arXiv:2601.21985](https://arxiv.org/abs/2601.21985).

Below is a pragmatic mapping from the paper’s notation / pseudocode to the corresponding code and config knobs.

## Core algorithms

### Algorithm 1 (FED-GRPO post-training)

Paper concept → implementation:

- **Rollout + grouping** (`k=1..K` samples per prompt, group-relative normalization)
  - Prompt grouping: `elign/dataloader/dataloader.py` (`EDMDataLoader`, field: `group_index`)
  - Rollout sampling: `elign/worker/rollout/edm_rollout.py` (`EDMRollout.generate_minibatch`)
- **Shared prefix** (“cache `z_start` then branch into K rollouts”)
  - Sampler support: `elign/model/edm_model.py` (`EDMModel.sample`, args `share_initial_noise`, `skip_prefix`)
  - Training-time suffix truncation: `elign/worker/actor/edm_actor.py` (`EDMActor.update_policy`, slices by `meta_info["skip_prefix"]`)
- **Clipped GRPO/PPO objective** (ratio `ξ`, clip `ε`, optional KL)
  - Per-step ratio computation + clipping: `elign/worker/actor/edm_actor.py` (`EDMActor.calculate_loss`)
  - Optional KL-to-reference penalty: `elign/worker/actor/edm_actor.py` (`train.kl_penalty_weight`)
- **Orchestration**
  - Training loop + checkpointing: `elign/trainer/fed_grpo_trainer.py` (`FedGrpoTrainer.fit`)

### Algorithm 2 (REWARD: energy PBRS return-to-go + terminal force)

Paper symbol → code:

- `ẑ_{0|t}` (predicted clean geometry from noisy state `z_t`)
  - Stored as `z0_preds` in rollout output: `elign/model/edm_model.py` (`z0_pred` in `sample_p_zs_given_zt`)
  - Passed through the pipeline as `DataProto.batch["z0_preds"]`
- `Ψ_t = -E_ϕ(ẑ_{0|t})`
  - Computed in: `elign/worker/reward/force.py` (`UMAForceReward.calculate_rewards`)
  - Exposed as `energy_rewards_ts` when `reward.shaping.mode=pbrs_return_to_go`
- `G_t^(E) = γ^t Ψ_0 - Ψ_t` (paper Eq. 4)
  - Computed in: `elign/trainer/fed_grpo_trainer.py` (`FedGrpoTrainer.compute_advantage`) when `reward.shaping.mode=pbrs_return_to_go`
- Terminal force reward `G^(F)` (paper Eq. 2 / Alg. 2 line 8)
  - Terminal force metric from UMA forces: `elign/worker/reward/force.py` (`UMAForceReward`, `force_rewards`)

## Paper equations

### Eq. (2): terminal energy + force rewards

- Force term: `elign/worker/reward/force.py`
  - Per-atom forces from UMA → aggregation (`reward.force_aggregation`)
  - Terminal scalar stored in `force_rewards`
- Energy term: `elign/worker/reward/force.py`
  - UMA energy on `z0` (+ optional `energy_atom_refs`, `energy_normalize_by_atoms`)
  - Terminal scalar stored in `energy_rewards`

### Eq. (3)/(4): PBRS shaping and telescoping return-to-go

This repo supports two shaping styles:

- `reward.shaping.mode=delta` (legacy): stores per-step PBRS deltas (an implementation of Eq. 3).
- `reward.shaping.mode=pbrs_return_to_go` (paper-aligned): stores energy potentials `Ψ_t` and computes
  the return-to-go `G_t^(E)` in the trainer (Eq. 4).

## Config mapping (paper Table 4 → Hydra keys)

Table 4 name → this repo’s config key:

- learning rate → `train.learning_rate`
- clip range → `train.clip_range`
- KL weight → `train.kl_penalty_weight`
- discount `γ` → `reward.shaping.gamma`
- weight_energy `w_E` → `reward.energy_adv_weight`
- weight_force `w_F` → `reward.force_adv_weight`
- each prompt sample `K` → `dataloader.each_prompt_sample`
- prompts per iteration → `dataloader.sample_group_size`
- time step `T` → `model.time_step`
- UMA model → `reward.mlff_model`
- MLFF batch size → `reward.shaping.mlff_batch_size`
- force aggregation → `reward.force_aggregation`
- skip prefix (shared-prefix length) → `model.skip_prefix` (and usually `reward.shaping.scheduler.skip_prefix`)

Paper-aligned example config:

- `elign/trainer/config/fed_grpo_qm9_paper_pbrs.yaml`

## Important implementation notes / naming

- `group_index` is the prompt-group identifier used for *within-group* z-score normalization.
- `dataloader.sample_group_size` is “# prompts per iteration”; `dataloader.each_prompt_sample` is the group size `K`.
- `model.skip_prefix` counts how many *early* reverse-diffusion transitions are shared (and then skipped in PPO updates).
  If `model.time_step=1000` and `model.skip_prefix=600`, PPO updates cover the last 400 diffusion transitions.
