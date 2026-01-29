# FED-GRPO (Force and Energy Disentangled Group Relative Policy Optimization)

FED-GRPO is the RL algorithm used by ELIGN to post-train an equivariant diffusion policy.

Conceptually, FED-GRPO is a **group-relative** policy-gradient method:

- Generate *groups* of samples per prompt (a “prompt group” is identified by `group_index`).
- Compute per-sample rewards using force/energy signals from an ML force field.
- Convert rewards to **advantages by normalizing within each prompt group**.
- Update the diffusion policy using a **clipped** policy-gradient objective (PPO-style).

The key “FED” part is that **force and energy advantages are disentangled**:

- compute group-normalized force advantages
- compute group-normalized energy advantages
- mix them with configurable weights (`reward.force_adv_weight`, `reward.energy_adv_weight`)

This makes it easier to control learning dynamics when one signal (often forces) dominates raw reward scales.

## Where it lives in code

- Trainer (advantage computation, orchestration): `elign/trainer/fed_grpo_trainer.py` (`FedGrpoTrainer`)
- Actor (PPO-style update, KL penalty): `elign/worker/actor/edm_actor.py` (`EDMActor`)

## Paper alignment notes

If you are reading the Elign paper alongside this repo, start with:

- `docs/paper_code_map.md` (Algorithm/Eq → file/function mapping)

In particular, the paper’s PBRS return-to-go (Eq. 4) is enabled via:

- `reward.shaping.mode=pbrs_return_to_go`

## Advantage computation (group-relative)

Within a prompt group, for rewards `r_i`:

- Compute group mean and std
- Normalize: `a_i = (r_i - mean(r_group)) / (std(r_group) + eps)`

If reward shaping is enabled, FED-GRPO can operate on per-timestep reward traces (`*_rewards_ts`) and compute
per-timestep normalized advantages, then aggregate them into scalar advantages.

**Force/energy disentanglement**

When both `force_rewards(_ts)` and `energy_rewards(_ts)` exist:

- Normalize force rewards within-group → `a_force`
- Normalize energy rewards within-group → `a_energy`
- Combine:

```
a = force_adv_weight * a_force + energy_adv_weight * a_energy
```

## Policy update (PPO-style clipped objective)

For each diffusion transition, ELIGN computes:

- behavior-policy log-probability (from rollout)
- current-policy log-probability (recomputed during update)
- ratio `exp(logp_new - logp_old)`

The clipped surrogate objective is applied with `train.clip_range`.

## KL regularization (optional)

`EDMActor` optionally adds a KL penalty to a reference policy drift (mu) when:

- `train.kl_penalty_weight > 0`, and
- a reference model is available

This is useful to keep the post-trained policy close to the pretrained backbone early in training.

## Common knobs

- Grouping:
  - `dataloader.sample_group_size`
  - `dataloader.each_prompt_sample`
- Advantage mixing:
  - `reward.force_adv_weight`
  - `reward.energy_adv_weight`
- PPO update:
  - `train.clip_range`
  - `train.learning_rate`
  - `train.train_micro_batch_size`
  - `train.gradient_accumulation_steps`
- Reward shaping:
  - `reward.shaping.enabled`
  - `reward.shaping.scheduler.*`
- Stability/chemistry gating:
  - `reward.energy_only_if_stable`
  - `reward.force_only_if_stable`
  - `filters.invalid_penalty_scale`
