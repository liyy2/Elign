# Reward computation (QM9 VERL / DDPO)

This directory contains reward implementations used during VERL post-training.

The main one for QM9 is `UMAForceReward` in `force.py`, which scores sampled molecules with an
ML force field (UMA) using:

- **Forces** (always): encourage locally relaxed geometries.
- **Energies** (optional): encourage low energy, but only when geometry is reasonable.

## Terminal-only reward (recommended starting point)

When `reward.shaping.enabled=false`, the reward is computed once on the final sample.

For each molecule `i` with atoms `a=1..n`:

1) **MLFF forces**: get per-atom force vectors `F[i,a]`.
   - If `reward.force_clip_threshold` is set, per-atom magnitudes are clipped *before* aggregation:
     `F[i,a] <- F[i,a] * min(1, force_clip_threshold / (||F[i,a]|| + eps))`.

2) **Aggregate force to a scalar magnitude** `m[i]`:
   - `reward.force_aggregation="rms"`: `m[i] = sqrt(mean_a ||F[i,a]||^2)`
   - `reward.force_aggregation="max"`: `m[i] = max_a ||F[i,a]||`

3) **Force reward**:
   - `force_rewards[i] = -m[i]`

   Optional gate: if `reward.force_only_if_stable=true`, the *force term* is applied only to
   stable molecules:
   - `force_rewards[i] *= stable[i]`
   - (stability/valence penalties below are still applied, so unstable molecules are not rewarded)

4) **Stability bonus (optional)**:
   - `stable[i]` is computed by `check_stability(...)` (0/1).
   - `stability_rewards[i] = stability_weight * (2*stable[i] - 1)`  (i.e. `+w` if stable, `-w` otherwise)
   - `force_rewards[i] += stability_rewards[i]`

5) **Atom-level stability shaping (optional)**:
   - `check_stability(...)` also returns how many atoms satisfy valence rules under the same bond-order heuristic.
   - Define `atom_stability[i] = (# stable atoms) / (# atoms)` in `[0, 1]`.
   - `atom_stability_rewards[i] = -atom_stability_weight * (1 - atom_stability[i])`
   - `force_rewards[i] += atom_stability_rewards[i]`

6) **Valence under-bond penalty (optional)**:
   - Compute the *missing* bond order needed to reach a valid valence for each atom under the same
     bond-order heuristic used by `check_stability` (QM9/GEOM bond thresholds).
   - For atom types with multiple allowed valences (e.g., N can be 3 or 5), the under/over
     attribution is chosen to minimize the *weighted* penalty given the configured under/over
     weights (so N=4 can be treated as either direction depending on weights).
   - Let `missing_bonds[i]` be the total missing valence across atoms in molecule `i`.
   - `valence_underbond_rewards[i] = -valence_underbond_weight * missing_bonds[i]`
   - `force_rewards[i] += valence_underbond_rewards[i]`

7) **Valence over-bond penalty (optional)**:
   - Compute the *excess* bond order above the allowed valence for each atom under the same
     bond-order heuristic used by `check_stability`.
   - Let `excess_bonds[i]` be the total excess valence across atoms in molecule `i`.
   - `valence_overbond_rewards[i] = -valence_overbond_weight * excess_bonds[i]`
   - `force_rewards[i] += valence_overbond_rewards[i]`

8) **Energy reward (optional)** (only if `reward.use_energy=true`):
   - Get MLFF energy `E[i]`.
   - If `reward.energy_normalize_by_atoms=true`, divide by the number of atoms.
   - Transform: `E'[i] = (E[i] + energy_transform_offset) / energy_transform_scale`
     and optionally clamp to `[-energy_transform_clip, +energy_transform_clip]`.
   - `energy_rewards[i] = -E'[i]` (minimize energy).
   - If `reward.energy_only_if_stable=true`, gate energy to stable molecules:
     `energy_rewards[i] *= stable[i]`.

9) **Total reward**:
   - `rewards[i] = force_weight * force_rewards[i] + energy_weight * energy_rewards[i]`

### Smooth valence shaping (optional)

The discrete bond-order heuristic (`get_bond_order`) is a hard threshold: a bond can flip from
present→absent when a single distance crosses the cutoff. That makes the under/over-bond penalties
very sparse when the model is *almost* correct (e.g., one H is 0.02–0.05Å too far).

If enabled, `UMAForceReward` adds a **sigmoid-smoothed** version of the valence penalties:

- `reward.valence_underbond_soft_weight`
- `reward.valence_overbond_soft_weight`
- `reward.valence_soft_temperature`

This computes a soft bond order per pair by applying sigmoids at the same single/double/triple
bond thresholds and then penalizes the soft valence deficit/excess per atom. It provides a dense
signal that helps reduce the RDKit-valid-but-unstable tail without changing eval-time metrics.

## Reward shaping mode (optional)

When `reward.shaping.enabled=true`, `UMAForceReward` also emits per-diffusion-step reward traces:
`force_rewards_ts` and (optionally) `energy_rewards_ts`.

The trainer uses these traces to compute per-step GRPO-style advantages (see below). In practice,
shaping can be noisier because it requires many MLFF evaluations on intermediate diffusion states.

## What DDPOTrainer actually optimizes (important)

In `DDPOTrainer.compute_advantage()`:

- Rollouts are grouped by `group_index` (same prompt, multiple samples).
- If both force and energy channels exist, the trainer **normalizes them separately within each
  group** and then mixes the normalized advantages:
  - `adv = force_adv_weight * adv_force + energy_adv_weight * adv_energy`

Because of this group-wise normalization, **`energy_adv_weight` / `force_adv_weight` are the main
knobs** for trading off force vs energy influence during learning (raw `energy_weight` mostly
rescales rewards and often cancels out under normalization).

## Filters + RDKit metrics

The `Filter` module can:

- deduplicate within a batch (`filters.enable_filtering`)
- add penalties:
  - `filters.invalid_penalty_scale`: penalty only for RDKit-invalid molecules
  - `filters.penalty_scale`: novelty penalty relative to the QM9 SMILES set
  - `filters.duplicate_penalty_scale`: anti-collapse penalty when multiple samples in the same
    rollout batch share the same canonical SMILES. When `filters.enable_filtering=true`, the full
    cost `-duplicate_penalty_scale * (count - 1)` is applied to the kept representative so PPO
    observes the collapse signal (otherwise most duplicates would be dropped before learning).

RDKit SMILES are canonicalized on the **largest fragment** (matching `BasicMolecularMetrics`), so
models cannot inflate training-time uniqueness by appending many tiny disconnected fragments.

Penalties are injected into all available reward channels (`rewards`, `force_rewards`, and
`energy_rewards`, plus the corresponding `*_ts` tensors when present) so they always affect learning
even when DDPOTrainer uses separate force/energy advantages.

When `filters.invalid_penalty_scale > 0`, the filter also **zeros `energy_rewards` for RDKit-invalid
molecules**, so the energy channel cannot accidentally incentivize invalid chemistry via
out-of-distribution MLFF energies.

During training, these are logged to W&B:

- `train/rdkit_validity`
- `train/rdkit_uniqueness`
- `train/validity_x_uniqueness`

## Eval-time stability repair (optional)

Even when RDKit validity is high, QM9 stability can lag because `check_stability` is strict and
sensitive to small geometry errors (e.g. a single C/N valence off by ±1 due to a borderline bond
length). Many of these cases have **large UMA forces**, indicating they are simply not relaxed.

For evaluation, `compute_verl_metrics.py` supports a small MLFF-based relaxation loop:

- `--repair-invalid`: apply to RDKit-invalid molecules only
- `--repair-unstable`: apply to QM9-unstable molecules only

This typically improves stability (and often RDKit validity) without changing the underlying
diffusion policy.
