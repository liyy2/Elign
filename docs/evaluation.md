# Evaluation

ELIGN evaluation is a two-step process:

1) sample molecules (rollouts) from a trained checkpoint
2) compute metrics over the sampled set

## 1) Sample rollouts

Use `eval_elign_rollout.py`:

```bash
python eval_elign_rollout.py \
  --run-dir outputs/elign/qm9/my_run \
  --num-molecules 1024 \
  --output outputs/elign/qm9/my_run/eval_rollouts.pt
```

Key options:

- `--checkpoint`: pick a specific checkpoint (defaults to latest/best if present)
- `--time-step`: override diffusion steps used during sampling
- `--sample-group-size`, `--each-prompt-sample`: control grouping during eval
- `--share-initial-noise`: force shared initial noise within groups (for ablations)
- `--skip-prefix`: share a diffusion prefix across a group (useful to reduce sample variance)

**Backbone args.pickle**

If `--args-pickle` is omitted, the script tries:

1) `<run-dir>/args.pickle` (if you copied it there), then
2) `<run-dir>/config.yaml -> model.config` (recommended)

## 2) Compute metrics

Use `compute_elign_metrics.py`:

```bash
python compute_elign_metrics.py \
  --run-dir outputs/elign/qm9/my_run \
  --samples outputs/elign/qm9/my_run/eval_rollouts.pt \
  --output outputs/elign/qm9/my_run/eval_metrics.json
```

This computes:

- RDKit validity / uniqueness / novelty (largest-fragment canonicalization)
- stability metrics via `edm_source/qm9/analyze.py::check_stability`
- optional MLFF-based metrics (UMA forces and energies), unless `--skip-mlff-metrics` is set

### Optional: MLFF-based “repair” passes

`compute_elign_metrics.py` supports lightweight post-processing that nudges positions along MLFF forces:

- `--repair-invalid`: apply to RDKit-invalid molecules only
- `--repair-unstable`: apply to QM9-unstable molecules only
- `--repair-add-h`: add explicit hydrogens before RDKit checks (useful for some edge cases)
- `--repair-steps` / `--repair-alpha`: control the number and size of force steps

These are evaluation-only utilities; they do not change the trained policy.

### Saving raw tensors

To save per-sample tensors (for downstream analysis), use:

```bash
python compute_elign_metrics.py \
  --run-dir outputs/elign/qm9/my_run \
  --samples outputs/elign/qm9/my_run/eval_rollouts.pt \
  --save-raw outputs/elign/qm9/my_run/eval_metrics_raw.pt
```

