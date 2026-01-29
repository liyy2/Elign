# Repository Guidelines

This repo contains ELIGN, a post-training method for equivariant diffusion models, and its FED-GRPO trainer.

## Project Structure & Module Organization

- `edm_source/` – diffusion backbone (EDM-style) and dataset utilities
  - `edm_source/equivariant_diffusion/` – diffusion model and sampling logic
  - `edm_source/egnn/` – EGNN layers and utilities
  - `edm_source/qm9/` – QM9 dataset code and stability metrics
  - Entry points: `edm_source/main_qm9.py`, `edm_source/main_geom_drugs.py`, `edm_source/eval_mlff_guided.py`
- `elign/` – ELIGN post-training stack (FED-GRPO)
  - `elign/trainer/fed_grpo_trainer.py` – trainer/advantage computation/checkpointing
  - `elign/worker/actor/edm_actor.py` – PPO-style update + optional KL penalty
  - `elign/worker/reward/force.py` – UMA force/energy reward + shaping
  - `elign/worker/filter/filter.py` – RDKit-based filtering + penalties
- Top-level scripts:
  - `run_elign.py` – main ELIGN/FED-GRPO entrypoint (Hydra)
  - `eval_elign_rollout.py` – sampling rollouts from a trained run directory
  - `compute_elign_metrics.py` – RDKit/stability/MLFF metrics and optional repair

Outputs (checkpoints, samples) should be written under `outputs/` (recommended) or a user-provided `save_path`.

## Build, Test, and Development Commands

- Environment setup (recommended):
  - `conda create -n elign python=3.10 -c conda-forge rdkit && conda activate elign`
  - `pip install -r requirements.txt`
- Run unit tests:
  - `pytest -q`
- Train backbone (QM9):
  - `python edm_source/main_qm9.py --exp_name edm_qm9 --n_epochs 3000 --diffusion_steps 1000`
- Post-train with ELIGN (smoke test):
  - `python run_elign.py --config-name fed_grpo_config reward.type=dummy wandb.enabled=false dataloader.epoches=1`

## Coding Style & Naming Conventions

- Python 3.x, 4-space indentation, PEP 8.
- Naming: `snake_case` for functions/variables, `PascalCase` for classes, `UPPER_CASE` for constants.
- Prefer explicit args over globals; keep functions small and testable.

## Testing Guidelines

- Prefer `pytest` with tests under `tests/` named `test_*.py`.
- Keep tests lightweight (avoid heavy dataset downloads).

## Security & Configuration Tips

- Do not commit credentials. Use `huggingface-cli login` or export `HF_TOKEN` at runtime.
- Write large artifacts to `outputs/` and avoid committing them.

