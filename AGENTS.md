# Repository Guidelines

## Project Structure & Module Organization
- Root contains experiment scripts, logs, and docs. Core code lives under `e3_diffusion_for_molecules-main/`.
- Key modules (paths are relative to `e3_diffusion_for_molecules-main/`):
  - `equivariant_diffusion/` – diffusion model and sampling logic
  - `egnn/` – EGNN layers and utilities
  - `qm9/`, `data/`, `configs/` – datasets, loaders, and configs
  - Entry points: `main_qm9.py`, `main_geom_drugs.py`, `eval_mlff_guided.py`
- Outputs (checkpoints, samples) are written to `outputs/` unless overridden by flags.

## Build, Test, and Development Commands
- Environment setup (recommended):
  - `conda create -c conda-forge -n molecular-diffusion rdkit && conda activate molecular-diffusion`
  - `pip install -r e3_diffusion_for_molecules-main/requirements.txt`
- Train (QM9 example):
  - `cd e3_diffusion_for_molecules-main && python main_qm9.py --exp_name edm_qm9 --n_epochs 3000 --diffusion_steps 1000`
- Train (GEOM-Drugs example):
  - `python main_geom_drugs.py --exp_name edm_geom_drugs --diffusion_steps 1000`
- Evaluate with MLFF guidance:
  - `python eval_mlff_guided.py --model_path outputs/edm_qm9 --n_samples 100`

## Coding Style & Naming Conventions
- Python 3.x, 4-space indentation, PEP 8.
- Naming: `snake_case` for functions/variables, `PascalCase` for classes, `UPPER_CASE` for constants, module files `lower_snake_case.py`.
- Keep functions small and pure where possible; prefer explicit args over globals.
- If formatters are available, run: `black . && isort .` (optional but encouraged).

## Testing Guidelines
- Prefer `pytest` with tests under `tests/` named `test_*.py`.
- Aim for coverage on utility functions, samplers, and data transforms.
- Quick run: `pytest -q`. Add minimal fixtures for dataset stubs to avoid heavy downloads.

## Commit & Pull Request Guidelines
- Commits: concise, imperative subject (<=72 chars), e.g., `Add MLFF guidance flag to evaluator`.
- Include context in body when changing behavior or configs.
- PRs: clear description, motivation, reproduction steps, and sample command lines; link related issues; include before/after metrics or logs when relevant.

## Security & Configuration Tips
- Do not commit credentials. Use `huggingface-cli login` or export `HF_TOKEN` at runtime.
- Large artifacts: write to `outputs/` and avoid committing them.
- Add any local `.env` or cache paths to `.gitignore`.
