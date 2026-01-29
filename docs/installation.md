# Installation

This repo has two major components:

- `edm_source/`: the E(3)-equivariant diffusion backbone and dataset utilities
- `elign/`: ELIGN post-training code (FED-GRPO)

## System requirements

- Python 3.9+ (recommended 3.10)
- A working PyTorch install (CPU is fine for smoke tests; CUDA recommended for training)
- RDKit (recommended via conda) for RDKit-based validity/uniqueness metrics and filtering

Optional:

- `fairchem` / FAIRChem UMA predictor backend (for UMA-based force/energy rewards)

## Recommended environment (conda)

```bash
conda create -n elign python=3.10 -c conda-forge rdkit
conda activate elign
pip install -r requirements.txt
```

## Optional: FAIRChem / UMA MLFF backend

ELIGN’s default reward implementation (`UMAForceReward`) uses:

- `from fairchem.core import pretrained_mlip` (see `edm_source/mlff_modules/mlff_utils.py`)

You must install a compatible `fairchem` package (or build from source) and ensure UMA weights can be loaded on your machine.

If UMA weights require authentication, use:

```bash
huggingface-cli login
# or
export HF_TOKEN=...
```

## Quick verification

1) Import check:

```bash
python -c "import elign; import edm_source"
```

2) Run unit tests:

```bash
pytest -q
```

3) Smoke test ELIGN wiring (no MLFF required):

```bash
python run_elign.py \
  --config-name fed_grpo_config \
  reward.type=dummy \
  wandb.enabled=false \
  dataloader.epoches=1 \
  dataloader.sample_group_size=2 \
  dataloader.each_prompt_sample=4 \
  model.time_step=50
```

