# Repository Guidelines

## Project Overview
This repository implements **E(3) Equivariant Diffusion Models** for molecule generation, with a focus on **Reinforcement Learning (RL)** via PPO/DDPO for guided molecular design. The project supports two primary datasets:
- **QM9** – Small organic molecules (up to 9 heavy atoms)
- **GEOM-Drugs** – Drug-like molecules with larger and more diverse structures

## Project Structure & Module Organization
- Root contains experiment scripts, logs, and docs.
- **Main entry point**: `post_train_diffusion.sh` – SLURM script for RL post-training
- Key modules:
  - `equivariant_diffusion/` – diffusion model and sampling logic
  - `egnn/` – EGNN layers and utilities
  - `qm9/`, `data/`, `configs/` – datasets, loaders, and configs
  - `run_verl_diffusion.py` – RL training script (called by `post_train_diffusion.sh`)
- Outputs: checkpoints to `/home/yl2428/logs/<run_name>/`, metrics to Weights & Biases

## Key Metrics
When evaluating experiments, focus on these primary metrics:
1. **Validity** – Fraction of generated molecules that are chemically valid
2. **Uniqueness** – Fraction of valid molecules that are unique
3. **Atom Stability** – Fraction of atoms with correct valency
4. **Molecule Stability** – Fraction of molecules where all atoms are stable
5. **Validity × Uniqueness** – Combined metric (validity multiplied by uniqueness)

## Experiment Execution Guidelines (FOR CODING AGENTS)

### Main Script: `post_train_diffusion.sh`
This is the primary script for running RL experiments. It is already configured as a SLURM job with:
- GPU: H200 partition (`--partition=gpu`)
- Time: 48 hours (`--time=48:00:00`)
- Memory: 64G, 8 CPUs, 1 GPU
- Conda environment: `edm`

### Pre-flight GPU Check
Before submitting, **always check GPU availability**:
```bash
nvidia-smi
squeue -u $USER  # Check existing jobs
```
- If GPUs are available and you want interactive → run `torchrun` command directly
- Otherwise → submit via `sbatch post_train_diffusion.sh`

### Submitting Experiments
```bash
# Standard submission
cd /gpfs/radev/home/yl2428/e3_diffusion_for_molecules-main
sbatch post_train_diffusion.sh

# With checkpoint resume
CHECKPOINT_PATH=/home/yl2428/logs/<previous_run>/checkpoint sbatch post_train_diffusion.sh
```

### Key Configurable Parameters (in `post_train_diffusion.sh`)
| Parameter | Default | Description |
|-----------|---------|-------------|
| `LEARNING_RATE` | 4e-6 | PPO learning rate |
| `CLIP_RANGE` | 2e-3 | PPO clipping |
| `KL_PENALTY_WEIGHT` | 0.08 | KL divergence penalty |
| `SAMPLE_GROUP_SIZE` | 4 | Prompts per batch |
| `EACH_PROMPT_SAMPLE` | 6 | Samples per prompt (total = 24) |
| `TIME_STEP` | 1000 | Diffusion timesteps |
| `MLFF_MODEL` | uma-s-1p1 | MLFF model for energy/force reward |
| `STABILITY_WEIGHT` | 2.0 | Weight for stability reward |
| `ENERGY_ADV_WEIGHT` | 0.05 | Weight for energy advantage |
| `SKIP_PREFIX` | 700 | Timesteps to skip for reward shaping |

### Monitoring Strategy
Full experiments typically run **6+ hours** (up to 48 hours). Implement active monitoring:

1. **Monitor every 30 minutes** by checking:
   ```bash
   # Job status
   squeue -u $USER
   
   # Training logs (check stdout)
   tail -100 exp_cond_alpha_<job_id>.out
   
   # Error logs
   tail -50 exp_cond_alpha_<job_id>.err
   
   # Check Weights & Biases for real-time metrics
   # Project: look for wandb.wandb_name in logs
   ```

2. **Track key metrics** in logs and W&B:
   - `reward/mean` – Average reward (should trend upward)
   - `validity` – Should improve or stay high
   - `uniqueness` – Should stay high
   - `atom_stability` – Should improve
   - `molecule_stability` – Should improve
   - `kl_divergence` – Should stay controlled

3. **Early Stopping Criteria** – Stop the experiment if:
   - Reward plateaus for 3+ consecutive checkpoints (~1-2 hours of no improvement)
   - Validation metrics (validity, atom stability) show no upward trend
   - KL divergence explodes (model collapse)
   - Loss diverges or becomes NaN
   - Cancel via: `scancel <job_id>`

### Example Workflow for Coding Agent
```bash
# Step 1: Check GPU availability and existing jobs
nvidia-smi
squeue -u $USER

# Step 2: Submit the job
cd /gpfs/radev/home/yl2428/e3_diffusion_for_molecules-main
sbatch post_train_diffusion.sh
# Note the job ID from output

# Step 3: Monitor (repeat every 30 minutes)
squeue -u $USER
tail -100 exp_cond_alpha_<job_id>.out | grep -E "(validity|uniqueness|stability|reward|loss|kl)"

# Step 4: Check for plateau
# If reward/metrics not improving for ~1-2 hours, consider early stop:
scancel <job_id>
```

### Modifying Experiments
To run different configurations, edit variables at the top of `post_train_diffusion.sh`:
```bash
# Example: Try different learning rate
LEARNING_RATE="1e-5"

# Example: Enable reward shaping
REWARD_SHAPING_ENABLED=true

# Example: Change MLFF model
MLFF_MODEL="uma-m-1p1"
```

## Build, Test, and Development Commands
- Environment setup:
  ```bash
  module load miniconda
  conda activate edm
  ```
- For QM9 pre-training (if needed):
  ```bash
  python main_qm9.py --exp_name edm_qm9 --n_epochs 3000 --diffusion_steps 1000
  ```

## Coding Style & Naming Conventions
- Python 3.x, 4-space indentation, PEP 8.
- Naming: `snake_case` for functions/variables, `PascalCase` for classes, `UPPER_CASE` for constants.
- Keep functions small and pure where possible; prefer explicit args over globals.

## Commit & Pull Request Guidelines
- Commits: concise, imperative subject (<=72 chars), e.g., `Adjust KL penalty weight for stability`.
- Include context in body when changing behavior or configs.
- PRs: clear description, motivation, before/after metrics or logs when relevant.

## Security & Configuration Tips
- Do not commit credentials. W&B login should be configured separately.
- Large artifacts: written to `/home/yl2428/logs/` – do not commit.
- Add any local `.env` or cache paths to `.gitignore`.
