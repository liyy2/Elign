#!/bin/bash
#SBATCH --job-name=post_train_qm9
#SBATCH --output=post_train_qm9_%j.out
#SBATCH --error=post_train_qm9_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu_h200

set -euo pipefail

# Usage:
#   sbatch post_train_diffusion_qm9.sh
# Optional overrides:
#   sbatch --export=ALL,MODEL_CONFIG=./pretrained/edm/edm_qm9/args.pickle,MODEL_WEIGHTS=./pretrained/edm/edm_qm9/generative_model_ema.npy post_train_diffusion_qm9.sh
#   sbatch --export=ALL,WANDB_ENABLED=1,WANDB_PROJECT=myproj post_train_diffusion_qm9.sh
#
# Notes:
# - No tokens are embedded. Set `WANDB_API_KEY`/`WANDB_MODE` and any HF credentials in your environment if needed.
# - Adds repo paths to `PYTHONPATH` so imports work even if `run_elign.py` has a stale hardcoded sys.path.

CONDA_ENV="${CONDA_ENV:-edm}"
WANDB_ENABLED="${WANDB_ENABLED:-0}"   # 1=enable wandb.init, 0=disable
WANDB_PROJECT="${WANDB_PROJECT:-elign}"

# Recommended default: the best configuration found so far in this tuning loop.
CONFIG_NAME="${CONFIG_NAME:-fed_grpo_qm9_energy_force_group4x6}"

# Starting checkpoint for the diffusion policy (EDM)
MODEL_CONFIG="${MODEL_CONFIG:-./pretrained/edm/edm_qm9/args.pickle}"
MODEL_WEIGHTS="${MODEL_WEIGHTS:-./pretrained/edm/edm_qm9/generative_model_ema.npy}"

# Optional: resume a FED-GRPO checkpoint (full optimizer/model state).
# NOTE: `checkpoint_path` is ignored unless `resume=true`.
CHECKPOINT_PATH="${CHECKPOINT_PATH:-}"

# QM9 smiles pickle used by the novelty filter (must exist even if filtering/penalty is disabled)
SMILES_PATH="${SMILES_PATH:-qm9/temp/qm9_smiles.pickle}"

if command -v module >/dev/null 2>&1; then
  module load miniconda
fi

if command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook)"
  conda activate "${CONDA_ENV}"
fi

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
cd "${REPO_ROOT}"

export PYTHONPATH="${REPO_ROOT}:${REPO_ROOT}/edm_source:${PYTHONPATH:-}"

sanitize_for_name() {
  local value="$1"
  value="${value//[^a-zA-Z0-9]/_}"
  value="${value##_}"
  value="${value%%_}"
  echo "${value}"
}

# ----------------------------
# Optimization / training loop
# ----------------------------
LEARNING_RATE="${LEARNING_RATE:-4e-6}"
CLIP_RANGE="${CLIP_RANGE:-2e-3}"
# Matches `fed_grpo_qm9_energy_force_group4x6.yaml`
TRAIN_MICRO_BATCH_SIZE="${TRAIN_MICRO_BATCH_SIZE:-8}"
EPOCH_PER_ROLLOUT="${EPOCH_PER_ROLLOUT:-1}"
# Matches `fed_grpo_qm9_energy_force_group4x6.yaml`
KL_PENALTY_WEIGHT="${KL_PENALTY_WEIGHT:-0.08}"

# ----------------------------
# Diffusion rollout settings
# ----------------------------
# Best-performing split: 4 prompts × 6 samples each = 24 rollouts/iter.
SAMPLE_GROUP_SIZE="${SAMPLE_GROUP_SIZE:-4}"
EACH_PROMPT_SAMPLE="${EACH_PROMPT_SAMPLE:-6}"
TIME_STEP="${TIME_STEP:-1000}"
SHARE_INITIAL_NOISE="${SHARE_INITIAL_NOISE:-true}"
FORCE_ALIGNMENT_ENABLED="${FORCE_ALIGNMENT_ENABLED:-false}"

# ----------------------------
# Reward configuration
# ----------------------------
USE_ENERGY="${USE_ENERGY:-true}"
# Default to UMA-S for stability. UMA-M can be enabled via `MLFF_MODEL=uma-m-1p1`,
# but in our quick tests it was much more sensitive to PPO hyperparameters.
MLFF_MODEL="${MLFF_MODEL:-uma-s-1p1}"
MLFF_BATCH_SIZE="${MLFF_BATCH_SIZE:-32}"
FORCE_AGGREGATION="${FORCE_AGGREGATION:-rms}"
STABILITY_WEIGHT="${STABILITY_WEIGHT:-2.0}"
ENERGY_ONLY_IF_STABLE="${ENERGY_ONLY_IF_STABLE:-true}"
ENERGY_ADV_WEIGHT="${ENERGY_ADV_WEIGHT:-0.05}"
# Clip per-atom force magnitudes above this threshold before aggregating (0/empty disables).
FORCE_CLIP_THRESHOLD="${FORCE_CLIP_THRESHOLD:-2.0}"

# Shaping settings
SKIP_PREFIX="${SKIP_PREFIX:-700}"
# Shaping is noisier and was not needed for the best run so far.
REWARD_SHAPING_ENABLED="${REWARD_SHAPING_ENABLED:-false}"
SHAPING_ONLY_ENERGY="${SHAPING_ONLY_ENERGY:-false}"
TERMINAL_WEIGHT="${TERMINAL_WEIGHT:-5.0}"

# ----------------------------
# Novelty filtering configuration
# ----------------------------
ENABLE_NOVELTY_PENALTY="${ENABLE_NOVELTY_PENALTY:-false}"
NOVELTY_PENALTY_SCALE="${NOVELTY_PENALTY_SCALE:-0.5}"
ENABLE_FILTERING="${ENABLE_FILTERING:-true}"
INVALID_PENALTY_SCALE="${INVALID_PENALTY_SCALE:-2.0}"
# Penalize duplicate SMILES within a rollout batch (anti-collapse; 0 disables).
DUPLICATE_PENALTY_SCALE="${DUPLICATE_PENALTY_SCALE:-0.0}"

# ----------------------------
# Scheduler configuration
# ----------------------------
SCHEDULER_NAME="${SCHEDULER_NAME:-cosine}"
SCHEDULER_WARMUP_STEPS="${SCHEDULER_WARMUP_STEPS:-60}"
SCHEDULER_TOTAL_STEPS="${SCHEDULER_TOTAL_STEPS:-1500}"
SCHEDULER_MIN_LR_RATIO="${SCHEDULER_MIN_LR_RATIO:-0.3}"

# ----------------------------
# Run naming / save path
# ----------------------------
timestamp=$(date +"%Y%m%d_%H%M%S")
MODEL_TAG=$(sanitize_for_name "${MLFF_MODEL}")
RUN_NAME="elign_qm9_${MODEL_TAG}_lr_$(sanitize_for_name "${LEARNING_RATE}")_${timestamp}"

SAVE_ROOT="${SAVE_ROOT:-${REPO_ROOT}/outputs/elign/qm9}"
SAVE_PATH="${SAVE_ROOT}/${RUN_NAME}"
mkdir -p "${SAVE_PATH}"

GPUS_PER_NODE="${GPUS_PER_NODE:-1}"
USE_TORCHRUN="${USE_TORCHRUN:-1}"  # 1=use torchrun (DDP-ready), 0=run single-process python

export MASTER_ADDR=${MASTER_ADDR:-$(hostname)}
export MASTER_PORT=${MASTER_PORT:-29500}

declare -a SHAPING_FLAGS
if [[ "${REWARD_SHAPING_ENABLED}" == true ]]; then
  SHAPING_FLAGS=(
    "reward.shaping.enabled=true"
    "reward.shaping.scheduler.skip_prefix=${SKIP_PREFIX}"
    "reward.shaping.only_energy_reshape=${SHAPING_ONLY_ENERGY}"
    "reward.shaping.terminal_weight=${TERMINAL_WEIGHT}"
  )
else
  SHAPING_FLAGS=("reward.shaping.enabled=false")
fi

if [[ "${WANDB_ENABLED}" == "1" ]]; then
  export WANDB_MODE="${WANDB_MODE:-online}"
  WANDB_FLAGS=("wandb.enabled=true" "wandb.wandb_project=${WANDB_PROJECT}" "wandb.wandb_name=${RUN_NAME}")
else
  export WANDB_MODE="${WANDB_MODE:-offline}"
  WANDB_FLAGS=("wandb.enabled=false")
fi

declare -a RESUME_FLAGS=()
if [[ -n "${CHECKPOINT_PATH}" ]]; then
  RESUME_FLAGS=("resume=true" "checkpoint_path=${CHECKPOINT_PATH}")
fi

# Launch training (Hydra)
if [[ "${USE_TORCHRUN}" == "1" ]]; then
  LAUNCHER=(torchrun --standalone --nproc_per_node="${GPUS_PER_NODE}")
else
  LAUNCHER=(python -u)
fi

"${LAUNCHER[@]}" run_elign.py \
  --config-name "${CONFIG_NAME}" \
  "${WANDB_FLAGS[@]}" \
  save_path="${SAVE_PATH}" \
  "${RESUME_FLAGS[@]}" \
  model.config="${MODEL_CONFIG}" \
  model.model_path="${MODEL_WEIGHTS}" \
  dataloader.smiles_path="${SMILES_PATH}" \
  train.learning_rate="${LEARNING_RATE}" \
  train.clip_range="${CLIP_RANGE}" \
  train.kl_penalty_weight="${KL_PENALTY_WEIGHT}" \
  train.train_micro_batch_size="${TRAIN_MICRO_BATCH_SIZE}" \
  train.epoch_per_rollout="${EPOCH_PER_ROLLOUT}" \
  model.time_step="${TIME_STEP}" \
  model.share_initial_noise="${SHARE_INITIAL_NOISE}" \
  dataloader.sample_group_size="${SAMPLE_GROUP_SIZE}" \
  dataloader.each_prompt_sample="${EACH_PROMPT_SAMPLE}" \
  train.force_alignment_enabled="${FORCE_ALIGNMENT_ENABLED}" \
  reward.use_energy="${USE_ENERGY}" \
  reward.mlff_model="${MLFF_MODEL}" \
  reward.shaping.mlff_batch_size="${MLFF_BATCH_SIZE}" \
  reward.force_aggregation="${FORCE_AGGREGATION}" \
  reward.force_clip_threshold="${FORCE_CLIP_THRESHOLD}" \
  reward.stability_weight="${STABILITY_WEIGHT}" \
  reward.energy_only_if_stable="${ENERGY_ONLY_IF_STABLE}" \
  reward.energy_adv_weight="${ENERGY_ADV_WEIGHT}" \
  filters.enable_filtering="${ENABLE_FILTERING}" \
  filters.enable_penalty="${ENABLE_NOVELTY_PENALTY}" \
  filters.penalty_scale="${NOVELTY_PENALTY_SCALE}" \
  filters.invalid_penalty_scale="${INVALID_PENALTY_SCALE}" \
  filters.duplicate_penalty_scale="${DUPLICATE_PENALTY_SCALE}" \
  train.scheduler.name="${SCHEDULER_NAME}" \
  train.scheduler.warmup_steps="${SCHEDULER_WARMUP_STEPS}" \
  train.scheduler.total_steps="${SCHEDULER_TOTAL_STEPS}" \
  train.scheduler.min_lr_ratio="${SCHEDULER_MIN_LR_RATIO}" \
  "${SHAPING_FLAGS[@]}"
