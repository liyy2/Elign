#!/bin/bash
#SBATCH --job-name=post_train_geom
#SBATCH --output=post_train_geom_%j.out
#SBATCH --error=post_train_geom_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu_h200

set -euo pipefail

# Usage:
#   sbatch --export=ALL,GEOM_DATA_FILE=/path/to/geom_drugs_30.npy post_train_diffusion_geom.sh
#
# Optional overrides:
#   sbatch --export=ALL,MODEL_CONFIG=./pretrained/edm/edm_geom_drugs/args.pickle,MODEL_WEIGHTS=./pretrained/edm/edm_geom_drugs/generative_model_ema.npy post_train_diffusion_geom.sh
#   sbatch --export=ALL,WANDB_ENABLED=1,WANDB_PROJECT=myproj post_train_diffusion_geom.sh
#
# Notes:
# - No tokens are embedded. Set `WANDB_API_KEY` / `WANDB_MODE` in your environment if needed.
# - Requires the GEOM conformation file `geom_drugs_30.npy` (pass via `GEOM_DATA_FILE`).

CONDA_ENV="${CONDA_ENV:-edm}"
WANDB_ENABLED="${WANDB_ENABLED:-0}"   # 1=enable wandb.init, 0=disable
WANDB_PROJECT="${WANDB_PROJECT:-ddpo}"

CONFIG_NAME="${CONFIG_NAME:-ddpo_geom_config}"
REWARD_TYPE="${REWARD_TYPE:-uma}"  # uma | dummy
EPOCHES="${EPOCHES:-}"             # optional override for dataloader.epoches

# Starting checkpoint for the diffusion policy (EDM)
MODEL_CONFIG="${MODEL_CONFIG:-./pretrained/edm/edm_geom_drugs/args.pickle}"
MODEL_WEIGHTS="${MODEL_WEIGHTS:-./pretrained/edm/edm_geom_drugs/generative_model_ema.npy}"

# GEOM data (.npy). Prefer passing an absolute path via `GEOM_DATA_FILE=...`.
GEOM_DATA_FILE="${GEOM_DATA_FILE:-}"

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

if [[ -z "${GEOM_DATA_FILE}" ]]; then
  if [[ -f "${REPO_ROOT}/datasets/geom/geom_drugs_30.npy" ]]; then
    GEOM_DATA_FILE="${REPO_ROOT}/datasets/geom/geom_drugs_30.npy"
  elif [[ -f "${REPO_ROOT}/data/geom/geom_drugs_30.npy" ]]; then
    GEOM_DATA_FILE="${REPO_ROOT}/data/geom/geom_drugs_30.npy"
  elif [[ -f "${REPO_ROOT}/edm_source/data/geom/geom_drugs_30.npy" ]]; then
    GEOM_DATA_FILE="${REPO_ROOT}/edm_source/data/geom/geom_drugs_30.npy"
  else
    echo "ERROR: GEOM conformation file not found."
    echo "Set GEOM_DATA_FILE=/path/to/geom_drugs_30.npy (recommended)."
    exit 1
  fi
fi

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
TRAIN_MICRO_BATCH_SIZE="${TRAIN_MICRO_BATCH_SIZE:-4}"
EPOCH_PER_ROLLOUT="${EPOCH_PER_ROLLOUT:-1}"
KL_PENALTY_WEIGHT="${KL_PENALTY_WEIGHT:-0.04}"

# ----------------------------
# Diffusion rollout settings
# ----------------------------
SAMPLE_GROUP_SIZE="${SAMPLE_GROUP_SIZE:-1}"
EACH_PROMPT_SAMPLE="${EACH_PROMPT_SAMPLE:-24}"
TIME_STEP="${TIME_STEP:-1000}"
SHARE_INITIAL_NOISE="${SHARE_INITIAL_NOISE:-true}"
FORCE_ALIGNMENT_ENABLED="${FORCE_ALIGNMENT_ENABLED:-false}"

# ----------------------------
# Reward configuration
# ----------------------------
USE_ENERGY="${USE_ENERGY:-true}"
MLFF_MODEL="${MLFF_MODEL:-uma-m-1p1}"
MLFF_BATCH_SIZE="${MLFF_BATCH_SIZE:-8}"
FORCE_AGGREGATION="${FORCE_AGGREGATION:-rms}"
STABILITY_WEIGHT="${STABILITY_WEIGHT:-0}"

# Shaping settings
SKIP_PREFIX="${SKIP_PREFIX:-700}"
REWARD_SHAPING_ENABLED="${REWARD_SHAPING_ENABLED:-true}"
SHAPING_ONLY_ENERGY="${SHAPING_ONLY_ENERGY:-true}"
TERMINAL_WEIGHT="${TERMINAL_WEIGHT:-5.0}"

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
RUN_NAME="verl_geom_${MODEL_TAG}_lr_$(sanitize_for_name "${LEARNING_RATE}")_${timestamp}"

SAVE_ROOT="${SAVE_ROOT:-${REPO_ROOT}/outputs/verl_geom}"
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

EXTRA_FLAGS=()
if [[ -n "${EPOCHES}" ]]; then
  EXTRA_FLAGS+=("dataloader.epoches=${EPOCHES}")
fi

if [[ "${USE_TORCHRUN}" == "1" ]]; then
  LAUNCHER=(torchrun --standalone --nproc_per_node="${GPUS_PER_NODE}")
else
  LAUNCHER=(python -u)
fi

"${LAUNCHER[@]}" run_verl_diffusion.py \
  --config-name "${CONFIG_NAME}" \
  "${WANDB_FLAGS[@]}" \
  save_path="${SAVE_PATH}" \
  model.config="${MODEL_CONFIG}" \
  model.model_path="${MODEL_WEIGHTS}" \
  dataloader.geom_data_file="${GEOM_DATA_FILE}" \
  reward.type="${REWARD_TYPE}" \
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
  reward.stability_weight="${STABILITY_WEIGHT}" \
  train.scheduler.name="${SCHEDULER_NAME}" \
  train.scheduler.warmup_steps="${SCHEDULER_WARMUP_STEPS}" \
  train.scheduler.total_steps="${SCHEDULER_TOTAL_STEPS}" \
  train.scheduler.min_lr_ratio="${SCHEDULER_MIN_LR_RATIO}" \
  "${EXTRA_FLAGS[@]}" \
  "${SHAPING_FLAGS[@]}"
