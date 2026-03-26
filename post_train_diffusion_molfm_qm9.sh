#!/bin/bash
#SBATCH --job-name=post_train_molfm_qm9
#SBATCH --output=post_train_molfm_qm9_%j.out
#SBATCH --error=post_train_molfm_qm9_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu_h200

set -euo pipefail

# MolFM QM9 post-training entrypoint.
#
# This launcher uses the vendored public MolFM QM9 sampling bundle under:
#   pretrained/molfm/qm9/args.pickle
#   pretrained/molfm/qm9/generative_model_ema_0.npy
#
# Unlike the upstream MolFM repo, this script does not require Docker. It runs
# directly in the local `edm` environment and uses the fixed-step SDE wrapper
# added for PPO/DDPO compatibility.

CONDA_ENV="${CONDA_ENV:-edm}"
WANDB_ENABLED="${WANDB_ENABLED:-0}"
WANDB_PROJECT="${WANDB_PROJECT:-ddpo}"

CONFIG_NAME="${CONFIG_NAME:-ddpo_molfm_qm9_sde}"
MODEL_CONFIG="${MODEL_CONFIG:-./pretrained/molfm/qm9/args.pickle}"
MODEL_WEIGHTS="${MODEL_WEIGHTS:-./pretrained/molfm/qm9/generative_model_ema_0.npy}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-}"
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

LEARNING_RATE="${LEARNING_RATE:-4e-6}"
CLIP_RANGE="${CLIP_RANGE:-2e-3}"
TRAIN_MICRO_BATCH_SIZE="${TRAIN_MICRO_BATCH_SIZE:-8}"
EPOCH_PER_ROLLOUT="${EPOCH_PER_ROLLOUT:-1}"
KL_PENALTY_WEIGHT="${KL_PENALTY_WEIGHT:-0.08}"

SAMPLE_GROUP_SIZE="${SAMPLE_GROUP_SIZE:-4}"
EACH_PROMPT_SAMPLE="${EACH_PROMPT_SAMPLE:-6}"
TIME_STEP="${TIME_STEP:-256}"
SHARE_INITIAL_NOISE="${SHARE_INITIAL_NOISE:-true}"
RETURN_SUFFIX_ONLY="${RETURN_SUFFIX_ONLY:-true}"
SKIP_PREFIX="${SKIP_PREFIX:-192}"
POLICY_START_IDX="${POLICY_START_IDX:-192}"
SDE_WINDOW_SIZE="${SDE_WINDOW_SIZE:-4}"
SDE_MODE="${SDE_MODE:-sigma_corrected_hb}"

SDE_NOISE_SCALE="${SDE_NOISE_SCALE:-0.35}"
SDE_COORDINATE_NOISE_SCALE="${SDE_COORDINATE_NOISE_SCALE:-0.35}"
SDE_FEATURE_NOISE_SCALE="${SDE_FEATURE_NOISE_SCALE:-0.05}"
SDE_MIN_SIGMA="${SDE_MIN_SIGMA:-1e-6}"

FORCE_ALIGNMENT_ENABLED="${FORCE_ALIGNMENT_ENABLED:-false}"

USE_ENERGY="${USE_ENERGY:-true}"
REWARD_TYPE="${REWARD_TYPE:-polar_mace}"  # polar_mace | uma | dummy
MLFF_BACKEND="${MLFF_BACKEND:-$REWARD_TYPE}"  # polar_mace | uma
MLFF_MODEL="${MLFF_MODEL:-polar-1-m}"
MLFF_BATCH_SIZE="${MLFF_BATCH_SIZE:-32}"
FORCE_AGGREGATION="${FORCE_AGGREGATION:-rms}"
FORCE_CLIP_THRESHOLD="${FORCE_CLIP_THRESHOLD:-2.0}"
STABILITY_WEIGHT="${STABILITY_WEIGHT:-2.0}"
ENERGY_ONLY_IF_STABLE="${ENERGY_ONLY_IF_STABLE:-true}"
ENERGY_ADV_WEIGHT="${ENERGY_ADV_WEIGHT:-0.05}"

REWARD_SHAPING_ENABLED="${REWARD_SHAPING_ENABLED:-false}"
SHAPING_ONLY_ENERGY="${SHAPING_ONLY_ENERGY:-false}"
TERMINAL_WEIGHT="${TERMINAL_WEIGHT:-5.0}"

ENABLE_NOVELTY_PENALTY="${ENABLE_NOVELTY_PENALTY:-false}"
NOVELTY_PENALTY_SCALE="${NOVELTY_PENALTY_SCALE:-0.5}"
ENABLE_FILTERING="${ENABLE_FILTERING:-false}"
INVALID_PENALTY_SCALE="${INVALID_PENALTY_SCALE:-2.0}"
DUPLICATE_PENALTY_SCALE="${DUPLICATE_PENALTY_SCALE:-0.0}"

SCHEDULER_NAME="${SCHEDULER_NAME:-cosine}"
SCHEDULER_WARMUP_STEPS="${SCHEDULER_WARMUP_STEPS:-60}"
SCHEDULER_TOTAL_STEPS="${SCHEDULER_TOTAL_STEPS:-1500}"
SCHEDULER_MIN_LR_RATIO="${SCHEDULER_MIN_LR_RATIO:-0.3}"

timestamp=$(date +"%Y%m%d_%H%M%S")
RUN_NAME="verl_molfm_qm9_ts_$(sanitize_for_name "${TIME_STEP}")_lr_$(sanitize_for_name "${LEARNING_RATE}")_sde_$(sanitize_for_name "${SDE_NOISE_SCALE}")_${timestamp}"

SAVE_ROOT="${SAVE_ROOT:-${REPO_ROOT}/outputs/verl}"
SAVE_PATH="${SAVE_ROOT}/${RUN_NAME}"
mkdir -p "${SAVE_PATH}"

GPUS_PER_NODE="${GPUS_PER_NODE:-1}"
USE_TORCHRUN="${USE_TORCHRUN:-1}"

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

if [[ "${USE_TORCHRUN}" == "1" ]]; then
  LAUNCHER=(torchrun --standalone --nproc_per_node="${GPUS_PER_NODE}")
else
  LAUNCHER=(python -u)
fi

"${LAUNCHER[@]}" run_verl_diffusion.py \
  --config-name "${CONFIG_NAME}" \
  "${WANDB_FLAGS[@]}" \
  save_path="${SAVE_PATH}" \
  "${RESUME_FLAGS[@]}" \
  model.backend="molfm" \
  model.config="${MODEL_CONFIG}" \
  model.model_path="${MODEL_WEIGHTS}" \
  model.time_step="${TIME_STEP}" \
  model.share_initial_noise="${SHARE_INITIAL_NOISE}" \
  model.return_suffix_only="${RETURN_SUFFIX_ONLY}" \
  model.skip_prefix="${SKIP_PREFIX}" \
  model.policy_start_idx="${POLICY_START_IDX}" \
  model.sde_window_size="${SDE_WINDOW_SIZE}" \
  model.sde_mode="${SDE_MODE}" \
  model.sde_noise_scale="${SDE_NOISE_SCALE}" \
  model.sde_coordinate_noise_scale="${SDE_COORDINATE_NOISE_SCALE}" \
  model.sde_feature_noise_scale="${SDE_FEATURE_NOISE_SCALE}" \
  model.sde_min_sigma="${SDE_MIN_SIGMA}" \
  dataloader.smiles_path="${SMILES_PATH}" \
  dataloader.sample_group_size="${SAMPLE_GROUP_SIZE}" \
  dataloader.each_prompt_sample="${EACH_PROMPT_SAMPLE}" \
  train.learning_rate="${LEARNING_RATE}" \
  train.clip_range="${CLIP_RANGE}" \
  train.kl_penalty_weight="${KL_PENALTY_WEIGHT}" \
  train.train_micro_batch_size="${TRAIN_MICRO_BATCH_SIZE}" \
  train.epoch_per_rollout="${EPOCH_PER_ROLLOUT}" \
  train.force_alignment_enabled="${FORCE_ALIGNMENT_ENABLED}" \
  reward.use_energy="${USE_ENERGY}" \
  reward.type="${REWARD_TYPE}" \
  reward.mlff_backend="${MLFF_BACKEND}" \
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
