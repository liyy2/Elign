#!/bin/bash
#SBATCH --job-name=exp_cond_alpha
#SBATCH --output=exp_cond_alpha_%j.out
#SBATCH --error=exp_cond_alpha_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu_h200
#SBATCH --mail-type=ALL
#SBATCH --mail-user=yunyang.li@yale.edu

module load miniconda
eval "$(conda shell.bash hook)"
conda activate edm

sanitize_for_name() {
  local value="$1"
  value="${value//[^a-zA-Z0-9]/_}"
  value="${value##_}"
  value="${value%%_}"
  echo "${value}"
}

# Optional resume from a DDPO checkpoint.
# NOTE: DDPOTrainer only loads `checkpoint_path` when `resume=true`.
CHECKPOINT_PATH="${CHECKPOINT_PATH:-}"
declare -a RESUME_FLAGS=()
if [[ -n "${CHECKPOINT_PATH}" ]]; then
  RESUME_FLAGS=("resume=true" "checkpoint_path=${CHECKPOINT_PATH}")
fi

# Optimization / training loop (defaults match `ddpo_qm9_energy_force_group4x6.yaml`)
LEARNING_RATE="4e-6"
CLIP_RANGE="2e-3"
TRAIN_MICRO_BATCH_SIZE=8 # PPO micro-batch size during the update (reduce if OOM)
EPOCH_PER_ROLLOUT=1
KL_PENALTY_WEIGHT=0.08
# Diffusion rollout settings
SAMPLE_GROUP_SIZE=4 # Number of prompts per batch
EACH_PROMPT_SAMPLE=6 # Samples per prompt (group size); total rollouts = 24
TIME_STEP=1000
SHARE_INITIAL_NOISE=true
FORCE_ALIGNMENT_ENABLED=false

# Reward configuration
USE_ENERGY=true
MLFF_MODEL="uma-s-1p1"
MLFF_BATCH_SIZE=32 # Only used when shaping is enabled
FORCE_AGGREGATION="rms"
FORCE_CLIP_THRESHOLD="2.0"
STABILITY_WEIGHT="2.0"
# Energy reward is easy to exploit on physically invalid geometries. Gate it on stability by default.
ENERGY_ONLY_IF_STABLE=true
ENERGY_ADV_WEIGHT="0.05"
SKIP_PREFIX=700
REWARD_SHAPING_ENABLED=false # Prefer terminal-only rewards; shaping is noisier
# When shaping is enabled, `true` reshapes only energy deltas (force only appears at terminal).
# For stability, prefer `false` so both force + energy deltas contribute to advantages.
SHAPING_ONLY_ENERGY=false

# Note: when shaping is enabled, PPO advantages are computed from per-step
# force/energy traces, so scalar rewards (and any novelty penalty applied to them)
# only affect logs unless the penalty is injected into those traces.

# Filtering configuration
ENABLE_FILTERING=true
ENABLE_NOVELTY_PENALTY=false
NOVELTY_PENALTY_SCALE="0.5"
# Optional: penalize RDKit-invalid molecules only (without pushing away from QM9).
INVALID_PENALTY_SCALE="2.0"
# Optional: penalize duplicate SMILES within a rollout batch (anti-collapse).
DUPLICATE_PENALTY_SCALE="0.0"

# Scheduler configuration
SCHEDULER_NAME="cosine"
SCHEDULER_WARMUP_STEPS=60
SCHEDULER_TOTAL_STEPS=1500 # total scheduler steps before decay completes
SCHEDULER_MIN_LR_RATIO="0.3" # final learning rate as a fraction of the initial LR

# Each dataloader batch emits SAMPLE_GROUP_SIZE * EACH_PROMPT_SAMPLE trajectories per GPU.
# PPO consumes the same batch as TRAIN_MICRO_BATCH_SIZE-sized chunks (per GPU).
# Set TRAIN_MICRO_BATCH_SIZE < SAMPLE_GROUP_SIZE * EACH_PROMPT_SAMPLE to avoid OOM.

# Compose run / logging names
timestamp=$(date +"%Y%m%d_%H%M%S")

if [[ "${USE_ENERGY}" == true ]]; then
  ENERGY_TAG="energy"
else
  ENERGY_TAG="no_energy"
fi
MODEL_TAG=$(sanitize_for_name "${MLFF_MODEL}")
NOVELTY_PENALTY_TAG=$(sanitize_for_name "${ENABLE_NOVELTY_PENALTY}")
NOVELTY_PENALTY_SCALE_TAG=$(sanitize_for_name "${NOVELTY_PENALTY_SCALE}")

RUN_NAME_BASE="verl_model_${MODEL_TAG}\
_energy_$(sanitize_for_name "${ENERGY_TAG}")\
_lr_$(sanitize_for_name "${LEARNING_RATE}")\
_pps_$(sanitize_for_name "${EACH_PROMPT_SAMPLE}")\
_skip_$(sanitize_for_name "${SKIP_PREFIX}")\
_klw_$(sanitize_for_name "${KL_PENALTY_WEIGHT}")\
_force_align_$(sanitize_for_name "${FORCE_ALIGNMENT_ENABLED}")\
_sg_$(sanitize_for_name "${SAMPLE_GROUP_SIZE}")\
_shaping_$(sanitize_for_name "${REWARD_SHAPING_ENABLED}")\
_shape_energy_only_$(sanitize_for_name "${SHAPING_ONLY_ENERGY}")\
_novpen_${NOVELTY_PENALTY_TAG}\
_novpen_scale_${NOVELTY_PENALTY_SCALE_TAG}\
_epoch_per_rollout_$(sanitize_for_name "${EPOCH_PER_ROLLOUT}")"

RUN_NAME="${RUN_NAME_BASE}_${timestamp}"
SAVE_ROOT="/home/yl2428/logs"
SAVE_PATH="${SAVE_ROOT}/${RUN_NAME}"

mkdir -p "${SAVE_PATH}"

GPUS_PER_NODE=1

export MASTER_ADDR=${MASTER_ADDR:-$(hostname)}
export MASTER_PORT=${MASTER_PORT:-29500}

declare -a SHAPING_FLAGS
if [[ "${REWARD_SHAPING_ENABLED}" == true ]]; then
  SHAPING_FLAGS=(
    "reward.shaping.enabled=true"
    "reward.shaping.scheduler.skip_prefix=${SKIP_PREFIX}"
    "reward.shaping.only_energy_reshape=${SHAPING_ONLY_ENERGY}"
  )
else
  SHAPING_FLAGS=("reward.shaping.enabled=false")
fi

# Launch training
torchrun --standalone --nproc_per_node="${GPUS_PER_NODE}" run_verl_diffusion.py \
  --config-name ddpo_qm9_energy_force_group4x6 \
  wandb.enabled=true \
  wandb.wandb_name="${RUN_NAME}" \
  save_path="${SAVE_PATH}" \
  "${RESUME_FLAGS[@]}" \
  train.learning_rate="${LEARNING_RATE}" \
  train.clip_range="${CLIP_RANGE}" \
  train.kl_penalty_weight="${KL_PENALTY_WEIGHT}" \
  model.share_initial_noise="${SHARE_INITIAL_NOISE}" \
  dataloader.each_prompt_sample="${EACH_PROMPT_SAMPLE}" \
  train.force_alignment_enabled="${FORCE_ALIGNMENT_ENABLED}" \
  train.train_micro_batch_size="${TRAIN_MICRO_BATCH_SIZE}" \
  train.epoch_per_rollout="${EPOCH_PER_ROLLOUT}" \
  dataloader.sample_group_size="${SAMPLE_GROUP_SIZE}" \
  model.time_step="${TIME_STEP}" \
  reward.shaping.mlff_batch_size="${MLFF_BATCH_SIZE}" \
  reward.force_aggregation="${FORCE_AGGREGATION}" \
  reward.force_clip_threshold="${FORCE_CLIP_THRESHOLD}" \
  reward.stability_weight="${STABILITY_WEIGHT}" \
  reward.energy_only_if_stable="${ENERGY_ONLY_IF_STABLE}" \
  reward.energy_adv_weight="${ENERGY_ADV_WEIGHT}" \
  train.scheduler.name="${SCHEDULER_NAME}" \
  train.scheduler.warmup_steps="${SCHEDULER_WARMUP_STEPS}" \
  train.scheduler.total_steps="${SCHEDULER_TOTAL_STEPS}" \
  train.scheduler.min_lr_ratio="${SCHEDULER_MIN_LR_RATIO}" \
  filters.enable_filtering="${ENABLE_FILTERING}" \
  reward.use_energy="${USE_ENERGY}" \
  reward.mlff_model="${MLFF_MODEL}" \
  filters.enable_penalty="${ENABLE_NOVELTY_PENALTY}" \
  filters.penalty_scale="${NOVELTY_PENALTY_SCALE}" \
  filters.invalid_penalty_scale="${INVALID_PENALTY_SCALE}" \
  filters.duplicate_penalty_scale="${DUPLICATE_PENALTY_SCALE}" \
  "${SHAPING_FLAGS[@]}"
  
