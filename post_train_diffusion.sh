#!/bin/bash
#SBATCH --job-name=exp_cond_alpha
#SBATCH --output=exp_cond_alpha_%j.out
#SBATCH --error=exp_cond_alpha_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --gres=gpu:4
#SBATCH --partition=gpu
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

# Optimization / training loop
LEARNING_RATE="1e-5"
CLIP_RANGE="5e-3"
TRAIN_MICRO_BATCH_SIZE=2 # batch size for doing policy gradient, reduce if the training part is a bottleneck
EPOCH_PER_ROLLOUT=1

# Diffusion rollout settings
SAMPLE_GROUP_SIZE=1 # Number of groups
EACH_PROMPT_SAMPLE=12 # Group Size in GRPO
TIME_STEP=1000
SHARE_INITIAL_NOISE=true
FORCE_ALIGNMENT_ENABLED=false

# Reward configuration
USE_ENERGY=false
MLFF_MODEL="uma-m-1p1"
MLFF_BATCH_SIZE=16 # Batch size for calculating reward, reduce if the reward calculation is a bottleneck
FORCE_AGGREGATION="rms"
STABILITY_WEIGHT="1"
SKIP_PREFIX=700

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

RUN_NAME_BASE="verl_model_${MODEL_TAG}\
_energy_$(sanitize_for_name "${ENERGY_TAG}")\
_lr_$(sanitize_for_name "${LEARNING_RATE}")\
_clip_$(sanitize_for_name "${CLIP_RANGE}")\
_share_noise_$(sanitize_for_name "${SHARE_INITIAL_NOISE}")\
_pps_$(sanitize_for_name "${EACH_PROMPT_SAMPLE}")\
_skip_$(sanitize_for_name "${SKIP_PREFIX}")\
_force_align_$(sanitize_for_name "${FORCE_ALIGNMENT_ENABLED}")\
_tstep_$(sanitize_for_name "${TIME_STEP}")\
_sg_$(sanitize_for_name "${SAMPLE_GROUP_SIZE}")\
_mlff_$(sanitize_for_name "${MLFF_BATCH_SIZE}")\
_fagg_$(sanitize_for_name "${FORCE_AGGREGATION}")\
_stabW_$(sanitize_for_name "${STABILITY_WEIGHT}")\
_sched_$(sanitize_for_name "${SCHEDULER_NAME}")\
_warmup_$(sanitize_for_name "${SCHEDULER_WARMUP_STEPS}")\
_steps_$(sanitize_for_name "${SCHEDULER_TOTAL_STEPS}")\
_decay_$(sanitize_for_name "${SCHEDULER_MIN_LR_RATIO}")\
_epoch_per_rollout_$(sanitize_for_name "${EPOCH_PER_ROLLOUT}")"

RUN_NAME="${RUN_NAME_BASE}_${timestamp}"
SAVE_ROOT="/home/yl2428/project_pi_mg269/yl2428/logs"
SAVE_PATH="${SAVE_ROOT}/${RUN_NAME}"

mkdir -p "${SAVE_PATH}"

GPUS_PER_NODE=1

export MASTER_ADDR=${MASTER_ADDR:-$(hostname)}
export MASTER_PORT=${MASTER_PORT:-29500}

# Launch training
torchrun --standalone --nproc_per_node="${GPUS_PER_NODE}" run_verl_diffusion.py \
  wandb.enabled=true \
  wandb.wandb_name="${RUN_NAME}" \
  save_path="${SAVE_PATH}" \
  train.learning_rate="${LEARNING_RATE}" \
  train.clip_range="${CLIP_RANGE}" \
  model.share_initial_noise="${SHARE_INITIAL_NOISE}" \
  dataloader.each_prompt_sample="${EACH_PROMPT_SAMPLE}" \
  reward.shaping.scheduler.skip_prefix="${SKIP_PREFIX}" \
  train.force_alignment_enabled="${FORCE_ALIGNMENT_ENABLED}" \
  train.train_micro_batch_size="${TRAIN_MICRO_BATCH_SIZE}" \
  train.epoch_per_rollout="${EPOCH_PER_ROLLOUT}" \
  dataloader.sample_group_size="${SAMPLE_GROUP_SIZE}" \
  model.time_step="${TIME_STEP}" \
  reward.shaping.mlff_batch_size="${MLFF_BATCH_SIZE}" \
  reward.force_aggregation="${FORCE_AGGREGATION}" \
  reward.stability_weight="${STABILITY_WEIGHT}" \
  train.scheduler.name="${SCHEDULER_NAME}" \
  train.scheduler.warmup_steps="${SCHEDULER_WARMUP_STEPS}" \
  train.scheduler.total_steps="${SCHEDULER_TOTAL_STEPS}" \
  train.scheduler.min_lr_ratio="${SCHEDULER_MIN_LR_RATIO}" \
  reward.use_energy="${USE_ENERGY}" \
  reward.mlff_model="${MLFF_MODEL}"
  
