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

LEARNING_RATE="0.000005"
CLIP_RANGE="0.2"
SHARE_INITIAL_NOISE=true
EACH_PROMPT_SAMPLE=128 # Group Size in GRPO
SKIP_PREFIX=50
FORCE_ALIGNMENT_ENABLED=false
SAMPLE_GROUP_SIZE=4 # Number of groups
TIME_STEP=100
MLFF_BATCH_SIZE=32 # Batch size for calculating reward, reduce if the reward calculation is a bottleneck
FORCE_AGGREGATION="max"
TRAIN_MICRO_BATCH_SIZE=128 # batch size for doing policy gradient, reduce if the training part is a bottleneck

# Each dataloader batch emits SAMPLE_GROUP_SIZE * EACH_PROMPT_SAMPLE trajectories per GPU.
# PPO consumes the same batch as TRAIN_MICRO_BATCH_SIZE-sized chunks (per GPU).
# Set TRAIN_MICRO_BATCH_SIZE < SAMPLE_GROUP_SIZE * EACH_PROMPT_SAMPLE to avoid OOM.

timestamp=$(date +"%Y%m%d_%H%M%S")

sanitize_for_name() {
  local value="$1"
  value="${value//[^a-zA-Z0-9]/_}"
  value="${value##_}"
  value="${value%%_}"
  echo "${value}"
}

RUN_NAME_BASE="verl_lr_$(sanitize_for_name "${LEARNING_RATE}")\
_clip_$(sanitize_for_name "${CLIP_RANGE}")\
_share_noise_$(sanitize_for_name "${SHARE_INITIAL_NOISE}")\
_pps_$(sanitize_for_name "${EACH_PROMPT_SAMPLE}")\
_skip_$(sanitize_for_name "${SKIP_PREFIX}")\
_force_align_$(sanitize_for_name "${FORCE_ALIGNMENT_ENABLED}")\
_tstep_$(sanitize_for_name "${TIME_STEP}")\
_sg_$(sanitize_for_name "${SAMPLE_GROUP_SIZE}")\
_mlff_$(sanitize_for_name "${MLFF_BATCH_SIZE}")\
_fagg_$(sanitize_for_name "${FORCE_AGGREGATION}")"

RUN_NAME="${RUN_NAME_BASE}_${timestamp}"
SAVE_ROOT="/home/yl2428/scratch_pi_mg269/yl2428/logs"
SAVE_PATH="${SAVE_ROOT}/${RUN_NAME}"

mkdir -p "${SAVE_PATH}"

GPUS_PER_NODE=4

export MASTER_ADDR=${MASTER_ADDR:-$(hostname)}
export MASTER_PORT=${MASTER_PORT:-29500}

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
  dataloader.sample_group_size="${SAMPLE_GROUP_SIZE}" \
  model.time_step="${TIME_STEP}" \
  reward.shaping.mlff_batch_size="${MLFF_BATCH_SIZE}" \
  reward.force_aggregation="${FORCE_AGGREGATION}"
  
