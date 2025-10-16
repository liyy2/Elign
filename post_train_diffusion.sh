#!/bin/bash
#SBATCH --job-name=exp_cond_alpha
#SBATCH --output=exp_cond_alpha_%j.out
#SBATCH --error=exp_cond_alpha_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu_h200
#SBATCH --mail-type=ALL
#SBATCH --mail-user=yunyang.li@yale.edu

module load miniconda
eval "$(conda shell.bash hook)"
conda activate edm

LEARNING_RATE="0.000005"
CLIP_RANGE="0.2"
SHARE_INITIAL_NOISE=true
EACH_PROMPT_SAMPLE=128
SKIP_PREFIX=500
FORCE_ALIGNMENT_ENABLED=false
SAMPLE_GROUP_SIZE=16
TIME_STEP=1000
MLFF_BATCH_SIZE=32

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
_mlff_$(sanitize_for_name "${MLFF_BATCH_SIZE}")"

RUN_NAME="${RUN_NAME_BASE}_${timestamp}"
SAVE_ROOT="/home/yl2428/scratch_pi_mg269/yl2428/logs"
SAVE_PATH="${SAVE_ROOT}/${RUN_NAME}"

mkdir -p "${SAVE_PATH}"

python run_verl_diffusion.py \
  wandb.enabled=true \
  wandb.wandb_name="${RUN_NAME}" \
  save_path="${SAVE_PATH}" \
  train.learning_rate="${LEARNING_RATE}" \
  train.clip_range="${CLIP_RANGE}" \
  model.share_initial_noise="${SHARE_INITIAL_NOISE}" \
  dataloader.each_prompt_sample="${EACH_PROMPT_SAMPLE}" \
  reward.shaping.scheduler.skip_prefix="${SKIP_PREFIX}" \
  train.force_alignment_enabled="${FORCE_ALIGNMENT_ENABLED}" \
  dataloader.sample_group_size="${SAMPLE_GROUP_SIZE}" \
  model.time_step="${TIME_STEP}" \
  reward.shaping.mlff_batch_size="${MLFF_BATCH_SIZE}"
  
