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

module load miniconda
eval "$(conda shell.bash hook)"
conda activate edm

RUN_NAME="test_reward_shaping_lr_0.00001_clip_range_0.15_not_share_initial_noise_skip_prefix_50_force_alignment_disabled_false_time_step_100"
SAVE_ROOT="/home/yl2428/scratch_pi_mg269/yl2428/logs"
SAVE_PATH="${SAVE_ROOT}/${RUN_NAME}"

mkdir -p "${SAVE_PATH}"

python run_verl_diffusion.py \
  wandb.enabled=true \
  wandb.wandb_name="${RUN_NAME}" \
  save_path="${SAVE_PATH}" \
  train.learning_rate=0.00001 \
  train.clip_range=0.15 \
  model.share_initial_noise=false \
  reward.shaping.scheduler.skip_prefix=50 \
  train.force_alignment_enabled=false \
  model.time_step=100
