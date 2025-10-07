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
python run_verl_diffusion.py \
  wandb.enabled=true \
  wandb.wandb_name=test_reward_shaping_lr_0.000005_clip_range_0.15 \
  train.learning_rate=0.000005 \
  train.clip_range=0.15
