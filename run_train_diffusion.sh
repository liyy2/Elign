#!/bin/bash
#SBATCH --job-name=exp_cond_alpha
#SBATCH --output=exp_cond_alpha_%j.out
#SBATCH --error=exp_cond_alpha_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=48:00:00
#SBATCH --gres=gpu:4
#SBATCH --partition=gpu



module load miniconda
eval "$(conda shell.bash hook)"
conda activate edm
cd /home/yl2428/project_pi_mg269/yl2428/e3_diffusion_for_molecules-main/e3_diffusion_for_molecules-main
wandb login 6f1080f993d5d7ad6103e69ef57dd9291f1bf366
python main_qm9.py \
  --exp_name exp_cond_alpha \
  --model egnn_dynamics \
  --lr 1e-4 \
  --nf 192 \
  --n_layers 9 \
  --save_model True \
  --diffusion_steps 1000 \
  --sin_embedding False \
  --n_epochs 3000 \
  --n_stability_samples 500 \
  --diffusion_noise_schedule polynomial_2 \
  --diffusion_noise_precision 1e-5 \
  --dequantization deterministic \
  --include_charges False \
  --diffusion_loss_type l2 \
  --batch_size 64 \
  --normalize_factors '[1,8,1]' \
  --conditioning alpha \
  --dataset qm9_second_half
