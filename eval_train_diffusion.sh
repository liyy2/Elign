#!/bin/bash
#SBATCH --job-name=exp_cond_alpha
#SBATCH --output=exp_cond_alpha_%j.out
#SBATCH --error=exp_cond_alpha_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=48:00:00
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu_h200

module load miniconda
eval "$(conda shell.bash hook)"
conda activate edm
cd /home/yl2428/project_pi_mg269/yl2428/e3_diffusion_for_molecules-main/edm_source
wandb login 6f1080f993d5d7ad6103e69ef57dd9291f1bf366
huggingface-cli login --token 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
nvidia-smi
python -m torch.distributed.launch --nproc_per_node=1 eval_mlff_guided.py \
    --model_path outputs/exp_cond_alpha --use_distributed \
    --n_samples 1000 --guidance_iterations 1 \
    --use_wandb --guidance_scales 0.002 0.004 \
    --skip_visualization --skip_analysis --skip_chain \
    --noise_threshold 0.3 \
    --force_clip_threshold 10 --sampling_method ddpm --dpm_solver_order 2 --dpm_solver_steps 20 --batch_size 25
