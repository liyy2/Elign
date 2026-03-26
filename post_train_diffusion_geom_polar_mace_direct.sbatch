#!/bin/bash
#SBATCH --job-name=geom_polar_direct
#SBATCH --output=/gpfs/radev/home/yl2428/e3_diffusion_for_molecules-main/slurm_logs/%x_%j.out
#SBATCH --error=/gpfs/radev/home/yl2428/e3_diffusion_for_molecules-main/slurm_logs/%x_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu

set -euo pipefail

REPO_ROOT="/gpfs/radev/home/yl2428/e3_diffusion_for_molecules-main"
cd "${REPO_ROOT}"

mkdir -p /gpfs/radev/home/yl2428/e3_diffusion_for_molecules-main/slurm_logs

module load miniconda
eval "$(conda shell.bash hook)"
conda activate edm

timestamp=$(date +"%Y%m%d_%H%M%S")

RUN_NAME="${RUN_NAME_OVERRIDE:-geom_pretrained_tunedonly_polar_1_m_lowvar_seed45_scratch_direct_slurm_${timestamp}}"
SAVE_PATH="${SAVE_PATH_OVERRIDE:-${REPO_ROOT}/outputs/verl_geom/${RUN_NAME}}"
WANDB_PROJECT="${WANDB_PROJECT_OVERRIDE:-ddpo}"
WANDB_ENABLED="${WANDB_ENABLED_OVERRIDE:-true}"

mkdir -p "${SAVE_PATH}"

echo "RUN_NAME=${RUN_NAME}"
echo "SAVE_PATH=${SAVE_PATH}"
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-}"
echo "HOSTNAME=$(hostname)"

/gpfs/radev/home/yl2428/.conda/envs/edm/bin/python -u "${REPO_ROOT}/run_verl_diffusion.py" \
  --config-name ddpo_geom_local_lowvar_rms_bestonly \
  wandb.enabled="${WANDB_ENABLED}" \
  wandb.wandb_project="${WANDB_PROJECT}" \
  wandb.wandb_name="${RUN_NAME}" \
  save_path="${SAVE_PATH}" \
  seed=45 \
  model.config="${REPO_ROOT}/pretrained/edm/edm_geom_drugs/args.pickle" \
  model.model_path="${REPO_ROOT}/pretrained/edm/edm_geom_drugs/generative_model_ema.npy" \
  dataloader.geom_data_file="${REPO_ROOT}/geom_drugs_30.npy" \
  dataloader.micro_batch_size=32 \
  reward.type=polar_mace \
  reward.mlff_backend=polar_mace \
  train.learning_rate=1e-7 \
  train.clip_range=0.03 \
  train.adv_clip_max=2 \
  train.max_grad_norm=0.5 \
  train.kl_penalty_weight=0.2 \
  train.kl_adaptive=false \
  train.train_micro_batch_size=16 \
  train.gradient_accumulation_steps=8 \
  train.epoch_per_rollout=1 \
  model.time_step=1000 \
  model.share_initial_noise=true \
  model.skip_prefix=700 \
  dataloader.sample_group_size=4 \
  dataloader.each_prompt_sample=32 \
  train.force_alignment_enabled=true \
  train.force_alignment_weight=0.05 \
  train.force_alignment_min_force=0.0001 \
  train.force_alignment_min_delta=0.0001 \
  reward.use_energy=true \
  reward.force_weight=1.0 \
  reward.energy_weight=0.05 \
  reward.energy_adv_weight=0.05 \
  reward.mlff_model=polar-1-m \
  reward.mlff_default_dtype=float32 \
  reward.shaping.mlff_batch_size=16 \
  reward.force_aggregation=rms \
  reward.stability_weight=1.0 \
  reward.atom_stability_weight=5.0 \
  reward.valence_underbond_weight=0.20 \
  reward.valence_overbond_weight=0.50 \
  reward.valence_underbond_soft_weight=0.05 \
  reward.valence_overbond_soft_weight=0.15 \
  reward.valence_soft_temperature=0.02 \
  reward.force_clip_threshold=null \
  reward.adv_schedule.enabled=true \
  reward.adv_schedule.energy.start=0.02 \
  reward.adv_schedule.energy.end=0.05 \
  reward.adv_schedule.energy.warmup_epochs=0 \
  reward.adv_schedule.energy.ramp_epochs=20 \
  train.scheduler.name=none \
  train.scheduler.warmup_steps=0 \
  train.scheduler.total_steps=0 \
  train.scheduler.min_lr_ratio=1.0 \
  2>&1 | tee -a "${SAVE_PATH}/launch.log"
