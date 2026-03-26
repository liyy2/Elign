#!/bin/bash
#SBATCH --job-name=ala2_pretrain
#SBATCH --output=ala2_pretrain_%j.out
#SBATCH --error=ala2_pretrain_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu

set -euo pipefail

module load miniconda
eval "$(conda shell.bash hook)"
conda activate edm

REPO_ROOT="/home/yl2428/e3_diffusion_for_molecules-main"
cd "${REPO_ROOT}"

CONFIG_PATH="${CONFIG_PATH:-ala2_scratch/configs/ala2_base.json}"
PROCESSED_DATA="${PROCESSED_DATA:-data/ala2/ala2_300k.npz}"
PHI_INDICES="${PHI_INDICES:-4,6,8,14}"
PSI_INDICES="${PSI_INDICES:-6,8,14,16}"
PRETRAIN_ARGS="${PRETRAIN_ARGS:-}"

echo "JOBID=${SLURM_JOB_ID}"
echo "HOST=$(hostname)"
echo "CONFIG_PATH=${CONFIG_PATH}"
echo "PROCESSED_DATA=${PROCESSED_DATA}"
echo "PRETRAIN_ARGS=${PRETRAIN_ARGS}"
echo "WANDB_MODE=${WANDB_MODE:-offline}"
nvidia-smi || true

echo "HEALTH_GATE data_smoke_start"
python -m ala2_scratch.eval \
  --smoke-check-data \
  --processed-data "${PROCESSED_DATA}" \
  --phi-indices "${PHI_INDICES}" \
  --psi-indices "${PSI_INDICES}" \
  --batch-size 8 \
  --output-json "/tmp/ala2_pretrain_smoke_${SLURM_JOB_ID}.json"
echo "HEALTH_GATE data_smoke_passed"
echo "Initial health gate passed: processed data loads, splits, batches, and metric path is live."

CMD=(python -m ala2_scratch.pretrain --config "${CONFIG_PATH}")
if [[ -n "${PRETRAIN_ARGS}" ]]; then
  # shellcheck disable=SC2206
  EXTRA_ARGS=(${PRETRAIN_ARGS})
  CMD+=("${EXTRA_ARGS[@]}")
fi
"${CMD[@]}"
