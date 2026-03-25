#!/bin/bash
#SBATCH --job-name=prepare_geom30
#SBATCH --partition=day
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=prepare_geom30_%j.out
#SBATCH --error=prepare_geom30_%j.err

set -euo pipefail

if command -v module >/dev/null 2>&1; then
  module load miniconda
fi

if command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook)"
  conda activate edm
fi

export DATA_DIR="${DATA_DIR:-/gpfs/radev/scratch/gerstein/yl2428/datasets/geom}"
export CONFORMERS="${CONFORMERS:-30}"
export OUTPUT_DTYPE="${OUTPUT_DTYPE:-float32}"

echo "DATA_DIR=${DATA_DIR}"
echo "CONFORMERS=${CONFORMERS}"
echo "OUTPUT_DTYPE=${OUTPUT_DTYPE}"

bash prepare_geom_drugs_30.sh
