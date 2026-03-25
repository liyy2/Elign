#!/usr/bin/env bash
set -euo pipefail

# Local (non-Slurm) GEOM RL runs for quick iteration on stable / low-variance settings.
# Runs sequentially on a single GPU.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

export PYTHONPATH="${REPO_ROOT}:${REPO_ROOT}/edm_source:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

CONDA_ENV="${CONDA_ENV:-edm}"
CONFIG_NAME="${CONFIG_NAME:-ddpo_geom_local_lowvar_rms_bestonly}"
WANDB_PROJECT="${WANDB_PROJECT:-ddpo}"
# Optional: wall-clock cap for local runs. Leave unset to run until you stop it.
MAX_TIME_HOURS="${MAX_TIME_HOURS:-}"

# Warmstart from the best checkpoint of the last good GEOM run.
CHECKPOINT_PATH="${CHECKPOINT_PATH:-${REPO_ROOT}/outputs/verl_geom/geom_rms_lr1e6_eadv02_dynE_kladapt_seed45_20260122_011205/checkpoint_best.pth}"

GEOM_DATA_FILE="${GEOM_DATA_FILE:-${REPO_ROOT}/geom_drugs_30.npy}"

if [[ ! -f "${CHECKPOINT_PATH}" ]]; then
  echo "ERROR: CHECKPOINT_PATH not found: ${CHECKPOINT_PATH}" >&2
  exit 1
fi
if [[ ! -f "${GEOM_DATA_FILE}" ]]; then
  echo "ERROR: GEOM_DATA_FILE not found: ${GEOM_DATA_FILE}" >&2
  exit 1
fi

timestamp="$(date +"%Y%m%d_%H%M%S")"

run_one() {
  local seed="$1"
  local lr="$2"
  local tag="$3"
  # Hydra CLI parsing is strict; keep run names alnum/underscore only.
  local lr_tag="${lr//[^a-zA-Z0-9]/_}"
  local run_name="geom_local_${tag}_lr_${lr_tag}_seed_${seed}_${timestamp}"
  local save_path="${REPO_ROOT}/outputs/verl_geom/${run_name}"
  mkdir -p "${save_path}"
  mkdir -p "${save_path}/wandb"

  echo "[INFO] starting ${run_name}"
  echo "[INFO] save_path=${save_path}"

  # Optional wall-clock cap; default is "no max_time_hours" (run until stopped).
  local -a time_flags=()
  if [[ -n "${MAX_TIME_HOURS}" ]]; then
    time_flags=("train.max_time_hours=${MAX_TIME_HOURS}")
  fi

  export WANDB_DIR="${save_path}/wandb"
  conda run --no-capture-output -n "${CONDA_ENV}" torchrun --standalone --nproc_per_node=1 run_verl_diffusion.py \
    --config-name "${CONFIG_NAME}" \
    wandb.enabled=true \
    wandb.wandb_project="${WANDB_PROJECT}" \
    wandb.wandb_name="${run_name}" \
    save_path="${save_path}" \
    resume=true \
    checkpoint_path="${CHECKPOINT_PATH}" \
    seed="${seed}" \
    dataloader.geom_data_file="${GEOM_DATA_FILE}" \
    train.learning_rate="${lr}" \
    train.kl_adaptive=true \
    train.kl_penalty_weight=0.12 \
    train.clip_range=0.05 \
    "${time_flags[@]}"
}

# Two quick local runs to compare LR sensitivity under the new "lowvar" config.
run_one 45 8e-7 "lowvar_bestonly"
run_one 45 1e-6 "lowvar_bestonly"

echo "[INFO] all local runs completed"
