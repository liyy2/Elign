#!/bin/bash
#SBATCH --job-name=post_train_geom
#SBATCH --output=post_train_geom_%j.out
#SBATCH --error=post_train_geom_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
##SBATCH --constraint=h200  # Optional: request H200 nodes on clusters that tag them as `h200`.

set -euo pipefail

# Usage:
#   sbatch --export=ALL,GEOM_DATA_FILE=/path/to/geom_drugs_30.npy post_train_diffusion_geom.sh
#
# Optional overrides:
#   sbatch --export=ALL,MODEL_CONFIG=./pretrained/edm/edm_geom_drugs/args.pickle,MODEL_WEIGHTS=./pretrained/edm/edm_geom_drugs/generative_model_ema.npy post_train_diffusion_geom.sh
#   sbatch --export=ALL,WANDB_ENABLED=1,WANDB_PROJECT=myproj post_train_diffusion_geom.sh
#
# Notes:
# - No tokens are embedded. Set `WANDB_API_KEY` / `WANDB_MODE` in your environment if needed.
# - Requires the GEOM conformation file `geom_drugs_30.npy` (pass via `GEOM_DATA_FILE`).

CONDA_ENV="${CONDA_ENV:-edm}"
WANDB_ENABLED="${WANDB_ENABLED:-1}"   # 1=enable wandb.init, 0=disable
WANDB_PROJECT="${WANDB_PROJECT:-ddpo}"

CONFIG_NAME="${CONFIG_NAME:-ddpo_geom_energy_force_vxu_suffix_tuned}"
REWARD_TYPE="${REWARD_TYPE:-polar_mace}"  # polar_mace | uma | dummy
EPOCHES="${EPOCHES:-}"             # optional override for dataloader.epoches
SEED="${SEED:-42}"

# Starting checkpoint for the diffusion policy (EDM)
MODEL_BACKEND="${MODEL_BACKEND:-auto}"  # auto | edm | geoldm
MODEL_CONFIG="${MODEL_CONFIG:-./pretrained/edm/edm_geom_drugs/args.pickle}"
MODEL_WEIGHTS="${MODEL_WEIGHTS:-./pretrained/edm/edm_geom_drugs/generative_model_ema.npy}"

# Optional: resume a DDPO checkpoint (full optimizer/model state).
# NOTE: `checkpoint_path` is ignored unless `resume=true`.
CHECKPOINT_PATH="${CHECKPOINT_PATH:-}"

# GEOM data (.npy). Prefer passing an absolute path via `GEOM_DATA_FILE=...`.
GEOM_DATA_FILE="${GEOM_DATA_FILE:-}"

# Prefer an existing `conda` on PATH; fall back to environment modules if needed.
# Some clusters' `module load miniconda` logic attempts to `conda deactivate` and can
# emit errors in non-interactive shells, so only load the module when `conda` is missing.
if ! command -v conda >/dev/null 2>&1; then
  if command -v module >/dev/null 2>&1; then
    module load miniconda || true
  fi
fi

if command -v conda >/dev/null 2>&1; then
  # Prefer sourcing conda.sh to ensure `conda activate` works in non-interactive shells.
  # Fall back to the shell hook when conda.sh is unavailable.
  conda_base="$(conda info --base 2>/dev/null || true)"
  if [[ -n "${conda_base}" && -f "${conda_base}/etc/profile.d/conda.sh" ]]; then
    # shellcheck source=/dev/null
    source "${conda_base}/etc/profile.d/conda.sh"
  else
    eval "$(conda shell.bash hook)"
  fi
  conda activate "${CONDA_ENV}"
fi

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
cd "${REPO_ROOT}"

export PYTHONPATH="${REPO_ROOT}:${REPO_ROOT}/edm_source:${PYTHONPATH:-}"

if [[ -z "${GEOM_DATA_FILE}" ]]; then
  if [[ -f "${REPO_ROOT}/geom_drugs_30.npy" ]]; then
    GEOM_DATA_FILE="${REPO_ROOT}/geom_drugs_30.npy"
  elif [[ -f "${REPO_ROOT}/datasets/geom/geom_drugs_30.npy" ]]; then
    GEOM_DATA_FILE="${REPO_ROOT}/datasets/geom/geom_drugs_30.npy"
  elif [[ -f "${REPO_ROOT}/data/geom/geom_drugs_30.npy" ]]; then
    GEOM_DATA_FILE="${REPO_ROOT}/data/geom/geom_drugs_30.npy"
  elif [[ -f "${REPO_ROOT}/edm_source/data/geom/geom_drugs_30.npy" ]]; then
    GEOM_DATA_FILE="${REPO_ROOT}/edm_source/data/geom/geom_drugs_30.npy"
  else
    echo "ERROR: GEOM conformation file not found."
    echo "Set GEOM_DATA_FILE=/path/to/geom_drugs_30.npy (recommended)."
    exit 1
  fi
fi

sanitize_for_name() {
  local value="$1"
  value="${value//[^a-zA-Z0-9]/_}"
  value="${value##_}"
  value="${value%%_}"
  echo "${value}"
}

# ----------------------------
# Optimization / training loop
# ----------------------------
LEARNING_RATE="${LEARNING_RATE:-2e-7}"
CLIP_RANGE="${CLIP_RANGE:-0.05}"
ADV_CLIP_MAX="${ADV_CLIP_MAX:-3}"
MAX_GRAD_NORM="${MAX_GRAD_NORM:-0.5}"
TRAIN_MICRO_BATCH_SIZE="${TRAIN_MICRO_BATCH_SIZE:-16}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-8}"
EPOCH_PER_ROLLOUT="${EPOCH_PER_ROLLOUT:-1}"
KL_PENALTY_WEIGHT="${KL_PENALTY_WEIGHT:-0.12}"
KL_ADAPTIVE="${KL_ADAPTIVE:-false}"
# Optional: allow long-running local jobs without being cut off by config defaults.
MAX_TIME_HOURS="${MAX_TIME_HOURS:-}"
EARLY_STOP_PATIENCE_MINUTES="${EARLY_STOP_PATIENCE_MINUTES:-}"
EARLY_STOP_MIN_DELTA="${EARLY_STOP_MIN_DELTA:-}"
SAVE_INTERVAL_OVERRIDE="${SAVE_INTERVAL_OVERRIDE:-}"

# ----------------------------
# Diffusion rollout settings
# ----------------------------
SAMPLE_GROUP_SIZE="${SAMPLE_GROUP_SIZE:-4}"
EACH_PROMPT_SAMPLE="${EACH_PROMPT_SAMPLE:-32}"
ROLLOUT_MICRO_BATCH_SIZE="${ROLLOUT_MICRO_BATCH_SIZE:-32}"
TIME_STEP="${TIME_STEP:-1000}"
SHARE_INITIAL_NOISE="${SHARE_INITIAL_NOISE:-true}"
MODEL_SKIP_PREFIX="${MODEL_SKIP_PREFIX:-${SKIP_PREFIX:-700}}"
FORCE_ALIGNMENT_ENABLED="${FORCE_ALIGNMENT_ENABLED:-true}"
FORCE_ALIGNMENT_WEIGHT="${FORCE_ALIGNMENT_WEIGHT:-0.02}"
FORCE_ALIGNMENT_MIN_FORCE="${FORCE_ALIGNMENT_MIN_FORCE:-0.0001}"
FORCE_ALIGNMENT_MIN_DELTA="${FORCE_ALIGNMENT_MIN_DELTA:-0.0001}"

# ----------------------------
# Reward configuration
# ----------------------------
USE_ENERGY="${USE_ENERGY:-true}"
FORCE_WEIGHT="${FORCE_WEIGHT:-1.0}"
# Optional override: GRPO advantage mixing weight for the *force* channel.
# When unset, the trainer falls back to `reward.force_weight` (usually 1.0).
FORCE_ADV_WEIGHT="${FORCE_ADV_WEIGHT:-}"
ENERGY_WEIGHT="${ENERGY_WEIGHT:-0.05}"
ENERGY_ADV_WEIGHT="${ENERGY_ADV_WEIGHT:-0.05}"
MLFF_MODEL="${MLFF_MODEL:-polar-1-m}"
MLFF_BATCH_SIZE="${MLFF_BATCH_SIZE:-16}"
FORCE_AGGREGATION="${FORCE_AGGREGATION:-rms}"
STABILITY_WEIGHT="${STABILITY_WEIGHT:-6.0}"
ATOM_STABILITY_WEIGHT="${ATOM_STABILITY_WEIGHT:-15.0}"
VALENCE_UNDERBOND_WEIGHT="${VALENCE_UNDERBOND_WEIGHT:-0.20}"
VALENCE_OVERBOND_WEIGHT="${VALENCE_OVERBOND_WEIGHT:-0.50}"
VALENCE_UNDERBOND_SOFT_WEIGHT="${VALENCE_UNDERBOND_SOFT_WEIGHT:-0.05}"
VALENCE_OVERBOND_SOFT_WEIGHT="${VALENCE_OVERBOND_SOFT_WEIGHT:-0.15}"
VALENCE_SOFT_TEMPERATURE="${VALENCE_SOFT_TEMPERATURE:-0.02}"
FORCE_CLIP_THRESHOLD="${FORCE_CLIP_THRESHOLD:-10.0}"
# Optional override; when unset, defer to the Hydra config.
FORCE_ONLY_IF_STABLE="${FORCE_ONLY_IF_STABLE:-}"
# Optional geometry guard: penalize extremely close atom pairs (to avoid clashy, DFT-high-force geometries).
MIN_PAIR_DIST_THRESHOLD="${MIN_PAIR_DIST_THRESHOLD:-}"
MIN_PAIR_DIST_PENALTY_WEIGHT="${MIN_PAIR_DIST_PENALTY_WEIGHT:-}"
MIN_PAIR_DIST_PENALTY_POWER="${MIN_PAIR_DIST_PENALTY_POWER:-}"

# Optional: speed/behavior toggles for UMA energy computation + GRPO advantage mixing schedule.
# When unset, defer to the Hydra config.
DYNAMIC_ENERGY_ENABLED="${DYNAMIC_ENERGY_ENABLED:-}"
DYNAMIC_ENERGY_THRESHOLD="${DYNAMIC_ENERGY_THRESHOLD:-}"
ADV_SCHEDULE_ENABLED="${ADV_SCHEDULE_ENABLED:-}"
ADV_SCHEDULE_USE_ABSOLUTE_EPOCH="${ADV_SCHEDULE_USE_ABSOLUTE_EPOCH:-}"
ENERGY_ADV_SCHED_START_EPOCH="${ENERGY_ADV_SCHED_START_EPOCH:-}"
ENERGY_ADV_SCHED_START="${ENERGY_ADV_SCHED_START:-}"
ENERGY_ADV_SCHED_END="${ENERGY_ADV_SCHED_END:-}"
ENERGY_ADV_SCHED_WARMUP_EPOCHS="${ENERGY_ADV_SCHED_WARMUP_EPOCHS:-}"
ENERGY_ADV_SCHED_RAMP_EPOCHS="${ENERGY_ADV_SCHED_RAMP_EPOCHS:-}"

# Optional overrides; when unset, defer to the Hydra config.
FILTER_INVALID_PENALTY_SCALE="${FILTER_INVALID_PENALTY_SCALE:-}"
FILTER_INVALID_REWARD_GATE_MODE="${FILTER_INVALID_REWARD_GATE_MODE:-}"
FILTER_DUPLICATE_PENALTY_SCALE="${FILTER_DUPLICATE_PENALTY_SCALE:-}"
FILTER_ENABLE_FILTERING="${FILTER_ENABLE_FILTERING:-}"
FILTER_HISTORY_SIZE="${FILTER_HISTORY_SIZE:-}"
FILTER_HISTORY_PENALTY_SCALE="${FILTER_HISTORY_PENALTY_SCALE:-}"
FILTER_HISTORY_PENALTY_MODE="${FILTER_HISTORY_PENALTY_MODE:-}"
FILTER_HISTORY_PENALTY_MAX_MULTIPLIER="${FILTER_HISTORY_PENALTY_MAX_MULTIPLIER:-}"

# Shaping settings
# NOTE: `reward.shaping.scheduler.skip_prefix` is an *index into the returned rollout trajectory*.
# With suffix-only rollouts (`model.return_suffix_only=true`), the prefix is already removed, so the
# shaping scheduler should typically use `0` to cover the full suffix.
SHAPING_SCHEDULER_SKIP_PREFIX="${SHAPING_SCHEDULER_SKIP_PREFIX:-0}"
# Optional override; when unset, defer to the Hydra config.
REWARD_SHAPING_ENABLED="${REWARD_SHAPING_ENABLED:-}"
SHAPING_ONLY_ENERGY="${SHAPING_ONLY_ENERGY:-true}"
SHAPING_GAMMA="${SHAPING_GAMMA:-1.0}"
TERMINAL_WEIGHT="${TERMINAL_WEIGHT:-2.0}"
SHAPING_SCHEDULER_MODE="${SHAPING_SCHEDULER_MODE:-adaptive}"   # adaptive | uniform
SHAPING_SCHEDULER_UNIFORM_STRIDE="${SHAPING_SCHEDULER_UNIFORM_STRIDE:-1}"
SHAPING_SCHEDULER_INCLUDE_TERMINAL="${SHAPING_SCHEDULER_INCLUDE_TERMINAL:-true}"
SHAPING_ADAPTIVE_COARSE_STRIDE="${SHAPING_ADAPTIVE_COARSE_STRIDE:-10}"
SHAPING_ADAPTIVE_FINE_STRIDE="${SHAPING_ADAPTIVE_FINE_STRIDE:-2}"
SHAPING_ADAPTIVE_THRESHOLD_FRACTION="${SHAPING_ADAPTIVE_THRESHOLD_FRACTION:-0.25}"

# ----------------------------
# Scheduler configuration
# ----------------------------
SCHEDULER_NAME="${SCHEDULER_NAME:-none}"
SCHEDULER_WARMUP_STEPS="${SCHEDULER_WARMUP_STEPS:-0}"
SCHEDULER_TOTAL_STEPS="${SCHEDULER_TOTAL_STEPS:-0}"
SCHEDULER_MIN_LR_RATIO="${SCHEDULER_MIN_LR_RATIO:-1.0}"

# ----------------------------
# Run naming / save path
# ----------------------------
timestamp=$(date +"%Y%m%d_%H%M%S")
MODEL_TAG=$(sanitize_for_name "${MLFF_MODEL}")
DEFAULT_RUN_NAME="verl_geom_${MODEL_TAG}_lr_$(sanitize_for_name "${LEARNING_RATE}")_${timestamp}"
RUN_NAME="${RUN_NAME:-${DEFAULT_RUN_NAME}}"

SAVE_ROOT="${SAVE_ROOT:-${REPO_ROOT}/outputs/verl_geom}"
SAVE_PATH="${SAVE_ROOT}/${RUN_NAME}"
mkdir -p "${SAVE_PATH}"
export WANDB_DIR="${WANDB_DIR:-${SAVE_PATH}/wandb}"
mkdir -p "${WANDB_DIR}" || true

# ----------------------------
# Reproducibility metadata
# ----------------------------
{
  echo "timestamp=${timestamp}"
  echo "hostname=$(hostname 2>/dev/null || true)"
  echo "slurm_job_id=${SLURM_JOB_ID:-}"
  echo "slurm_nodelist=${SLURM_JOB_NODELIST:-}"
  echo "config_name=${CONFIG_NAME}"
  echo "seed=${SEED}"
  echo "model_backend=${MODEL_BACKEND}"
  echo "geom_data_file=${GEOM_DATA_FILE}"
  echo "model_config=${MODEL_CONFIG}"
  echo "model_weights=${MODEL_WEIGHTS}"
} > "${SAVE_PATH}/run_metadata.txt" || true

if command -v git >/dev/null 2>&1 && git -C "${REPO_ROOT}" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  git -C "${REPO_ROOT}" rev-parse HEAD > "${SAVE_PATH}/git_commit.txt" || true
  git -C "${REPO_ROOT}" status --porcelain=v1 > "${SAVE_PATH}/git_status_porcelain.txt" || true
fi

if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi -L > "${SAVE_PATH}/nvidia_smi_L.txt" || true
fi

GPUS_PER_NODE="${GPUS_PER_NODE:-1}"
USE_TORCHRUN="${USE_TORCHRUN:-1}"  # 1=use torchrun (DDP-ready), 0=run single-process python

export MASTER_ADDR=${MASTER_ADDR:-$(hostname)}
export MASTER_PORT=${MASTER_PORT:-29500}

declare -a SHAPING_FLAGS
SHAPING_FLAGS=()
if [[ -n "${REWARD_SHAPING_ENABLED}" ]]; then
  if [[ "${REWARD_SHAPING_ENABLED}" == true ]]; then
    SHAPING_FLAGS=(
      "reward.shaping.enabled=true"
      "reward.shaping.gamma=${SHAPING_GAMMA}"
      "reward.shaping.scheduler.mode=${SHAPING_SCHEDULER_MODE}"
      "reward.shaping.scheduler.skip_prefix=${SHAPING_SCHEDULER_SKIP_PREFIX}"
      "reward.shaping.scheduler.uniform_stride=${SHAPING_SCHEDULER_UNIFORM_STRIDE}"
      "reward.shaping.scheduler.include_terminal=${SHAPING_SCHEDULER_INCLUDE_TERMINAL}"
      "reward.shaping.scheduler.adaptive.coarse_stride=${SHAPING_ADAPTIVE_COARSE_STRIDE}"
      "reward.shaping.scheduler.adaptive.fine_stride=${SHAPING_ADAPTIVE_FINE_STRIDE}"
      "reward.shaping.scheduler.adaptive.threshold_fraction=${SHAPING_ADAPTIVE_THRESHOLD_FRACTION}"
      "reward.shaping.only_energy_reshape=${SHAPING_ONLY_ENERGY}"
      "reward.shaping.terminal_weight=${TERMINAL_WEIGHT}"
    )
  else
    SHAPING_FLAGS=("reward.shaping.enabled=false")
  fi
fi

if [[ "${WANDB_ENABLED}" == "1" ]]; then
  # Guard against accidentally exporting a truncated API key (wandb errors out on <40 chars).
  if [[ -n "${WANDB_API_KEY:-}" && ${#WANDB_API_KEY} -lt 40 ]]; then
    echo "[WARN] WANDB_API_KEY is set but appears too short (<40 chars); unsetting and falling back to offline unless ~/.netrc provides a key."
    unset WANDB_API_KEY
  fi
  # Avoid Slurm jobs crashing on `wandb.init()` when no API key is configured.
  # If users want online logging they can either:
  # - export WANDB_API_KEY, or
  # - run `wandb login` (writes ~/.netrc).
  if [[ -n "${WANDB_API_KEY:-}" ]]; then
    export WANDB_MODE="${WANDB_MODE:-online}"
  elif [[ "${WANDB_MODE:-online}" == "offline" ]]; then
    export WANDB_MODE="offline"
  elif [[ -f "${HOME}/.netrc" ]] && grep -q "api.wandb.ai" "${HOME}/.netrc"; then
    export WANDB_MODE="${WANDB_MODE:-online}"
  else
    echo "[WARN] WANDB_ENABLED=1 but WANDB_API_KEY is not set and ~/.netrc has no api.wandb.ai entry; forcing WANDB_MODE=offline."
    export WANDB_MODE="offline"
  fi
  WANDB_FLAGS=("wandb.enabled=true" "wandb.wandb_project=${WANDB_PROJECT}" "wandb.wandb_name=${RUN_NAME}")
else
  export WANDB_MODE="${WANDB_MODE:-offline}"
  WANDB_FLAGS=("wandb.enabled=false")
fi

declare -a RESUME_FLAGS=()
if [[ -n "${CHECKPOINT_PATH}" ]]; then
  RESUME_FLAGS=("resume=true" "checkpoint_path=${CHECKPOINT_PATH}")
fi

EXTRA_FLAGS=()
if [[ -n "${EPOCHES}" ]]; then
  # When resuming from a checkpoint, `dataloader.epoches` is interpreted as the *total* number
  # of rollout/update iterations, and the trainer restarts at `checkpoint_epoch + 1`.
  #
  # If users pass a small `EPOCHES` (e.g. 20/50) intending "run 20 more iterations", the job
  # would otherwise terminate immediately because `start_epoch` is already large.
  #
  # Heuristic: when `CHECKPOINT_PATH` is set and `EPOCHES <= start_epoch`, treat it as an
  # additional-iterations budget and convert to a total epoch count.
  if [[ -n "${CHECKPOINT_PATH}" ]]; then
    resume_epoch=""
    if command -v python >/dev/null 2>&1; then
      resume_epoch="$(python - <<PY 2>/dev/null || true
import torch
try:
    ckpt = torch.load(r\"${CHECKPOINT_PATH}\", map_location=\"cpu\")
    print(int(ckpt.get(\"epoch\", 0)))
except Exception:
    print(\"\")
PY
)"
    fi
    if [[ -z "${resume_epoch}" ]]; then
      # Fallback: infer epoch from the checkpoint filename when torch.load is unavailable.
      ckpt_name="$(basename "${CHECKPOINT_PATH}")"
      if [[ "${ckpt_name}" =~ checkpoint_epoch_([0-9]+) ]]; then
        resume_epoch="${BASH_REMATCH[1]}"
      elif [[ "${ckpt_name}" =~ (^|[^0-9])[eE]([0-9]+)($|[^0-9]) ]]; then
        resume_epoch="${BASH_REMATCH[2]}"
      fi
    fi
    if [[ -n "${resume_epoch}" ]] && [[ "${resume_epoch}" =~ ^[0-9]+$ ]] && [[ "${EPOCHES}" =~ ^[0-9]+$ ]]; then
      start_epoch=$((resume_epoch + 1))
      if [[ "${EPOCHES}" -le "${start_epoch}" ]]; then
        echo "[INFO] resume checkpoint epoch=${resume_epoch}; interpreting EPOCHES=${EPOCHES} as additional iterations." >&2
        EPOCHES=$((start_epoch + EPOCHES))
        echo "[INFO] setting dataloader.epoches=${EPOCHES} (total)." >&2
      fi
    fi
  fi
  EXTRA_FLAGS+=("dataloader.epoches=${EPOCHES}")
fi

TRAIN_OPTIONAL_FLAGS=()
if [[ -n "${MAX_TIME_HOURS}" ]]; then
  TRAIN_OPTIONAL_FLAGS+=("train.max_time_hours=${MAX_TIME_HOURS}")
fi
if [[ -n "${EARLY_STOP_PATIENCE_MINUTES}" ]]; then
  TRAIN_OPTIONAL_FLAGS+=("train.early_stop_patience_minutes=${EARLY_STOP_PATIENCE_MINUTES}")
fi
if [[ -n "${EARLY_STOP_MIN_DELTA}" ]]; then
  TRAIN_OPTIONAL_FLAGS+=("train.early_stop_min_delta=${EARLY_STOP_MIN_DELTA}")
fi

SAVE_OPTIONAL_FLAGS=()
if [[ -n "${SAVE_INTERVAL_OVERRIDE}" ]]; then
  SAVE_OPTIONAL_FLAGS+=("save_interval=${SAVE_INTERVAL_OVERRIDE}")
fi

FILTER_HISTORY_FLAGS=()
if [[ -n "${FILTER_HISTORY_PENALTY_MODE}" ]]; then
  FILTER_HISTORY_FLAGS+=("filters.history_penalty_mode=${FILTER_HISTORY_PENALTY_MODE}")
fi
if [[ -n "${FILTER_HISTORY_PENALTY_MAX_MULTIPLIER}" ]]; then
  FILTER_HISTORY_FLAGS+=("filters.history_penalty_max_multiplier=${FILTER_HISTORY_PENALTY_MAX_MULTIPLIER}")
fi

FILTER_FLAGS=()
if [[ -n "${FILTER_ENABLE_FILTERING}" ]]; then
  FILTER_FLAGS+=("filters.enable_filtering=${FILTER_ENABLE_FILTERING}")
fi
if [[ -n "${FILTER_INVALID_PENALTY_SCALE}" ]]; then
  FILTER_FLAGS+=("filters.invalid_penalty_scale=${FILTER_INVALID_PENALTY_SCALE}")
fi
if [[ -n "${FILTER_INVALID_REWARD_GATE_MODE}" ]]; then
  FILTER_FLAGS+=("filters.invalid_reward_gate_mode=${FILTER_INVALID_REWARD_GATE_MODE}")
fi
if [[ -n "${FILTER_DUPLICATE_PENALTY_SCALE}" ]]; then
  FILTER_FLAGS+=("filters.duplicate_penalty_scale=${FILTER_DUPLICATE_PENALTY_SCALE}")
fi
if [[ -n "${FILTER_HISTORY_SIZE}" ]]; then
  FILTER_FLAGS+=("filters.history_size=${FILTER_HISTORY_SIZE}")
fi
if [[ -n "${FILTER_HISTORY_PENALTY_SCALE}" ]]; then
  FILTER_FLAGS+=("filters.history_penalty_scale=${FILTER_HISTORY_PENALTY_SCALE}")
fi

REWARD_OPTIONAL_FLAGS=()
if [[ -n "${FORCE_ONLY_IF_STABLE}" ]]; then
  REWARD_OPTIONAL_FLAGS+=("reward.force_only_if_stable=${FORCE_ONLY_IF_STABLE}")
fi
if [[ -n "${FORCE_ADV_WEIGHT}" ]]; then
  REWARD_OPTIONAL_FLAGS+=("reward.force_adv_weight=${FORCE_ADV_WEIGHT}")
fi

MIN_PAIR_DIST_FLAGS=()
if [[ -n "${MIN_PAIR_DIST_THRESHOLD}" ]]; then
  MIN_PAIR_DIST_FLAGS+=("++reward.min_pair_dist_threshold=${MIN_PAIR_DIST_THRESHOLD}")
fi
if [[ -n "${MIN_PAIR_DIST_PENALTY_WEIGHT}" ]]; then
  MIN_PAIR_DIST_FLAGS+=("++reward.min_pair_dist_penalty_weight=${MIN_PAIR_DIST_PENALTY_WEIGHT}")
fi
if [[ -n "${MIN_PAIR_DIST_PENALTY_POWER}" ]]; then
  MIN_PAIR_DIST_FLAGS+=("++reward.min_pair_dist_penalty_power=${MIN_PAIR_DIST_PENALTY_POWER}")
fi

DYNAMIC_ENERGY_FLAGS=()
if [[ -n "${DYNAMIC_ENERGY_ENABLED}" ]]; then
  # `reward.dynamic_energy` is not present in all Hydra configs; use `++` so the override
  # works whether the key exists or not (Hydra strict mode).
  DYNAMIC_ENERGY_FLAGS+=("++reward.dynamic_energy.enabled=${DYNAMIC_ENERGY_ENABLED}")
fi
if [[ -n "${DYNAMIC_ENERGY_THRESHOLD}" ]]; then
  DYNAMIC_ENERGY_FLAGS+=("++reward.dynamic_energy.enable_threshold=${DYNAMIC_ENERGY_THRESHOLD}")
fi

ADV_SCHEDULE_FLAGS=()
if [[ -n "${ADV_SCHEDULE_ENABLED}" ]]; then
  ADV_SCHEDULE_FLAGS+=("reward.adv_schedule.enabled=${ADV_SCHEDULE_ENABLED}")
fi
if [[ -n "${ADV_SCHEDULE_USE_ABSOLUTE_EPOCH}" ]]; then
  ADV_SCHEDULE_FLAGS+=("reward.adv_schedule.use_absolute_epoch=${ADV_SCHEDULE_USE_ABSOLUTE_EPOCH}")
fi
if [[ -n "${ENERGY_ADV_SCHED_START_EPOCH}" ]]; then
  ADV_SCHEDULE_FLAGS+=("reward.adv_schedule.energy.start_epoch=${ENERGY_ADV_SCHED_START_EPOCH}")
fi
if [[ -n "${ENERGY_ADV_SCHED_START}" ]]; then
  ADV_SCHEDULE_FLAGS+=("reward.adv_schedule.energy.start=${ENERGY_ADV_SCHED_START}")
fi
if [[ -n "${ENERGY_ADV_SCHED_END}" ]]; then
  ADV_SCHEDULE_FLAGS+=("reward.adv_schedule.energy.end=${ENERGY_ADV_SCHED_END}")
fi
if [[ -n "${ENERGY_ADV_SCHED_WARMUP_EPOCHS}" ]]; then
  ADV_SCHEDULE_FLAGS+=("reward.adv_schedule.energy.warmup_epochs=${ENERGY_ADV_SCHED_WARMUP_EPOCHS}")
fi
if [[ -n "${ENERGY_ADV_SCHED_RAMP_EPOCHS}" ]]; then
  ADV_SCHEDULE_FLAGS+=("reward.adv_schedule.energy.ramp_epochs=${ENERGY_ADV_SCHED_RAMP_EPOCHS}")
fi

if [[ "${USE_TORCHRUN}" == "1" ]]; then
  LAUNCHER=(torchrun --standalone --nproc_per_node="${GPUS_PER_NODE}")
else
  LAUNCHER=(python -u)
fi

"${LAUNCHER[@]}" run_verl_diffusion.py \
  --config-name "${CONFIG_NAME}" \
  "${WANDB_FLAGS[@]}" \
  save_path="${SAVE_PATH}" \
  "${RESUME_FLAGS[@]}" \
  seed="${SEED}" \
  model.backend="${MODEL_BACKEND}" \
  model.config="${MODEL_CONFIG}" \
  model.model_path="${MODEL_WEIGHTS}" \
  dataloader.geom_data_file="${GEOM_DATA_FILE}" \
  dataloader.micro_batch_size="${ROLLOUT_MICRO_BATCH_SIZE}" \
  reward.type="${REWARD_TYPE}" \
		  train.learning_rate="${LEARNING_RATE}" \
		  train.clip_range="${CLIP_RANGE}" \
		  train.adv_clip_max="${ADV_CLIP_MAX}" \
	  train.max_grad_norm="${MAX_GRAD_NORM}" \
	  train.kl_penalty_weight="${KL_PENALTY_WEIGHT}" \
	  train.kl_adaptive="${KL_ADAPTIVE}" \
	  train.train_micro_batch_size="${TRAIN_MICRO_BATCH_SIZE}" \
	  train.gradient_accumulation_steps="${GRADIENT_ACCUMULATION_STEPS}" \
	  train.epoch_per_rollout="${EPOCH_PER_ROLLOUT}" \
  model.time_step="${TIME_STEP}" \
  model.share_initial_noise="${SHARE_INITIAL_NOISE}" \
  model.skip_prefix="${MODEL_SKIP_PREFIX}" \
  dataloader.sample_group_size="${SAMPLE_GROUP_SIZE}" \
  dataloader.each_prompt_sample="${EACH_PROMPT_SAMPLE}" \
  train.force_alignment_enabled="${FORCE_ALIGNMENT_ENABLED}" \
  train.force_alignment_weight="${FORCE_ALIGNMENT_WEIGHT}" \
	  train.force_alignment_min_force="${FORCE_ALIGNMENT_MIN_FORCE}" \
	  train.force_alignment_min_delta="${FORCE_ALIGNMENT_MIN_DELTA}" \
	  reward.use_energy="${USE_ENERGY}" \
	  reward.force_weight="${FORCE_WEIGHT}" \
	  reward.energy_weight="${ENERGY_WEIGHT}" \
	  reward.energy_adv_weight="${ENERGY_ADV_WEIGHT}" \
	  reward.mlff_model="${MLFF_MODEL}" \
  reward.shaping.mlff_batch_size="${MLFF_BATCH_SIZE}" \
  reward.force_aggregation="${FORCE_AGGREGATION}" \
  reward.stability_weight="${STABILITY_WEIGHT}" \
  reward.atom_stability_weight="${ATOM_STABILITY_WEIGHT}" \
  reward.valence_underbond_weight="${VALENCE_UNDERBOND_WEIGHT}" \
  reward.valence_overbond_weight="${VALENCE_OVERBOND_WEIGHT}" \
	  reward.valence_underbond_soft_weight="${VALENCE_UNDERBOND_SOFT_WEIGHT}" \
	  reward.valence_overbond_soft_weight="${VALENCE_OVERBOND_SOFT_WEIGHT}" \
	  reward.valence_soft_temperature="${VALENCE_SOFT_TEMPERATURE}" \
	  reward.force_clip_threshold="${FORCE_CLIP_THRESHOLD}" \
	  "${REWARD_OPTIONAL_FLAGS[@]}" \
	  "${MIN_PAIR_DIST_FLAGS[@]}" \
	  "${DYNAMIC_ENERGY_FLAGS[@]}" \
	  "${ADV_SCHEDULE_FLAGS[@]}" \
  "${FILTER_FLAGS[@]}" \
	  "${FILTER_HISTORY_FLAGS[@]}" \
	  "${TRAIN_OPTIONAL_FLAGS[@]}" \
	  "${SAVE_OPTIONAL_FLAGS[@]}" \
	  train.scheduler.name="${SCHEDULER_NAME}" \
	  train.scheduler.warmup_steps="${SCHEDULER_WARMUP_STEPS}" \
	  train.scheduler.total_steps="${SCHEDULER_TOTAL_STEPS}" \
  train.scheduler.min_lr_ratio="${SCHEDULER_MIN_LR_RATIO}" \
  "${EXTRA_FLAGS[@]}" \
  "${SHAPING_FLAGS[@]}"
