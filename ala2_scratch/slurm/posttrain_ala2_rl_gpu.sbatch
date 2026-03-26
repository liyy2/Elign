#!/bin/bash
#SBATCH --job-name=ala2_rl
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=/home/yl2428/logs/slurm/%x_%j.out
#SBATCH --error=/home/yl2428/logs/slurm/%x_%j.err

set -euo pipefail

REPO_DIR="/home/yl2428/e3_diffusion_for_molecules-main"
cd "${REPO_DIR}"

mkdir -p /home/yl2428/logs/slurm

module load miniconda
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate edm

RUN_NAME="${RUN_NAME:-ala2_rl_ctrl}"
CONFIG_PATH="${CONFIG_PATH:-ala2_scratch/configs/ala2_posttrain.json}"
PRETRAINED_CKPT="${PRETRAINED_CKPT:-/home/yl2428/logs/ala2_scratch/stride_sweep/stride20_match2750_idx16_20260326_1228/checkpoints/final.pt}"
REFERENCE_CKPT="${REFERENCE_CKPT:-${PRETRAINED_CKPT}}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/home/yl2428/logs/ala2_scratch/slurm_runs}"
OUTPUT_DIR="${OUTPUT_DIR:-${OUTPUT_ROOT}/${RUN_NAME}}"

DATASET_PATH="${DATASET_PATH:-data/ala2/stride20/ala2_300k_stride20.npz}"
TOPOLOGY_PATH="${TOPOLOGY_PATH:-data/ala2/stride20/ala2_topology.json}"
METADATA_PATH="${METADATA_PATH:-${TOPOLOGY_PATH}}"
SPLIT_PATH="${SPLIT_PATH:-data/ala2/stride20/ala2_splits.npz}"

MLFF_DEVICE="${MLFF_DEVICE:-cuda}"
MLFF_MICROBATCH="${MLFF_MICROBATCH:-2}"

TRAIN_STEPS="${TRAIN_STEPS:-10}"
BATCH_SIZE="${BATCH_SIZE:-128}"
SAMPLING_MICROBATCH="${SAMPLING_MICROBATCH:-16}"
REPLAY_MICROBATCH="${REPLAY_MICROBATCH:-2}"
REPLAY_SUBSET_SIZE="${REPLAY_SUBSET_SIZE:-16}"
REPLAY_SUBSET_MODE="${REPLAY_SUBSET_MODE:-abs_advantage}"
LEARNING_RATE="${LEARNING_RATE:-1e-7}"
REFERENCE_L2="${REFERENCE_L2:-1e-4}"
NORMALIZE_ADVANTAGES="${NORMALIZE_ADVANTAGES:-true}"
REWARD_BASELINE_MODE="${REWARD_BASELINE_MODE:-batch_mean}"
LOGPROB_TAIL_STEPS="${LOGPROB_TAIL_STEPS:-25}"
LOGPROB_SKIP_PREFIX="${LOGPROB_SKIP_PREFIX:-0}"
SAVE_EVERY="${SAVE_EVERY:-2}"
EVAL_EVERY="${EVAL_EVERY:-2}"
LOG_EVERY="${LOG_EVERY:-1}"
EVAL_SAMPLE_COUNT="${EVAL_SAMPLE_COUNT:-256}"
EVAL_SAMPLE_BATCH_SIZE="${EVAL_SAMPLE_BATCH_SIZE:-32}"
WANDB_ENABLED="${WANDB_ENABLED:-false}"

mkdir -p "${OUTPUT_ROOT}"

echo "JOBID=${SLURM_JOB_ID}"
echo "RUN_NAME=${RUN_NAME}"
echo "OUTPUT_DIR=${OUTPUT_DIR}"

python -m ala2_scratch.posttrain \
  --config "${CONFIG_PATH}" \
  --output-dir "${OUTPUT_DIR}" \
  --override "pretrained_checkpoint=${PRETRAINED_CKPT}" \
  --override "reference_checkpoint=${REFERENCE_CKPT}" \
  --override "data.dataset_path=${DATASET_PATH}" \
  --override "data.topology_path=${TOPOLOGY_PATH}" \
  --override "data.metadata_path=${METADATA_PATH}" \
  --override "data.split_path=${SPLIT_PATH}" \
  --override "mlff.device=${MLFF_DEVICE}" \
  --override "mlff.microbatch_size=${MLFF_MICROBATCH}" \
  --override "train.steps=${TRAIN_STEPS}" \
  --override "train.batch_size=${BATCH_SIZE}" \
  --override "train.sampling_microbatch_size=${SAMPLING_MICROBATCH}" \
  --override "train.rollout_microbatch_size=${REPLAY_MICROBATCH}" \
  --override "train.replay_subset_size=${REPLAY_SUBSET_SIZE}" \
  --override "train.replay_subset_mode=${REPLAY_SUBSET_MODE}" \
  --override "train.lr=${LEARNING_RATE}" \
  --override "train.reference_l2_weight=${REFERENCE_L2}" \
  --override "train.normalize_advantages=${NORMALIZE_ADVANTAGES}" \
  --override "train.reward_baseline_mode=${REWARD_BASELINE_MODE}" \
  --override "train.logprob_tail_steps=${LOGPROB_TAIL_STEPS}" \
  --override "train.logprob_skip_prefix=${LOGPROB_SKIP_PREFIX}" \
  --override "train.save_every=${SAVE_EVERY}" \
  --override "train.eval_every=${EVAL_EVERY}" \
  --override "train.log_every=${LOG_EVERY}" \
  --override "eval.sample_count=${EVAL_SAMPLE_COUNT}" \
  --override "eval.sample_batch_size=${EVAL_SAMPLE_BATCH_SIZE}" \
  --override "wandb.enabled=${WANDB_ENABLED}"
