#!/bin/bash
#SBATCH --job-name=eval_geom_1024
#SBATCH --output=eval_geom_1024_%j.out
#SBATCH --error=eval_geom_1024_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu

set -euo pipefail

# Usage:
#   sbatch --export=ALL,RUN_DIR=/abs/path/to/run codex_scripts/eval_geom_1024.sbatch
#
# Optional overrides:
#   sbatch --export=ALL,RUN_DIR=...,CHECKPOINT=checkpoint_best.pth,SEED=43 codex_scripts/eval_geom_1024.sbatch
#   sbatch --export=ALL,RUN_DIR=...,NUM_MOLECULES=1024,NO_SHARE=1 codex_scripts/eval_geom_1024.sbatch
#   sbatch --export=ALL,RUN_DIR=...,REPAIR_UNSTABLE=1 codex_scripts/eval_geom_1024.sbatch
#   sbatch --export=ALL,RUN_DIR=...,REPAIR_ADD_H=1 codex_scripts/eval_geom_1024.sbatch

RUN_DIR="${RUN_DIR:?Set RUN_DIR to the VERL run directory (contains config.yaml/checkpoints).}"
CHECKPOINT="${CHECKPOINT:-}"        # optional (relative to RUN_DIR or absolute). Defaults to checkpoint_best/latest.
TAG="${TAG:-}"                      # optional label to avoid overwriting eval files (e.g., checkpoint_epoch_1200).
SEED="${SEED:-42}"
NUM_MOLECULES="${NUM_MOLECULES:-1024}"
NO_SHARE="${NO_SHARE:-0}"           # 1 = disable shared initial noise (evaluation only)
REPAIR_INVALID="${REPAIR_INVALID:-0}"  # 1 = run UMA relaxation on RDKit-invalid samples before metrics
REPAIR_UNSTABLE="${REPAIR_UNSTABLE:-0}" # 1 = run UMA relaxation on QM9-unstable samples before metrics
REPAIR_ADD_H="${REPAIR_ADD_H:-0}"      # 1 = add explicit H atoms before stability check
REPAIR_STEPS="${REPAIR_STEPS:-3}"
REPAIR_ALPHA="${REPAIR_ALPHA:-0.02}"

if command -v module >/dev/null 2>&1; then
  module load miniconda
fi

if command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook)"
  conda activate edm
fi

cd /gpfs/radev/home/yl2428/e3_diffusion_for_molecules-main

samples_suffix=""
extra_rollout_flags=()
if [[ "${NO_SHARE}" == "1" ]]; then
  samples_suffix="_noshare"
  extra_rollout_flags+=(--no-share-initial-noise)
fi

if [[ -z "${CHECKPOINT}" ]]; then
  # Prefer the run's best checkpoint when present, else fall back to latest.
  if [[ -f "${RUN_DIR}/checkpoint_best.pth" ]]; then
    CHECKPOINT="checkpoint_best.pth"
  elif [[ -f "${RUN_DIR}/checkpoint_latest.pth" ]]; then
    CHECKPOINT="checkpoint_latest.pth"
  else
    echo "ERROR: Could not infer checkpoint. Expected ${RUN_DIR}/checkpoint_best.pth or checkpoint_latest.pth." >&2
    exit 1
  fi
fi

if [[ "${CHECKPOINT}" = /* ]]; then
  CHECKPOINT_PATH="${CHECKPOINT}"
else
  CHECKPOINT_PATH="${RUN_DIR}/${CHECKPOINT}"
fi

if [[ ! -f "${CHECKPOINT_PATH}" ]]; then
  echo "ERROR: Checkpoint not found: ${CHECKPOINT_PATH}" >&2
  exit 1
fi

# Snapshot the checkpoint to avoid races with training overwriting `checkpoint_best.pth`.
# Prefer hardlinking (fast + preserves old inode even if training replaces the file), fall back to copying.
SNAPSHOT_TMP_SUFFIX="${SLURM_JOB_ID:-$$}"
SNAPSHOT_CKPT_TMP="${RUN_DIR}/eval_checkpoint_${NUM_MOLECULES}_seed${SEED}${samples_suffix}_tmp_${SNAPSHOT_TMP_SUFFIX}.pth"
if ! ln -f "${CHECKPOINT_PATH}" "${SNAPSHOT_CKPT_TMP}" 2>/dev/null; then
  cp -f "${CHECKPOINT_PATH}" "${SNAPSHOT_CKPT_TMP}"
fi

EVAL_EPOCH="$(python - "${SNAPSHOT_CKPT_TMP}" <<'PY'
from pathlib import Path
import sys

ckpt_path = Path(sys.argv[1]).resolve()
epoch = None
try:
    import torch
except Exception:
    print("")
    raise SystemExit(0)

try:
    payload = torch.load(ckpt_path, map_location="cpu")
    if isinstance(payload, dict):
        epoch = payload.get("epoch")
except Exception:
    epoch = None

try:
    epoch = int(epoch) if epoch is not None else None
except Exception:
    epoch = None

print("" if epoch is None else str(epoch))
PY
)"

if [[ -z "${TAG}" ]]; then
  TAG="$(basename "${CHECKPOINT}")"
  TAG="${TAG%.*}"
  if [[ -n "${EVAL_EPOCH}" && ( "${TAG}" == "checkpoint_best" || "${TAG}" == "checkpoint_latest" ) ]]; then
    TAG="${TAG}_e${EVAL_EPOCH}"
  fi
fi

SAMPLES_PATH="${RUN_DIR}/eval_${NUM_MOLECULES}_seed${SEED}${samples_suffix}_${TAG}.pt"
METRICS_JSON_BASE="${RUN_DIR}/eval_metrics_${NUM_MOLECULES}_seed${SEED}${samples_suffix}_${TAG}_skipmlff.json"
METRICS_JSON_REPAIR=""
repair_suffix=""
repair_flags=()

alpha_tag=""
if [[ "${REPAIR_INVALID}" == "1" || "${REPAIR_UNSTABLE}" == "1" ]]; then
  alpha_tag="$(python - <<'PY'
import os
alpha = float(os.environ.get("REPAIR_ALPHA", "0.0") or 0.0)
print(f"{int(round(alpha * 100)):03d}")
PY
)"
fi

if [[ "${REPAIR_INVALID}" == "1" ]]; then
  repair_suffix="${repair_suffix}_repair_invalid${REPAIR_STEPS}_a${alpha_tag}"
  repair_flags+=(--repair-invalid)
fi

if [[ "${REPAIR_UNSTABLE}" == "1" ]]; then
  repair_suffix="${repair_suffix}_repair_unstable${REPAIR_STEPS}_a${alpha_tag}"
  repair_flags+=(--repair-unstable)
fi

if [[ "${REPAIR_ADD_H}" == "1" ]]; then
  repair_suffix="${repair_suffix}_repair_addh"
  repair_flags+=(--repair-add-h)
fi

if [[ -n "${repair_suffix}" ]]; then
  METRICS_JSON_REPAIR="${RUN_DIR}/eval_metrics_${NUM_MOLECULES}_seed${SEED}${samples_suffix}_${TAG}_skipmlff${repair_suffix}.json"
fi

if [[ "${REPAIR_INVALID}" == "1" || "${REPAIR_UNSTABLE}" == "1" ]]; then
  repair_flags+=(--repair-steps "${REPAIR_STEPS}" --repair-alpha "${REPAIR_ALPHA}")
fi

SNAPSHOT_CKPT="${RUN_DIR}/eval_checkpoint_${NUM_MOLECULES}_seed${SEED}${samples_suffix}_${TAG}.pth"
rm -f "${SNAPSHOT_CKPT}"
mv -f "${SNAPSHOT_CKPT_TMP}" "${SNAPSHOT_CKPT}"

python -u eval_verl_rollout.py \
  --run-dir "${RUN_DIR}" \
  --checkpoint "${SNAPSHOT_CKPT}" \
  --output "${SAMPLES_PATH}" \
  --num-molecules "${NUM_MOLECULES}" \
  --seed "${SEED}" \
  "${extra_rollout_flags[@]}"

python -u compute_verl_metrics.py \
  --run-dir "${RUN_DIR}" \
  --samples "${SAMPLES_PATH}" \
  --output "${METRICS_JSON_BASE}" \
  --skip-mlff-metrics

if [[ -n "${METRICS_JSON_REPAIR}" ]]; then
  python -u compute_verl_metrics.py \
    --run-dir "${RUN_DIR}" \
    --samples "${SAMPLES_PATH}" \
    --output "${METRICS_JSON_REPAIR}" \
    --skip-mlff-metrics \
    "${repair_flags[@]}"
fi

python - "${METRICS_JSON_BASE}" "${SNAPSHOT_CKPT}" <<'PY'
import json
from pathlib import Path
import sys

import torch

metrics_path = Path(sys.argv[1]).resolve()
ckpt_path = Path(sys.argv[2]).resolve()

payload = json.loads(metrics_path.read_text())
epoch = None
try:
    ckpt = torch.load(ckpt_path, map_location="cpu")
    if isinstance(ckpt, dict):
        epoch = ckpt.get("epoch")
except Exception:
    epoch = None

payload["eval_checkpoint"] = str(ckpt_path)
payload["eval_checkpoint_epoch"] = epoch
metrics_path.write_text(json.dumps(payload) + "\n")
PY

if [[ -n "${METRICS_JSON_REPAIR}" ]]; then
  python - "${METRICS_JSON_REPAIR}" "${SNAPSHOT_CKPT}" <<'PY'
import json
from pathlib import Path
import sys

import torch

metrics_path = Path(sys.argv[1]).resolve()
ckpt_path = Path(sys.argv[2]).resolve()

payload = json.loads(metrics_path.read_text())
epoch = None
try:
    ckpt = torch.load(ckpt_path, map_location="cpu")
    if isinstance(ckpt, dict):
        epoch = ckpt.get("epoch")
except Exception:
    epoch = None

payload["eval_checkpoint"] = str(ckpt_path)
payload["eval_checkpoint_epoch"] = epoch
metrics_path.write_text(json.dumps(payload) + "\n")
PY
fi

echo "[OK] Wrote ${METRICS_JSON_BASE}"
if [[ -n "${METRICS_JSON_REPAIR}" ]]; then
  echo "[OK] Wrote ${METRICS_JSON_REPAIR}"
fi
