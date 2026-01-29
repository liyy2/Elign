#!/bin/bash

set -euo pipefail

RUN_DIR="${RUN_DIR:-}"
if [[ -z "${RUN_DIR}" ]]; then
  echo "Set RUN_DIR to an ELIGN run directory (contains checkpoints + config.yaml)." >&2
  exit 1
fi
CHECKPOINT_PATH="${RUN_DIR}/checkpoint_latest.pth"
SAMPLES_PATH="${RUN_DIR}/eval_rollouts.pt"
METRICS_JSON="${RUN_DIR}/eval_metrics.json"
RAW_METRICS="${RUN_DIR}/eval_metrics_raw.pt"
ARGS_PICKLE="${ARGS_PICKLE:-pretrained/edm/edm_qm9/args.pickle}"
NUM_MOLECULES="${1:-10000}"
TIME_STEP="${2:-1000}"
SAMPLE_GROUP_SIZE="${3:-16}"
EACH_PROMPT_SAMPLE="${4:-32}"
FORCE_RESAMPLE="${5:-0}"

if [[ -f "${SAMPLES_PATH}" && "${FORCE_RESAMPLE}" == "0" ]]; then
  echo "[INFO] Found existing ${SAMPLES_PATH}; skipping rollout generation."
else
  if [[ "${FORCE_RESAMPLE}" != "0" ]]; then
    echo "[INFO] Regenerating rollouts (FORCE_RESAMPLE=${FORCE_RESAMPLE})."
  else
    echo "[INFO] Generating rollouts into ${SAMPLES_PATH}."
  fi
  python eval_elign_rollout.py \
    --run-dir "${RUN_DIR}" \
    --checkpoint "${CHECKPOINT_PATH}" \
    --args-pickle "${ARGS_PICKLE}" \
    --output "${SAMPLES_PATH}" \
    --num-molecules "${NUM_MOLECULES}" \
    --time-step "${TIME_STEP}" \
    --sample-group-size "${SAMPLE_GROUP_SIZE}" \
    --each-prompt-sample "${EACH_PROMPT_SAMPLE}" \
    --skip-prefix 0
fi

python compute_elign_metrics.py \
  --run-dir "${RUN_DIR}" \
  --samples "${SAMPLES_PATH}" \
  --args-pickle "${ARGS_PICKLE}" \
  --output "${METRICS_JSON}" \
  --save-raw "${RAW_METRICS}"
