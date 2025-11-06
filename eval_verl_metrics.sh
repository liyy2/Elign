#!/bin/bash

set -euo pipefail

RUN_DIR="/home/yl2428/logs/verl_model_uma_s_1p1_energy_no_energy_lr_4e_6_clip_2e_3_share_noise_true_pps_24_skip_700_force_align_false_tstep_1000_sg_1_mlff_16_fagg_rms_stabW_1_sched_cosine_warmup_60_steps_1500_decay_0_3_shaping_false_epoch_per_rollout_1_20251104_193508"
CHECKPOINT_PATH="${RUN_DIR}/checkpoint_latest.pth"
SAMPLES_PATH="${RUN_DIR}/eval_rollouts.pt"
METRICS_JSON="${RUN_DIR}/eval_metrics.json"
RAW_METRICS="${RUN_DIR}/eval_metrics_raw.pt"
ARGS_PICKLE="/home/yl2428/e3_diffusion_for_molecules-main/pretrained/edm/edm_qm9/args.pickle"
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
  python eval_verl_rollout.py \
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

python compute_verl_metrics.py \
  --run-dir "${RUN_DIR}" \
  --samples "${SAMPLES_PATH}" \
  --args-pickle "${ARGS_PICKLE}" \
  --output "${METRICS_JSON}" \
  --save-raw "${RAW_METRICS}"
