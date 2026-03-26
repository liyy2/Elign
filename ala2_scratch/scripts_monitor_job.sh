#!/bin/bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "usage: $0 <job_id> [interval_seconds]"
  exit 1
fi

JOB_ID="$1"
INTERVAL="${2:-300}"

echo "Monitoring SLURM job ${JOB_ID} every ${INTERVAL}s"
echo "Initial health gate: look for 'HEALTH_GATE data_smoke_passed' in stdout before trusting training metrics."

while true; do
  echo
  echo "===== $(date '+%Y-%m-%d %H:%M:%S') ====="
  squeue -j "${JOB_ID}" || true

  OUT_FILE="$(find . -maxdepth 1 -type f -name "*_${JOB_ID}.out" | head -n 1)"
  ERR_FILE="$(find . -maxdepth 1 -type f -name "*_${JOB_ID}.err" | head -n 1)"

  if [[ -n "${OUT_FILE}" ]]; then
    echo "--- stdout: ${OUT_FILE} ---"
    if grep -q "HEALTH_GATE data_smoke_passed" "${OUT_FILE}"; then
      echo "health_gate=passed"
    else
      echo "health_gate=pending"
    fi
    tail -n 40 "${OUT_FILE}" || true
  else
    echo "stdout file not found yet"
  fi

  if [[ -n "${ERR_FILE}" ]]; then
    echo "--- stderr: ${ERR_FILE} ---"
    tail -n 20 "${ERR_FILE}" || true
  else
    echo "stderr file not found yet"
  fi

  echo "Waiting ${INTERVAL}s ..."
  sleep "${INTERVAL}"
done
