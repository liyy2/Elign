#!/bin/bash
set -euo pipefail

# Submit two fresh-from-pretrained GEOM post-training jobs:
#  1) Tuned-only (no codepath changes; just safer reward/training knobs)
#  2) Tuned + min-pair-distance penalty (guards against collapsed/clashy geometries)
#
# Usage:
#   ./codex_scripts/submit_geom_pretrained_two_versions.sh
# Optional overrides:
#   WANDB_PROJECT=ddpo SEED_TUNED=45 SEED_MINDIST=46 GEOM_DATA_FILE=/path/to/geom_drugs_30.npy ./codex_scripts/submit_geom_pretrained_two_versions.sh

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

timestamp="$(date +"%Y%m%d_%H%M%S")"

WANDB_PROJECT="${WANDB_PROJECT:-ddpo}"
GEOM_DATA_FILE="${GEOM_DATA_FILE:-${REPO_ROOT}/geom_drugs_30.npy}"

SEED_TUNED="${SEED_TUNED:-45}"
SEED_MINDIST="${SEED_MINDIST:-46}"

CONFIG_NAME="${CONFIG_NAME:-ddpo_geom_local_lowvar_rms_bestonly}"
MLFF_MODEL="${MLFF_MODEL:-uma-s-1p1}"
MODEL_TAG="${MLFF_MODEL//[^a-zA-Z0-9]/_}"

# Common low-variance + conservative PPO knobs.
COMMON_EXPORTS="WANDB_ENABLED=1"
COMMON_EXPORTS+=",WANDB_PROJECT=${WANDB_PROJECT}"
COMMON_EXPORTS+=",CONFIG_NAME=${CONFIG_NAME}"
COMMON_EXPORTS+=",REWARD_TYPE=uma"
COMMON_EXPORTS+=",MODEL_CONFIG=./pretrained/edm/edm_geom_drugs/args.pickle"
COMMON_EXPORTS+=",MODEL_WEIGHTS=./pretrained/edm/edm_geom_drugs/generative_model_ema.npy"
COMMON_EXPORTS+=",GEOM_DATA_FILE=${GEOM_DATA_FILE}"
COMMON_EXPORTS+=",SAMPLE_GROUP_SIZE=4"
COMMON_EXPORTS+=",EACH_PROMPT_SAMPLE=32"
COMMON_EXPORTS+=",ROLLOUT_MICRO_BATCH_SIZE=32"
COMMON_EXPORTS+=",TRAIN_MICRO_BATCH_SIZE=16"
COMMON_EXPORTS+=",GRADIENT_ACCUMULATION_STEPS=8"
COMMON_EXPORTS+=",LEARNING_RATE=1e-7"
COMMON_EXPORTS+=",CLIP_RANGE=0.03"
COMMON_EXPORTS+=",KL_PENALTY_WEIGHT=0.20"
COMMON_EXPORTS+=",ADV_CLIP_MAX=2.0"
COMMON_EXPORTS+=",MAX_GRAD_NORM=0.5"
COMMON_EXPORTS+=",MODEL_SKIP_PREFIX=700"
COMMON_EXPORTS+=",SHARE_INITIAL_NOISE=true"
COMMON_EXPORTS+=",FORCE_ALIGNMENT_ENABLED=true"
COMMON_EXPORTS+=",FORCE_ALIGNMENT_WEIGHT=0.05"
COMMON_EXPORTS+=",FORCE_ALIGNMENT_MIN_FORCE=0.0001"
COMMON_EXPORTS+=",FORCE_ALIGNMENT_MIN_DELTA=0.0001"
COMMON_EXPORTS+=",USE_ENERGY=true"
COMMON_EXPORTS+=",ENERGY_WEIGHT=0.05"
COMMON_EXPORTS+=",ENERGY_ADV_WEIGHT=0.02"
COMMON_EXPORTS+=",MLFF_MODEL=${MLFF_MODEL}"
COMMON_EXPORTS+=",FORCE_AGGREGATION=rms"

# Tuned-only: reduce stability shaping pressure and remove UMA force clipping.
TUNED_EXPORTS="${COMMON_EXPORTS}"
TUNED_EXPORTS+=",STABILITY_WEIGHT=1.0"
TUNED_EXPORTS+=",ATOM_STABILITY_WEIGHT=5.0"
TUNED_EXPORTS+=",VALENCE_UNDERBOND_WEIGHT=0.20"
TUNED_EXPORTS+=",VALENCE_OVERBOND_WEIGHT=0.50"
TUNED_EXPORTS+=",VALENCE_UNDERBOND_SOFT_WEIGHT=0.05"
TUNED_EXPORTS+=",VALENCE_OVERBOND_SOFT_WEIGHT=0.15"
TUNED_EXPORTS+=",VALENCE_SOFT_TEMPERATURE=0.02"
TUNED_EXPORTS+=",FORCE_CLIP_THRESHOLD=null"

RUN_NAME_TUNED="geom_pretrained_tunedonly_${MODEL_TAG}_lowvar_seed${SEED_TUNED}_${timestamp}"
echo "[submit] tuned-only: ${RUN_NAME_TUNED}"
sbatch_out_1="$(sbatch --export=ALL,${TUNED_EXPORTS},SEED=${SEED_TUNED},RUN_NAME=${RUN_NAME_TUNED} post_train_diffusion_geom.sh)"
echo "${sbatch_out_1}"

# Mindist: same settings + min-pair-distance penalty.
RUN_NAME_MINDIST="geom_pretrained_mindist_${MODEL_TAG}_lowvar_seed${SEED_MINDIST}_${timestamp}"
echo "[submit] mindist-guard: ${RUN_NAME_MINDIST}"
sbatch_out_2="$(sbatch --export=ALL,${TUNED_EXPORTS},SEED=${SEED_MINDIST},RUN_NAME=${RUN_NAME_MINDIST},MIN_PAIR_DIST_THRESHOLD=0.9,MIN_PAIR_DIST_PENALTY_WEIGHT=20.0,MIN_PAIR_DIST_PENALTY_POWER=1 post_train_diffusion_geom.sh)"
echo "${sbatch_out_2}"
