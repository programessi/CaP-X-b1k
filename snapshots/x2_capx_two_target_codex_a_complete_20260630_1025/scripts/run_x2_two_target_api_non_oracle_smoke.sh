#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

BEHAVIOR_ROOT="${BEHAVIOR_ROOT:-/home/xingshu/workspaces/fys/BEHAVIOR-1K}"
CONDA_BIN="${CONDA_BIN:-/home/xingshu/miniforge3/bin/conda}"
CONDA_ENV="${CONDA_ENV:-behavior}"
MODEL="${MODEL:-gpt-5}"
SERVER_URL="${SERVER_URL:-http://127.0.0.1:8110/chat/completions}"
TIMEOUT_SECONDS="${TIMEOUT_SECONDS:-1200}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
OMNIGIBSON_GPU_ID="${OMNIGIBSON_GPU_ID:-0}"
OUTPUT_DIR="${OUTPUT_DIR:-./outputs/x2_pick_place_red_cube_two_targets_api_non_oracle_smoke}"
VISUAL_ARTIFACT_DIR="${VISUAL_ARTIFACT_DIR:-${REPO_ROOT}/outputs/x2_visual_artifacts/two_targets_api_non_oracle_smoke}"
CONFIG_PATH="${CONFIG_PATH:-env_configs/x2/x2_pick_place_red_cube_two_targets.yaml}"

cd "${REPO_ROOT}"
mkdir -p /tmp/isaac-sim/apps

echo "[x2-api-smoke] repo: ${REPO_ROOT}"
echo "[x2-api-smoke] behavior root: ${BEHAVIOR_ROOT}"
echo "[x2-api-smoke] conda env: ${CONDA_ENV}"
echo "[x2-api-smoke] model: ${MODEL}"
echo "[x2-api-smoke] server url: ${SERVER_URL}"
echo "[x2-api-smoke] config path: ${CONFIG_PATH}"
echo "[x2-api-smoke] output dir: ${OUTPUT_DIR}"
echo "[x2-api-smoke] visual artifact dir: ${VISUAL_ARTIFACT_DIR}"

timeout "${TIMEOUT_SECONDS}" env \
  PYTHONNOUSERSITE=1 \
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
  OMNIGIBSON_GPU_ID="${OMNIGIBSON_GPU_ID}" \
  PYTHONPATH="${BEHAVIOR_ROOT}/OmniGibson:${BEHAVIOR_ROOT}/bddl3:${REPO_ROOT}" \
  LD_LIBRARY_PATH="/home/xingshu/miniforge3/envs/${CONDA_ENV}/lib" \
  EXP_PATH=/tmp/isaac-sim/apps \
  OMNI_KIT_ACCEPT_EULA=YES \
  OMNIGIBSON_HEADLESS=1 \
  OMNIGIBSON_NO_OMNI_LOGS=1 \
  MPLCONFIGDIR=/tmp/og_mpl \
  NUMBA_CACHE_DIR=/tmp/numba-cache \
  HF_HUB_OFFLINE=1 \
  CAPX_FAST_EXIT_AFTER_MAIN=1 \
  CAPX_X2_VISUAL_ARTIFACT_DIR="${VISUAL_ARTIFACT_DIR}" \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  "${CONDA_BIN}" run --no-capture-output -n "${CONDA_ENV}" \
  python capx/envs/launch.py \
    --config-path "${CONFIG_PATH}" \
    --use-oracle-code False \
    --record-video True \
    --total-trials 1 \
    --num-workers 1 \
    --model "${MODEL}" \
    --server-url "${SERVER_URL}" \
    --output-dir "${OUTPUT_DIR}" \
    --debug True
