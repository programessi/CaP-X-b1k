#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

BEHAVIOR_ROOT="${BEHAVIOR_ROOT:-/home/xingshu/workspaces/fys/BEHAVIOR-1K}"
CONDA_BIN="${CONDA_BIN:-/home/xingshu/miniforge3/bin/conda}"
CONDA_ENV="${CONDA_ENV:-behavior}"
CODEX_BIN="${CODEX_BIN:-codex}"
CODEX_MODEL_PROVIDER="${CODEX_MODEL_PROVIDER:-axonhub}"
CODEX_PROXY_HOST="${CODEX_PROXY_HOST:-127.0.0.1}"
CODEX_PROXY_PORT="${CODEX_PROXY_PORT:-8120}"
CODEX_PROXY_TIMEOUT_SECONDS="${CODEX_PROXY_TIMEOUT_SECONDS:-180}"
MODEL="${MODEL:-codex-a}"
SERVER_URL="${SERVER_URL:-http://${CODEX_PROXY_HOST}:${CODEX_PROXY_PORT}/chat/completions}"
TIMEOUT_SECONDS="${TIMEOUT_SECONDS:-1200}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
OMNIGIBSON_GPU_ID="${OMNIGIBSON_GPU_ID:-0}"
OUTPUT_DIR="${OUTPUT_DIR:-./outputs/x2_pick_place_red_cube_two_targets_codex_a_non_oracle_smoke}"
VISUAL_ARTIFACT_DIR="${VISUAL_ARTIFACT_DIR:-${REPO_ROOT}/outputs/x2_visual_artifacts/two_targets_codex_a_non_oracle_smoke}"

cd "${REPO_ROOT}"

PROXY_PID=""
cleanup() {
  if [[ -n "${PROXY_PID}" ]] && kill -0 "${PROXY_PID}" 2>/dev/null; then
    echo "[x2-codex-a-smoke] stopping codex proxy pid=${PROXY_PID}"
    kill "${PROXY_PID}" 2>/dev/null || true
    wait "${PROXY_PID}" 2>/dev/null || true
  fi
}
trap cleanup EXIT

echo "[x2-codex-a-smoke] repo: ${REPO_ROOT}"
echo "[x2-codex-a-smoke] behavior root: ${BEHAVIOR_ROOT}"
echo "[x2-codex-a-smoke] conda env: ${CONDA_ENV}"
echo "[x2-codex-a-smoke] model provider: ${CODEX_MODEL_PROVIDER}"
echo "[x2-codex-a-smoke] server url: ${SERVER_URL}"
echo "[x2-codex-a-smoke] output dir: ${OUTPUT_DIR}"
echo "[x2-codex-a-smoke] visual artifact dir: ${VISUAL_ARTIFACT_DIR}"

"${CONDA_BIN}" run --no-capture-output -n "${CONDA_ENV}" \
  python capx/serving/codex_cli_server.py \
    --host "${CODEX_PROXY_HOST}" \
    --port "${CODEX_PROXY_PORT}" \
    --codex-bin "${CODEX_BIN}" \
    --model-provider "${CODEX_MODEL_PROVIDER}" \
    --cwd "${REPO_ROOT}" \
    --timeout-s "${CODEX_PROXY_TIMEOUT_SECONDS}" &
PROXY_PID="$!"

echo "[x2-codex-a-smoke] started codex proxy pid=${PROXY_PID}"
for _ in $(seq 1 60); do
  if curl -sS --max-time 2 "http://${CODEX_PROXY_HOST}:${CODEX_PROXY_PORT}/health" >/dev/null 2>&1; then
    echo "[x2-codex-a-smoke] codex proxy ready"
    break
  fi
  sleep 1
done

if ! curl -sS --max-time 2 "http://${CODEX_PROXY_HOST}:${CODEX_PROXY_PORT}/health" >/dev/null; then
  echo "[x2-codex-a-smoke] codex proxy did not become ready" >&2
  exit 1
fi

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
    --config-path env_configs/x2/x2_pick_place_red_cube_two_targets.yaml \
    --use-oracle-code False \
    --record-video True \
    --total-trials 1 \
    --num-workers 1 \
    --model "${MODEL}" \
    --server-url "${SERVER_URL}" \
    --output-dir "${OUTPUT_DIR}" \
    --debug True
