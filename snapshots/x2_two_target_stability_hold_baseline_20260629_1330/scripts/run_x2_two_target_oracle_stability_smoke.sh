#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

BEHAVIOR_ROOT="${BEHAVIOR_ROOT:-/home/xingshu/workspaces/fys/BEHAVIOR-1K}"
CONDA_BIN="${CONDA_BIN:-/home/xingshu/miniforge3/bin/conda}"
CONDA_ENV="${CONDA_ENV:-behavior}"
REPEATS="${REPEATS:-2}"
RUN_RIGHT="${RUN_RIGHT:-1}"
RUN_LEFT="${RUN_LEFT:-1}"
STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
TIMEOUT_SECONDS="${TIMEOUT_SECONDS:-1200}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
OMNIGIBSON_GPU_ID="${OMNIGIBSON_GPU_ID:-0}"

RIGHT_CONFIG="${RIGHT_CONFIG:-env_configs/x2/x2_pick_place_red_cube_two_targets.yaml}"
LEFT_CONFIG="${LEFT_CONFIG:-env_configs/x2/x2_pick_place_red_cube_two_targets_left.yaml}"

cd "${REPO_ROOT}"
mkdir -p /tmp/isaac-sim/apps

run_one() {
  local side="$1"
  local config_path="$2"
  local idx="$3"
  local run_name="two_targets_${side}_oracle_stability_${STAMP}_run$(printf '%02d' "${idx}")"

  echo "[x2-oracle-stability] side=${side} run=${idx}/${REPEATS}"
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
    CAPX_X2_VISUAL_ARTIFACT_DIR="${REPO_ROOT}/outputs/x2_visual_artifacts/stability/${run_name}" \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    "${CONDA_BIN}" run --no-capture-output -n "${CONDA_ENV}" \
    python capx/envs/launch.py \
      --config-path "${config_path}" \
      --use-oracle-code True \
      --record-video True \
      --total-trials 1 \
      --num-workers 1 \
      --output-dir "./outputs/oracle/stability/${run_name}" \
      --debug True
}

for idx in $(seq 1 "${REPEATS}"); do
  if [[ "${RUN_RIGHT}" == "1" ]]; then
    run_one "right" "${RIGHT_CONFIG}" "${idx}"
  fi
  if [[ "${RUN_LEFT}" == "1" ]]; then
    run_one "left" "${LEFT_CONFIG}" "${idx}"
  fi
done

echo "[x2-oracle-stability] done stamp=${STAMP}"
