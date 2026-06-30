#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

REPEATS="${REPEATS:-2}"
RUN_RIGHT="${RUN_RIGHT:-1}"
RUN_LEFT="${RUN_LEFT:-1}"
STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"

RIGHT_CONFIG="${RIGHT_CONFIG:-env_configs/x2/x2_pick_place_red_cube_two_targets.yaml}"
LEFT_CONFIG="${LEFT_CONFIG:-env_configs/x2/x2_pick_place_red_cube_two_targets_left.yaml}"

cd "${REPO_ROOT}"

run_one() {
  local side="$1"
  local config_path="$2"
  local idx="$3"
  local run_name="two_targets_${side}_codex_a_stability_${STAMP}_run$(printf '%02d' "${idx}")"

  echo "[x2-stability] side=${side} run=${idx}/${REPEATS}"
  CONFIG_PATH="${config_path}" \
  OUTPUT_DIR="./outputs/stability/${run_name}" \
  VISUAL_ARTIFACT_DIR="${REPO_ROOT}/outputs/x2_visual_artifacts/stability/${run_name}" \
  "${SCRIPT_DIR}/run_x2_two_target_codex_a_non_oracle_smoke.sh"
}

for idx in $(seq 1 "${REPEATS}"); do
  if [[ "${RUN_RIGHT}" == "1" ]]; then
    run_one "right" "${RIGHT_CONFIG}" "${idx}"
  fi
  if [[ "${RUN_LEFT}" == "1" ]]; then
    run_one "left" "${LEFT_CONFIG}" "${idx}"
  fi
done

echo "[x2-stability] done stamp=${STAMP}"
