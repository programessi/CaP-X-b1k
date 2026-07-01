#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"

CONFIG_PATH="${CONFIG_PATH:-env_configs/x2/x2_pick_place_two_objects_blue_right_rgbd_visual.yaml}"
OUTPUT_DIR="${OUTPUT_DIR:-./outputs/x2_pick_place_two_objects_blue_right_rgbd_visual_codex_a_non_oracle_${STAMP}}"
VISUAL_ARTIFACT_DIR="${VISUAL_ARTIFACT_DIR:-${REPO_ROOT}/outputs/x2_visual_artifacts/two_objects_blue_right_rgbd_visual_codex_a_non_oracle_${STAMP}}"
TIMEOUT_SECONDS="${TIMEOUT_SECONDS:-1200}"

cd "${REPO_ROOT}"

CONFIG_PATH="${CONFIG_PATH}" \
OUTPUT_DIR="${OUTPUT_DIR}" \
VISUAL_ARTIFACT_DIR="${VISUAL_ARTIFACT_DIR}" \
TIMEOUT_SECONDS="${TIMEOUT_SECONDS}" \
"${SCRIPT_DIR}/run_x2_two_target_codex_a_non_oracle_smoke.sh"
