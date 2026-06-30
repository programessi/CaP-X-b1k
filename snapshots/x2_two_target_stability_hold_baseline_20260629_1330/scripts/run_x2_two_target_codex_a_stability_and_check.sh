#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

STAMP="${STAMP:-manual_codex_a_$(date +%Y%m%d_%H%M%S)}"
REPEATS="${REPEATS:-1}"

export STAMP
export REPEATS

cd "${REPO_ROOT}"

echo "[x2-codex-a-stability-check] stamp=${STAMP}"
echo "[x2-codex-a-stability-check] repeats=${REPEATS}"

"${SCRIPT_DIR}/run_x2_two_target_codex_a_stability_smoke.sh"

shopt -s nullglob
RUN_DIRS=(
  outputs/stability/two_targets_*_codex_a_stability_"${STAMP}"_run*
  outputs/stability/*/two_targets_*_codex_a_stability_"${STAMP}"_run*
)
shopt -u nullglob

if [[ "${#RUN_DIRS[@]}" -eq 0 ]]; then
  echo "[x2-codex-a-stability-check] no matching run dirs for stamp=${STAMP}" >&2
  exit 1
fi

python scripts/summarize_x2_runs.py "${RUN_DIRS[@]}"
python scripts/check_x2_acceptance.py "${RUN_DIRS[@]}"
