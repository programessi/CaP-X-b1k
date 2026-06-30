#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

STAMP="${STAMP:-manual_direct_api_$(date +%Y%m%d_%H%M%S)}"
REPEATS="${REPEATS:-1}"

export STAMP
export REPEATS

cd "${REPO_ROOT}"

echo "[x2-api-stability-check] stamp=${STAMP}"
echo "[x2-api-stability-check] repeats=${REPEATS}"

"${SCRIPT_DIR}/run_x2_two_target_api_stability_smoke.sh"

python scripts/summarize_x2_runs.py outputs/stability/two_targets_*_api_stability_"${STAMP}"_run*
python scripts/check_x2_acceptance.py outputs/stability/two_targets_*_api_stability_"${STAMP}"_run*
