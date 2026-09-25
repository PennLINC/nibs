#!/usr/bin/env bash
#SBATCH --job-name=g_ratio_aggregate
#SBATCH --cpus-per-task=1
#SBATCH --mem=8gb
#SBATCH --time=02:00:00

set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
CODE_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"
export MIRROR_CODE_ROOT="${MIRROR_CODE_ROOT:-${CODE_ROOT}}"
PYTHON_BIN="${PYTHON_BIN:-python}"
source "${MIRROR_CODE_ROOT}/configuration/load_profile.sh"
"${PYTHON_BIN}" -u "${SCRIPT_DIR}/aggregate_g_ratio_scaling_factors.py" "$@"
