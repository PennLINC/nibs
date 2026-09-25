#!/bin/bash
#SBATCH --job-name=prep_mni_ribbon
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16gb
#SBATCH --time=24:00:00
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export MIRROR_CODE_ROOT="${MIRROR_CODE_ROOT:-$(cd "${SCRIPT_DIR}/../../.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-python}"
source "${MIRROR_CODE_ROOT}/configuration/load_profile.sh"
QSIPREP_CONTAINER="${QSIPREP_CONTAINER:-${MIRROR_APPTAINER_QSIPREP}}"

if [[ ! -f "${QSIPREP_CONTAINER}" ]]; then
  echo "QSIPrep container not found: ${QSIPREP_CONTAINER}" >&2
  exit 1
fi

bash "${SCRIPT_DIR}/prepare_mni_ribbon_masks.sh" \
  --project-root "${MIRROR_PROJECT_ROOT}" \
  --source-derivatives-dir "${MIRROR_SOURCE_DERIVATIVES}" \
  --run-derivatives-dir "${MIRROR_OUTPUT_DERIVATIVES}" \
  --qsiprep-container "${QSIPREP_CONTAINER}" \
  "$@"
