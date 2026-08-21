#!/bin/bash
#SBATCH --job-name=prep_mni_ribbon
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16gb
#SBATCH --time=24:00:00
# Outputs ----------------------------------
#SBATCH --output=/cbica/projects/nibs/logs/prep_mni_ribbon_%j.out
#SBATCH --error=/cbica/projects/nibs/logs/prep_mni_ribbon_%j.err
# ------------------------------------------

pwd; hostname; date
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/cbica/projects/nibs}"
CODE_DIR="${PROJECT_ROOT}/code/analysis"
QSIPREP_CONTAINER="${QSIPREP_CONTAINER:-${HOME}/apptainer/qsiprep-26.0.0.sif}"

if [[ ! -f "${QSIPREP_CONTAINER}" ]]; then
  echo "QSIPrep container not found: ${QSIPREP_CONTAINER}" >&2
  exit 1
fi

bash "${CODE_DIR}/prepare_mni_ribbon_masks.sh" \
  --project-root "${PROJECT_ROOT}" \
  --derivatives-dir "${PROJECT_ROOT}/derivatives" \
  --qsiprep-container "${QSIPREP_CONTAINER}" \
  "$@"
