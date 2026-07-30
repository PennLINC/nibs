#!/bin/bash
#SBATCH --job-name=spatial_qc
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32gb
#SBATCH --time=24:00:00
# Outputs ----------------------------------
#SBATCH --output=/cbica/projects/nibs/logs/spatial_qc_%j.out
#SBATCH --error=/cbica/projects/nibs/logs/spatial_qc_%j.err
# ------------------------------------------

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/cbica/projects/nibs}"
CODE_DIR="${CODE_DIR:-${PROJECT_ROOT}/code/analysis}"
PYTHON="${PYTHON:-${HOME}/.conda/envs/myelin_reliability/bin/python}"
OUTPUT="${OUTPUT:-${PROJECT_ROOT}/derivatives/spatial_qc/nibs_spatial_qc_report.pdf}"
WORK_DIR="${WORK_DIR:-${PROJECT_ROOT}/derivatives/spatial_qc/work}"

mkdir -p "$(dirname "${OUTPUT}")" "${WORK_DIR}"

# Keep package caches writable and limit libraries to the CPUs Slurm assigned.
export MPLCONFIGDIR="${WORK_DIR}/matplotlib"
export TMPDIR="${WORK_DIR}/tmp"
export ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS="${SLURM_CPUS_PER_TASK}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
mkdir -p "${MPLCONFIGDIR}" "${TMPDIR}"

if [[ ! -x "${PYTHON}" ]]; then
  echo "Python executable not found or not executable: ${PYTHON}" >&2
  exit 1
fi

SCRIPT="${CODE_DIR}/generate_spatial_qc_report.py"
if [[ ! -f "${SCRIPT}" ]]; then
  echo "Report script not found: ${SCRIPT}" >&2
  exit 1
fi

echo "Host: $(hostname)"
echo "Started: $(date)"
echo "Python: ${PYTHON}"
echo "Output: ${OUTPUT}"
echo "Work directory: ${WORK_DIR}"
echo "Additional arguments: $*"

"${PYTHON}" -u "${SCRIPT}" \
  --project-root "${PROJECT_ROOT}" \
  --output "${OUTPUT}" \
  --work-dir "${WORK_DIR}" \
  "$@"

echo "Finished: $(date)"
echo "Report: ${OUTPUT}"
