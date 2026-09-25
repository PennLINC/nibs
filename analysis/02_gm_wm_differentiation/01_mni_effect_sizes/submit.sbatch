#!/usr/bin/env bash
#SBATCH --job-name=mni_gmwm_effects
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=12gb
#SBATCH --time=24:00:00
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export MIRROR_CODE_ROOT="${MIRROR_CODE_ROOT:-$(cd "${SCRIPT_DIR}/../../.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-python}"
source "${MIRROR_CODE_ROOT}/configuration/load_profile.sh"
LOGS_DIR="${MIRROR_LOGS_DIR}"
ANALYSIS_SET="${ANALYSIS_SET:-primary}"
OUTPUT_DIR="${OUTPUT_DIR:-${MIRROR_OUTPUT_DERIVATIVES}/mni_gm_wm_effect_sizes}"
RECALCULATE="${RECALCULATE:-1}"
SUBJECT_TSV="${OUTPUT_DIR}/mni_gm_wm_effect_sizes_${ANALYSIS_SET}_subject.tsv"

mkdir -p "${LOGS_DIR}/jobs"

if [[ "${RECALCULATE}" == "1" || ! -s "${SUBJECT_TSV}" ]] || ! head -n 1 "${SUBJECT_TSV}" | grep -q 'gm_tissue'; then
  "${PYTHON_BIN}" "${SCRIPT_DIR}/compute_mni_gm_wm_effect_sizes.py" \
    --project-root "${MIRROR_PROJECT_ROOT}" \
    --derivatives-dir "${MIRROR_SOURCE_DERIVATIVES}" \
    --run-derivatives-dir "${MIRROR_OUTPUT_DERIVATIVES}" \
    --analysis-set "${ANALYSIS_SET}" \
    --output-dir "${OUTPUT_DIR}" \
    "$@"
else
  echo "Using cached subject-level effect sizes: ${SUBJECT_TSV}"
  echo "Set RECALCULATE=1 to recompute voxelwise GM/WM effects."
fi

exitcode=0
job_id="${SLURM_JOB_ID:-local}"
echo "all_subjects   ${job_id}    ${exitcode}" \
  >> "${LOGS_DIR}/jobs/${SLURM_JOB_NAME:-mni_gmwm_effects}.${job_id}.tsv"
echo "Finished MNI GM/WM effect-size job with exit code ${exitcode}"
exit "${exitcode}"
