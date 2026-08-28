#!/usr/bin/env bash
#SBATCH --job-name=mni_gmwm_effects
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=12gb
#SBATCH --time=24:00:00
#SBATCH --output=/cbica/projects/nibs/logs/mni_gmwm_effects_%A.out
#SBATCH --error=/cbica/projects/nibs/logs/mni_gmwm_effects_%A.err

pwd; hostname; date
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-/cbica/projects/nibs}"
CODE_DIR="${CODE_DIR:-${SCRIPT_DIR}}"
LOGS_DIR="${LOGS_DIR:-${PROJECT_ROOT}/logs}"
PYTHON_BIN="${PYTHON_BIN:-${HOME}/.conda/envs/myelin_reliability/bin/python}"
ANALYSIS_SET="${ANALYSIS_SET:-primary}"
EFFECT="${EFFECT:-robust_median_d}"
GM_TISSUE="${GM_TISSUE:-cortical_gm}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/derivatives/mni_gm_wm_effect_sizes}"
FIGURE_DIR="${FIGURE_DIR:-${PROJECT_ROOT}/figures/gm_wm_effect_sizes}"
RECALCULATE="${RECALCULATE:-0}"
SUBJECT_TSV="${OUTPUT_DIR}/mni_gm_wm_effect_sizes_${ANALYSIS_SET}_subject.tsv"

mkdir -p "${LOGS_DIR}/jobs"

if [[ "${RECALCULATE}" == "1" || ! -s "${SUBJECT_TSV}" ]] || ! head -n 1 "${SUBJECT_TSV}" | grep -q 'gm_tissue'; then
  "${PYTHON_BIN}" "${CODE_DIR}/compute_mni_gm_wm_effect_sizes.py" \
    --project-root "${PROJECT_ROOT}" \
    --analysis-set "${ANALYSIS_SET}" \
    --output-dir "${OUTPUT_DIR}" \
    "$@"
else
  echo "Using cached subject-level effect sizes: ${SUBJECT_TSV}"
  echo "Set RECALCULATE=1 to recompute voxelwise GM/WM effects."
fi

"${PYTHON_BIN}" "${CODE_DIR}/plot_gm_wm_effect_sizes.py" \
  --input "${SUBJECT_TSV}" \
  --effect "${EFFECT}" \
  --gm-tissue "${GM_TISSUE}" \
  --output "${FIGURE_DIR}/gm_wm_effect_sizes_${ANALYSIS_SET}_${EFFECT}"

exitcode=$?
job_id="${SLURM_JOB_ID:-local}"
echo "all_subjects   ${job_id}    ${exitcode}" \
  >> "${LOGS_DIR}/jobs/${SLURM_JOB_NAME:-mni_gmwm_effects}.${job_id}.tsv"
echo "Finished MNI GM/WM effect-size job with exit code ${exitcode}"
exit "${exitcode}"
