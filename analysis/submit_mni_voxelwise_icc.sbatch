#!/bin/bash
#SBATCH --job-name=mni_icc
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64gb
#SBATCH --time=48:00:00
# Outputs ----------------------------------
#SBATCH --output=/cbica/projects/nibs/logs/mni_icc_%j.out
#SBATCH --error=/cbica/projects/nibs/logs/mni_icc_%j.err
# ------------------------------------------

pwd; hostname; date
set -euo pipefail

PROJECT_ROOT="/cbica/projects/nibs"
CODE_DIR="${PROJECT_ROOT}/code/analysis"
PYTHON="${PYTHON:-${HOME}/.conda/envs/myelin_reliability/bin/python}"
if [[ ! -x "${PYTHON}" ]]; then
  echo "Python executable not found or not executable: ${PYTHON}" >&2
  exit 1
fi


"${PYTHON}" -u "${CODE_DIR}/compute_mni_voxelwise_icc.py" \
    --project-root "${PROJECT_ROOT}" \
    --derivatives-dir "${PROJECT_ROOT}/derivatives" \
    --patterns-file "${PROJECT_ROOT}/code/configuration/patterns.json" \
    --qc-file "${PROJECT_ROOT}/code/data/manual_qc_modality.tsv" \
    --gm-probseg "${PROJECT_ROOT}/code/data/tpl-MNI152NLin2009cAsym_res-01_label-GM_probseg.nii.gz" \
    --wm-probseg "${PROJECT_ROOT}/code/data/tpl-MNI152NLin2009cAsym_res-01_label-WM_probseg.nii.gz" \
    --gm-threshold 0.5 \
    --wm-threshold 0.5 \
    --output-dir "${PROJECT_ROOT}/derivatives/mni_voxelwise_icc" \
    --work-dir "${PROJECT_ROOT}/derivatives/mni_voxelwise_icc/work"
