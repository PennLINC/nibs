#!/bin/bash
#SBATCH --job-name=mni_icc
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64gb
#SBATCH --time=48:00:00
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export MIRROR_CODE_ROOT="${MIRROR_CODE_ROOT:-$(cd "${SCRIPT_DIR}/../../.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-python}"
source "${MIRROR_CODE_ROOT}/configuration/load_profile.sh"
PYTHON="${PYTHON:-${PYTHON_BIN}}"
ANALYSIS_SET="${ANALYSIS_SET:-primary}"
if [[ ! -x "${PYTHON}" ]]; then
  echo "Python executable not found or not executable: ${PYTHON}" >&2
  exit 1
fi


"${PYTHON}" -u "${SCRIPT_DIR}/compute_mni_voxelwise_icc.py" \
    --project-root "${MIRROR_PROJECT_ROOT}" \
    --derivatives-dir "${MIRROR_SOURCE_DERIVATIVES}" \
    --patterns-file "${MIRROR_CODE_ROOT}/configuration/metrics.yml" \
    --analysis-set "${ANALYSIS_SET}" \
    --qc-file "${MIRROR_CODE_ROOT}/data/qc/manual_qc_modality.tsv" \
    --template-dseg "${MIRROR_CODE_ROOT}/data/atlases/tpl-MNI152NLin2009cAsym_res-01_seg-aseg_dseg.nii.gz" \
    --output-dir "${MIRROR_OUTPUT_DERIVATIVES}/mni_voxelwise_icc" \
    --work-dir "${MIRROR_OUTPUT_DERIVATIVES}/mni_voxelwise_icc/work" \
    --force
