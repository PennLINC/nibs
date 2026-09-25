#!/bin/bash
#SBATCH --job-name=parcel_corr
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16gb
#SBATCH --time=24:00:00
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export MIRROR_CODE_ROOT="${MIRROR_CODE_ROOT:-$(cd "${SCRIPT_DIR}/../../.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-python}"
source "${MIRROR_CODE_ROOT}/configuration/load_profile.sh"
PYTHON="${PYTHON:-${PYTHON_BIN}}"
ANALYSIS="${ANALYSIS:-both}"
ANALYSIS_SET="${ANALYSIS_SET:-primary}"
QC_MODE="${QC_MODE:-metricqc}"
STAT="${STAT:-median}"
MIN_FEATURES="${MIN_FEATURES:-10}"
MIN_GROUP_SUBJECTS="${MIN_GROUP_SUBJECTS:-1}"
PREFER_MASKED="${PREFER_MASKED:-1}"

ARGS=(
  --patterns-file "${MIRROR_CODE_ROOT}/configuration/metrics.yml"
  --qc-file "${MIRROR_CODE_ROOT}/data/qc/manual_qc_modality.tsv"
  --outdir "${MIRROR_OUTPUT_DERIVATIVES}/parcel_bundle_correlations"
  --analysis "${ANALYSIS}"
  --analysis-set "${ANALYSIS_SET}"
  --qc-mode "${QC_MODE}"
  --stat "${STAT}"
  --min-features "${MIN_FEATURES}"
  --min-group-subjects "${MIN_GROUP_SUBJECTS}"
)

if [[ "${PREFER_MASKED}" == "1" ]]; then
  ARGS+=(--prefer-masked)
fi

"${PYTHON}" -u "${SCRIPT_DIR}/compute_parcel_bundle_correlations.py" \
  "${ARGS[@]}"
