#!/bin/bash
#SBATCH --job-name=parcel_disc
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

"${PYTHON}" -u "${SCRIPT_DIR}/compute_parcel_bundle_discriminability.py" \
  --patterns-file "${MIRROR_CODE_ROOT}/configuration/metrics.yml" \
  --qc-file "${MIRROR_CODE_ROOT}/data/qc/manual_qc_modality.tsv" \
  --outdir "${MIRROR_OUTPUT_DERIVATIVES}/parcel_bundle_discriminability" \
  --analysis both \
  --qc-mode metricqc \
  --stat median \
  --prefer-masked \
  --distance-metric correlation
