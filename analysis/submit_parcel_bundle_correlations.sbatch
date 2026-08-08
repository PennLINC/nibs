#!/bin/bash
#SBATCH --job-name=parcel_corr
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16gb
#SBATCH --time=24:00:00
#SBATCH --output=/cbica/projects/nibs/logs/parcel_corr_%A.out
#SBATCH --error=/cbica/projects/nibs/logs/parcel_corr_%A.err

pwd; hostname; date
set -euo pipefail

PROJECT_ROOT="/cbica/projects/nibs"
CODE_DIR="${PROJECT_ROOT}/code/analysis"
PYTHON="${PYTHON:-${HOME}/.conda/envs/myelin_reliability/bin/python}"
ANALYSIS="${ANALYSIS:-both}"
QC_MODE="${QC_MODE:-metricqc}"
STAT="${STAT:-median}"
MIN_FEATURES="${MIN_FEATURES:-10}"
MIN_GROUP_SUBJECTS="${MIN_GROUP_SUBJECTS:-1}"
PREFER_MASKED="${PREFER_MASKED:-1}"
PLOT_ONLY="${PLOT_ONLY:-0}"

ARGS=(
  --patterns-file "${PROJECT_ROOT}/code/configuration/patterns.json"
  --qc-file "${PROJECT_ROOT}/code/data/manual_qc_modality.tsv"
  --outdir "${PROJECT_ROOT}/derivatives/parcel_bundle_correlations"
  --analysis "${ANALYSIS}"
  --qc-mode "${QC_MODE}"
  --stat "${STAT}"
  --min-features "${MIN_FEATURES}"
  --min-group-subjects "${MIN_GROUP_SUBJECTS}"
)

if [[ "${PREFER_MASKED}" == "1" ]]; then
  ARGS+=(--prefer-masked)
fi

if [[ "${PLOT_ONLY}" == "1" ]]; then
  ARGS+=(--plot-only)
fi

"${PYTHON}" -u "${CODE_DIR}/compute_parcel_bundle_correlations.py" \
  "${ARGS[@]}"
