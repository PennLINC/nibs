#!/bin/bash
#SBATCH --job-name=parcel_icc
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16gb
#SBATCH --time=24:00:00
#SBATCH --output=/cbica/projects/nibs/logs/parcel_icc_%A.out
#SBATCH --error=/cbica/projects/nibs/logs/parcel_icc_%A.err

pwd; hostname; date
set -euo pipefail

PROJECT_ROOT="/cbica/projects/nibs"
CODE_DIR="${PROJECT_ROOT}/code/analysis"
PYTHON="${PYTHON:-${HOME}/.conda/envs/myelin_reliability/bin/python}"

"${PYTHON}" -u "${CODE_DIR}/compute_parcel_bundle_icc.py" \
  --patterns-file "${PROJECT_ROOT}/code/configuration/patterns.json" \
  --qc-file "${PROJECT_ROOT}/code/data/manual_qc_modality.tsv" \
  --outdir "${PROJECT_ROOT}/derivatives/parcel_bundle_icc" \
  --analysis both \
  --qc-mode metricqc completeqc \
  --stat median \
  --prefer-masked
