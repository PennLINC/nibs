#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-/cbica/projects/nibs}"
ANALYSIS_SET="${ANALYSIS_SET:-primary}"
EFFECT="${EFFECT:-robust_median_d}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/derivatives/mni_gm_wm_effect_sizes}"
FIGURE_DIR="${FIGURE_DIR:-${PROJECT_ROOT}/figures/gm_wm_effect_sizes}"

python "${SCRIPT_DIR}/compute_mni_gm_wm_effect_sizes.py" \
  --project-root "${PROJECT_ROOT}" \
  --analysis-set "${ANALYSIS_SET}" \
  --output-dir "${OUTPUT_DIR}" \
  "$@"

python "${SCRIPT_DIR}/plot_gm_wm_effect_sizes.py" \
  --input "${OUTPUT_DIR}/mni_gm_wm_effect_sizes_${ANALYSIS_SET}_subject.tsv" \
  --effect "${EFFECT}" \
  --output "${FIGURE_DIR}/gm_wm_effect_sizes_${ANALYSIS_SET}_${EFFECT}"
