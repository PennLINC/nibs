#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-/cbica/projects/nibs}"
ANALYSIS_SET="${ANALYSIS_SET:-primary}"
EFFECT="${EFFECT:-robust_median_d}"
GM_TISSUE="${GM_TISSUE:-cortical_gm}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/derivatives/mni_gm_wm_effect_sizes}"
FIGURE_DIR="${FIGURE_DIR:-${PROJECT_ROOT}/figures/gm_wm_effect_sizes}"
RECALCULATE="${RECALCULATE:-0}"
SUBJECT_TSV="${OUTPUT_DIR}/mni_gm_wm_effect_sizes_${ANALYSIS_SET}_subject.tsv"

if [[ "${RECALCULATE}" == "1" || ! -s "${SUBJECT_TSV}" ]] || ! head -n 1 "${SUBJECT_TSV}" | grep -q 'gm_tissue'; then
  python "${SCRIPT_DIR}/compute_mni_gm_wm_effect_sizes.py" \
    --project-root "${PROJECT_ROOT}" \
    --analysis-set "${ANALYSIS_SET}" \
    --output-dir "${OUTPUT_DIR}" \
    "$@"
else
  echo "Using cached subject-level effect sizes: ${SUBJECT_TSV}"
  echo "Set RECALCULATE=1 to recompute voxelwise GM/WM effects."
fi

python "${SCRIPT_DIR}/plot_gm_wm_effect_sizes.py" \
  --input "${SUBJECT_TSV}" \
  --effect "${EFFECT}" \
  --gm-tissue "${GM_TISSUE}" \
  --output "${FIGURE_DIR}/gm_wm_effect_sizes_${ANALYSIS_SET}_${EFFECT}"
