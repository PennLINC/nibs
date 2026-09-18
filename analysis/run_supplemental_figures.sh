#!/usr/bin/env bash
# Render the supplemental bundle table and full-metric figure set from existing derivatives.

set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"

"${PYTHON_BIN}" analysis/make_autotrack_bundle_html_table.py

"${PYTHON_BIN}" analysis/plot_gm_wm_effect_sizes.py \
    --analysis-set full \
    --gm-tissue cortical_gm \
    --effect robust_median_d

# Each domain is written as its own figure and clustered independently.
"${PYTHON_BIN}" analysis/plot_correlation_matrices.py \
    --analysis-set full \
    --tissue both \
    --mni-correlation pearson \
    --parcel-correlation pearson \
    --parcel-stat median \
    --strict

"${PYTHON_BIN}" analysis/plot_supplemental_icc_figures.py \
    --analysis-set full \
    --stat median \
    --icc-column ICC2_1 \
    --strict

"${PYTHON_BIN}" analysis/plot_supplemental_discriminability_heatmap.py \
    --analysis-set full \
    --stat median \
    --distance-metric correlation \
    --score-column discriminability \
    --category-level family
