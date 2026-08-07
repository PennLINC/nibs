#!/bin/bash
#SBATCH --job-name=fiber_r1_dki
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32gb
#SBATCH --time=12:00:00
# Outputs ----------------------------------
#SBATCH --output=/cbica/projects/nibs/logs/fiber_r1_dki_%j.out
#SBATCH --error=/cbica/projects/nibs/logs/fiber_r1_dki_%j.err
# ------------------------------------------

pwd; hostname; date
set -euo pipefail

CODE_DIR="/cbica/projects/nibs/code/analysis"
DERIVS_DIR="${HOME}/derivatives"

python "${CODE_DIR}/test_fiber_complexity_r1_dki.py" \
    --derivatives-dir "${DERIVS_DIR}" \
    --fixel-count-dir "${DERIVS_DIR}/fixel_count" \
    --output-dir "${DERIVS_DIR}/fixel_count_analysis" \
    --work-dir "${DERIVS_DIR}/fixel_count_analysis/work"
