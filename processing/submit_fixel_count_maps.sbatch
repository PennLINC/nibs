#!/bin/bash
#SBATCH --job-name=fixel_count
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --array=1-26
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=4gb
#SBATCH --time=12:00:00
# Outputs ----------------------------------
#SBATCH --output=/cbica/projects/nibs/logs/fixel_count_%A-%a.out
#SBATCH --error=/cbica/projects/nibs/logs/fixel_count_%A-%a.err
# ------------------------------------------

pwd; hostname; date
set -euo pipefail

BIDS_DIR="/cbica/projects/nibs/dset"
CODE_DIR="/cbica/projects/nibs/code/processing"
DERIVS_DIR="${HOME}/derivatives"
APPTAINER_IMAGE="${HOME}/apptainer/qsirecon-26.0.0.sif"

subject=$(
    sed -n -E "$((${SLURM_ARRAY_TASK_ID} + 1))s/sub-(\S*)\>.*/\1/gp" \
        "${BIDS_DIR}/participants.tsv"
)

python "${CODE_DIR}/create_fixel_count_maps.py" \
    --derivatives-dir "${DERIVS_DIR}" \
    --output-dir "${DERIVS_DIR}/fixel_count" \
    --work-dir "${DERIVS_DIR}/fixel_count/work" \
    --apptainer-image "${APPTAINER_IMAGE}" \
    --subject-id "${subject}" \
    --nthreads "${SLURM_CPUS_PER_TASK}"
