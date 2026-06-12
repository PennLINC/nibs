#!/bin/bash
#SBATCH --job-name=qsm
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --array=1,4,18,19
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=20gb
#SBATCH --time=48:00:00
# Outputs ----------------------------------
#SBATCH --output=/cbica/projects/nibs/logs/qsm_%A-%a.out
#SBATCH --error=/cbica/projects/nibs/logs/qsm_%A-%a.err
# ------------------------------------------
#
# Run SEPIA + chi-separation on CUBIC after process_qsm_prep.py.
# Submit from anywhere with the processing conda env active, e.g.:
#   sbatch /cbica/projects/nibs/code/processing/run_qsm.sbatch

pwd; hostname; date
set -euo pipefail

module load matlab/R2023A
echo "Using MATLAB at: $(which matlab)"

SCRIPT_DIR="/cbica/projects/nibs/code/processing"
cd "${SCRIPT_DIR}"

BIDS_DIR="/cbica/projects/nibs/dset"
subject=$(sed -n -E "$((${SLURM_ARRAY_TASK_ID} + 1))s/sub-(\S*)\>.*/\1/gp" "${BIDS_DIR}/participants.tsv")

if [[ -z "${subject}" ]]; then
    echo "No subject found for SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID}" >&2
    exit 1
fi

python process_qsm.py --subject-id "${subject}"
