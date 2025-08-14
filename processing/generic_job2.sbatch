#!/bin/bash
#SBATCH --job-name=nibs
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --array=5-14
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=20gb
#SBATCH --time=24:00:00
# Outputs ----------------------------------
#SBATCH --output=/cbica/projects/nibs/code/processing/jobs/job_%A-%a.out
#SBATCH --error=/cbica/projects/nibs/code/processing/jobs/job_%A-%a.err
# ------------------------------------------

pwd; hostname; date
set -e

# Parse the participants.tsv file and extract one subject ID from the line corresponding to this SLURM task.
BIDS_DIR="/cbica/projects/nibs/dset"
subject=$( sed -n -E "$((${SLURM_ARRAY_TASK_ID} + 1))s/sub-(\S*)\>.*/\1/gp" ${BIDS_DIR}/participants.tsv )

python generate_myelin_reports.py --subject-id ${subject}
