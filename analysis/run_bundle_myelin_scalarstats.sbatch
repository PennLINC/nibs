#!/bin/bash
#SBATCH --job-name=bundle_myelin_stats
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --array=1-26
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=8gb
#SBATCH --time=24:00:00
# Outputs ----------------------------------
#SBATCH --output=/cbica/projects/nibs/logs/bundle_myelin_stats_%A-%a.out
#SBATCH --error=/cbica/projects/nibs/logs/bundle_myelin_stats_%A-%a.err
# ------------------------------------------

pwd; hostname; date
set -euo pipefail

BIDS_DIR="/cbica/projects/nibs/dset"
CODE_DIR="/cbica/projects/nibs/code/analysis"
LOGS_DIR="/cbica/projects/nibs/logs"
TCKMAP_BIN="${HOME}/.conda/envs/mrtrix3/bin/tckmap"

mkdir -p "${LOGS_DIR}/jobs"

# Parse participants.tsv line corresponding to this array index.
subject=$( sed -n -E "$((${SLURM_ARRAY_TASK_ID} + 1))s/sub-(\S*)\>.*/\1/gp" "${BIDS_DIR}/participants.tsv" )

if [[ -z "${subject}" ]]; then
  echo "No subject found for SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID}"
  exit 1
fi

cmd="TCKMAP_BIN=${TCKMAP_BIN} ~/.conda/envs/myelin_reliability/bin/python ${CODE_DIR}/generate_bundle_myelin_scalarstats.py \
  --subject-id ${subject} \
  --out-root /cbica/projects/nibs/derivatives/bundle_myelin_stats"

echo "Running task ${SLURM_ARRAY_TASK_ID} for sub-${subject}"
echo "Commandline: ${cmd}"
eval "${cmd}"
exitcode=$?

echo "sub-${subject}   ${SLURM_ARRAY_TASK_ID}    ${exitcode}" \
  >> "${LOGS_DIR}/jobs/${SLURM_JOB_NAME}.${SLURM_ARRAY_JOB_ID}.tsv"
echo "Finished task ${SLURM_ARRAY_TASK_ID} with exit code ${exitcode}"
exit ${exitcode}
