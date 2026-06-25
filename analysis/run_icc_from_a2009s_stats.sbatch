#!/bin/bash
#SBATCH --job-name=icc_a2009s
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8gb
#SBATCH --time=24:00:00
# Outputs ----------------------------------
#SBATCH --output=/cbica/projects/nibs/logs/icc_a2009s_%A.out
#SBATCH --error=/cbica/projects/nibs/logs/icc_a2009s_%A.err
# ------------------------------------------

pwd; hostname; date
set -euo pipefail

CODE_DIR="/cbica/projects/nibs/code/analysis"
LOGS_DIR="/cbica/projects/nibs/logs"

mkdir -p "${LOGS_DIR}/jobs"

cmd="~/.conda/envs/myelin_reliability/bin/python ${CODE_DIR}/compute_icc_from_a2009s_stats.py \
  --outdir /cbica/projects/nibs/derivatives/ICC"

echo "Running ICC job"
echo "Commandline: ${cmd}"
eval "${cmd}"
exitcode=$?

echo "all_subjects   ${SLURM_JOB_ID}    ${exitcode}" \
  >> "${LOGS_DIR}/jobs/${SLURM_JOB_NAME}.${SLURM_JOB_ID}.tsv"
echo "Finished ICC job with exit code ${exitcode}"
exit ${exitcode}
