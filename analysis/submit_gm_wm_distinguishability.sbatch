#!/bin/bash
#SBATCH --job-name=gm_wm_distinct
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8gb
#SBATCH --time=24:00:00
#SBATCH --output=/cbica/projects/nibs/logs/gm_wm_distinct_%A.out
#SBATCH --error=/cbica/projects/nibs/logs/gm_wm_distinct_%A.err

pwd; hostname; date
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/cbica/projects/nibs}"
CODE_DIR="${CODE_DIR:-${PROJECT_ROOT}/code/analysis}"
LOGS_DIR="${LOGS_DIR:-${PROJECT_ROOT}/logs}"
PYTHON_BIN="${PYTHON_BIN:-${HOME}/.conda/envs/myelin_reliability/bin/python}"
OUTDIR="${OUTDIR:-${PROJECT_ROOT}/derivatives/ICC}"
QC_FILE="${QC_FILE:-${PROJECT_ROOT}/code/data/manual_qc_modality.tsv}"
QC_MODES="${QC_MODES:-metricqc completeqc}"
N_BOOTSTRAP="${N_BOOTSTRAP:-10000}"

mkdir -p "${LOGS_DIR}/jobs"

cmd="${PYTHON_BIN} ${CODE_DIR}/compute_gm_wm_distinguishability.py \
  --qc-file ${QC_FILE} \
  --qc-mode ${QC_MODES} \
  --outdir ${OUTDIR} \
  --n-bootstrap ${N_BOOTSTRAP}"

echo "Running GM/WM distinguishability job"
echo "Commandline: ${cmd}"
eval "${cmd}"
exitcode=$?

echo "all_subjects   ${SLURM_JOB_ID}    ${exitcode}" \
  >> "${LOGS_DIR}/jobs/${SLURM_JOB_NAME}.${SLURM_JOB_ID}.tsv"

echo "Finished GM/WM distinguishability job with exit code ${exitcode}"
exit "${exitcode}"
