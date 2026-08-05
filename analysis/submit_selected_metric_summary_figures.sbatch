#!/bin/bash
#SBATCH --job-name=metric_summary_figs
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8gb
#SBATCH --time=24:00:00
#SBATCH --output=/cbica/projects/nibs/logs/metric_summary_figs_%A.out
#SBATCH --error=/cbica/projects/nibs/logs/metric_summary_figs_%A.err

pwd; hostname; date
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/cbica/projects/nibs}"
CODE_DIR="${CODE_DIR:-${PROJECT_ROOT}/code/analysis}"
LOGS_DIR="${LOGS_DIR:-${PROJECT_ROOT}/logs}"
PYTHON_BIN="${PYTHON_BIN:-${HOME}/.conda/envs/myelin_reliability/bin/python}"
OUT_ROOT="${OUT_ROOT:-${PROJECT_ROOT}/derivatives/ICC/selected_metric_summary_figures}"
QC_FILE="${QC_FILE:-${PROJECT_ROOT}/code/data/manual_qc_modality.tsv}"
QC_MODES="${QC_MODES:-metricqc completeqc}"
MIN_COMMON_REGIONS="${MIN_COMMON_REGIONS:-10}"

mkdir -p "${LOGS_DIR}/jobs"

exitcode=0
for qc_mode in ${QC_MODES}; do
  out_dir="${OUT_ROOT}/${qc_mode}"
  cache_dir="${out_dir}/cache"
  cmd="${PYTHON_BIN} ${CODE_DIR}/selected_metric_summary_figures.py \
    --project-root ${PROJECT_ROOT} \
    --qc-mode ${qc_mode} \
    --qc-file ${QC_FILE} \
    --output-dir ${out_dir} \
    --cache-dir ${cache_dir} \
    --min-common-regions ${MIN_COMMON_REGIONS}"

  echo "Running selected metric summary figures for qc_mode=${qc_mode}"
  echo "Commandline: ${cmd}"
  eval "${cmd}" || exitcode=$?
  if [[ "${exitcode}" -ne 0 ]]; then
    break
  fi
done

echo "all_subjects   ${SLURM_JOB_ID}    ${exitcode}" \
  >> "${LOGS_DIR}/jobs/${SLURM_JOB_NAME}.${SLURM_JOB_ID}.tsv"

echo "Finished selected metric summary figures with exit code ${exitcode}"
exit "${exitcode}"
