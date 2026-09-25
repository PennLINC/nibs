#!/usr/bin/env bash
#SBATCH --job-name=qsm
#SBATCH --array=1-26
#SBATCH --cpus-per-task=4
#SBATCH --mem=48gb
#SBATCH --time=48:00:00

set -euo pipefail
if command -v module >/dev/null 2>&1; then
  module load "${MATLAB_MODULE:-matlab/R2023A}"
fi
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec bash "${SCRIPT_DIR}/../../run_subject_job.sh" \
  "${SCRIPT_DIR}/process_qsm.py" \
  "${SCRIPT_DIR}/process_qsm_post.py"
