#!/usr/bin/env bash
#SBATCH --job-name=coreg_qc
#SBATCH --array=1-26
#SBATCH --cpus-per-task=2
#SBATCH --mem=16gb
#SBATCH --time=12:00:00

set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec bash "${SCRIPT_DIR}/../../run_subject_job.sh" \
  "${SCRIPT_DIR}/generate_coregistration_reports.py"
