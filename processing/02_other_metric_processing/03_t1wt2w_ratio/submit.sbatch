#!/usr/bin/env bash
#SBATCH --job-name=t1wt2w_ratio
#SBATCH --array=1-26
#SBATCH --cpus-per-task=4
#SBATCH --mem=32gb
#SBATCH --time=24:00:00

set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec bash "${SCRIPT_DIR}/../../run_subject_job.sh" "${SCRIPT_DIR}/process_t1wt2w_ratio.py"
