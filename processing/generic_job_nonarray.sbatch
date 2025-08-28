#!/bin/bash
#SBATCH --job-name=nibs
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=20gb
#SBATCH --time=24:00:00
# Outputs ----------------------------------
#SBATCH --output=/cbica/projects/nibs/code/processing/jobs/job_%A_nonarray.out
#SBATCH --error=/cbica/projects/nibs/code/processing/jobs/job_%A_nonarray.err
# ------------------------------------------

pwd; hostname; date
set -e

python process_g_ratio_scaling_factors.py
