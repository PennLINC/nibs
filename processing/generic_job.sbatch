#!/bin/bash
#SBATCH --job-name=nibs
#SBATCH --nodes=1
#SBATCH --cpus-per-task=20
#SBATCH --mem-per-cpu=4gb
#SBATCH --time=100:00:00
#SBATCH --partition=long
# Outputs ----------------------------------
#SBATCH --output=/cbica/projects/nibs/code/processing/jobs/job.out
#SBATCH --error=/cbica/projects/nibs/code/processing/jobs/job.err
# ------------------------------------------

pwd; hostname; date
set -e

# conda activate processing

# python process_mp2rage.py
python process_ihmt.py
