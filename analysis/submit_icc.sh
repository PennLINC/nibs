#!/bin/bash
#SBATCH --job-name=myelin_icc
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G

# prevent thread oversubscription (very important)
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS=1

# Run (n-jobs defaults to $SLURM_CPUS_ON_NODE, i.e., 32 here)
~/.conda/envs/myelin_reliability/bin/python -u /cbica/projects/nibs/projects/myelin_reliability/code/myelin_reliability/compute_icc.py \
  --outdir /cbica/projects/nibs/derivatives/qa_icc_03152026 \
  --n-jobs $SLURM_CPUS_PER_TASK
