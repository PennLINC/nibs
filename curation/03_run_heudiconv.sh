#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=48:00:00

/cbica/projects/nibs/.bashrc

mamba activate curation

# Run heudiconv on the first session
subjects=($(ls -d /cbica/projects/nibs/sourcedata/imaging/scitran/bbl/NIBS_857664/*_* | sed 's|.*/\([0-9a-zA-Z]*\)_.*|\1|' | sort -u))

for sub in "${subjects[@]}"
do
    echo "$sub"
    heudiconv \
        -f /cbica/projects/nibs/code/curation/heuristic.py \
        -o /cbica/projects/nibs/dset \
        -d "/cbica/projects/nibs/sourcedata/imaging/scitran/bbl/NIBS_857664/{subject}_{session}/*/*/*/*.dcm" \
        --subjects "$sub" \
        --ses 1 \
        -g all \
        --bids \
        --queue SLURM
done
