#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=48:00:00

/cbica/projects/nibs/.bashrc

conda activate curation

# Run heudiconv on the first session
sessions="01 02"

# Loop over sessions
for session in $sessions; do
    # Find subjects with the requested session
    subjects=($(ls -d /cbica/projects/nibs/sourcedata/scitran/bbl/NIBS_857664/*_"$session" 2>/dev/null | sed 's|.*/\([0-9a-zA-Z]*\)_.*|\1|' | sort -u))

    if [ ${#subjects[@]} -eq 0 ]; then
        echo "No subjects found for session $session"
        continue
    fi

    echo "Found ${#subjects[@]} subjects for session $session: ${subjects[*]}"

    # Filter out already-converted subjects
    subjects=($(for s in "${subjects[@]}"; do [ ! -d "/cbica/projects/nibs/dset/sub-$s/ses-$session" ] && echo "$s"; done))

    for sub in "${subjects[@]}"
    do
        echo "$sub"
        heudiconv \
            -f /cbica/projects/nibs/code/curation/heuristic.py \
            -o /cbica/projects/nibs/dset \
            -d "/cbica/projects/nibs/sourcedata/scitran/bbl/NIBS_857664/{subject}_{session}/*/*/*/*.dcm" \
            --subjects "$sub" \
            --ses "$session" \
            -g all \
            --bids \
            --queue SLURM
    done
done
