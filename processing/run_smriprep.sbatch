#!/bin/bash
#SBATCH --job-name=smriprep
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --array=1-9
#SBATCH --cpus-per-task=20
#SBATCH --mem-per-cpu=4gb
#SBATCH --time=48:00:00
# Outputs ----------------------------------
#SBATCH --output=/cbica/projects/nibs/code/processing/jobs/smriprep_%A-%a.out
#SBATCH --error=/cbica/projects/nibs/code/processing/jobs/smriprep_%A-%a.err
# ------------------------------------------

pwd; hostname; date
set -e

#==============Shell script==============#
BIDS_DIR="/cbica/projects/nibs/dset"
CODE_DIR="/cbica/projects/nibs/code/processing"
JOBS_DIR="/cbica/projects/nibs/code/processing/jobs"
SCRATCH_DIR="/cbica/comp_space/nibs/smriprep"
DERIVS_DIR="/cbica/projects/nibs/derivatives/smriprep"
mkdir -p ${SCRATCH_DIR}
mkdir -p ${DERIVS_DIR}

# Make sure FS_LICENSE is defined in the container.
FS_LICENSE="/cbica/projects/nibs/tokens/freesurfer_license.txt"

# Parse the participants.tsv file and extract one subject ID from the line corresponding to this SLURM task.
subject=$( sed -n -E "$((${SLURM_ARRAY_TASK_ID} + 1))s/sub-(\S*)\>.*/\1/gp" ${BIDS_DIR}/participants.tsv )

# We'll run fMRIPrep in anat-only mode instead of sMRIPrep because we're more familiar with fMRIPrep.
# Only use the MPRAGE scan. Ignore the SPACE T1w and T2w.
cmd="apptainer run --home $HOME --cleanenv --containall --writable-tmpfs \
      -B ${BIDS_DIR}:/data \
      -B ${CODE_DIR}:/code \
      -B ${DERIVS_DIR}:/out \
      -B ${SCRATCH_DIR}:/work \
      -B ${FS_LICENSE}:/license.txt \
      /cbica/projects/nibs/apptainer/fmriprep-25.0.0.sif \
      /data \
      /out \
      participant \
      --participant-label ${subject} \
      -w /work \
      --omp-nthreads ${SLURM_CPUS_PER_TASK} \
      --nprocs ${SLURM_CPUS_PER_TASK} \
      --mem_mb 39000 \
      --output-spaces T1w MNI152NLin2009cAsym \
      --output-layout bids \
      --anat-only \
      --project-goodvoxels \
      --cifti-output 170k \
      --fs-license-file /license.txt \
      --skip_bids_validation \
      --level full \
      --longitudinal \
      --bids-filter-file /code/fmriprep_filter.json \
      --ignore t2w"

# Setup done, run the command
echo Running task "${SLURM_ARRAY_TASK_ID}"
echo Commandline: "$cmd"
eval "$cmd"
exitcode=$?

# Output results to a table
echo "sub-$subject   ${SLURM_ARRAY_TASK_ID}    $exitcode" \
      >> ${JOBS_DIR}/"${SLURM_JOB_NAME}"."${SLURM_ARRAY_JOB_ID}".tsv
echo Finished tasks "${SLURM_ARRAY_TASK_ID}" with exit code $exitcode
exit $exitcode
