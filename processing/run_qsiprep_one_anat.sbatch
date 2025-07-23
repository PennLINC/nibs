#!/bin/bash
#SBATCH --job-name=qsiprep
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --array=5
#SBATCH --cpus-per-task=20
#SBATCH --mem-per-cpu=4gb
#SBATCH --time=36:00:00
# Outputs ----------------------------------
#SBATCH --output=/cbica/projects/nibs/code/processing/jobs/qsiprep_%A-%a.out
#SBATCH --error=/cbica/projects/nibs/code/processing/jobs/qsiprep_%A-%a.err
# ------------------------------------------

pwd; hostname; date
set -e

#==============Shell script==============#
# Location of the *reduced* BIDS dataset
CODE_DIR="/cbica/projects/nibs/code/processing"
BIDS_DIR="/cbica/projects/nibs/dset"
SCRATCH_DIR="/cbica/comp_space/nibs/nibs-qsiprep-one-anat"
DERIVS_DIR="/cbica/projects/nibs/derivatives"
mkdir -p ${SCRATCH_DIR}
mkdir -p ${DERIVS_DIR}

# Make sure FS_LICENSE is defined in the container.
FS_LICENSE="/cbica/projects/nibs/tokens/freesurfer_license.txt"

# Parse the participants.tsv file and extract one subject ID from the line corresponding to this SLURM task.
subject=$( sed -n -E "$((${SLURM_ARRAY_TASK_ID} + 1))s/sub-(\S*)\>.*/\1/gp" ${BIDS_DIR}/participants.tsv )

# Compose the command line
cmd="apptainer run --cleanenv \
      -B ${BIDS_DIR}:/data \
      -B ${CODE_DIR}:/code \
      -B ${DERIVS_DIR}:/deriv \
      -B ${SCRATCH_DIR}:/work \
      -B ${FS_LICENSE}:/license.txt \
      /cbica/projects/nibs/apptainer/qsiprep-1.0.1.sif \
      /data \
      /deriv/qsiprep \
      participant \
      --participant-label ${subject} \
      -w /work/ \
      --omp-nthreads ${SLURM_CPUS_PER_TASK} \
      --nprocs ${SLURM_CPUS_PER_TASK} \
      --mem 39000 \
      --fs-license-file /license.txt \
      --skip-bids-validation \
      --denoise-method dwidenoise \
      --output-resolution 1.7 \
      --subject-anatomical-reference first-alphabetically \
      --hmc-model 3dSHORE \
      --pepolar-method DRBUDDI \
      --b1-biascorrect-stage none \
      --bids-filter-file /code/fmriprep_filter.json \
      --ignore t2w"

# Setup done, run the command
echo Running task "${SLURM_ARRAY_TASK_ID}"
echo Commandline: "$cmd"
eval "$cmd"
exitcode=$?

# Output results to a table
echo "sub-$subject   ${SLURM_ARRAY_TASK_ID}    $exitcode" \
      >> ${CODE_DIR}/jobs/"${SLURM_JOB_NAME}"."${SLURM_ARRAY_JOB_ID}".tsv
echo Finished tasks "${SLURM_ARRAY_TASK_ID}" with exit code $exitcode
exit $exitcode
