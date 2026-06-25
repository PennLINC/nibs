#!/bin/bash
#SBATCH --job-name=t1w_reg
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --array=1-26
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8gb
#SBATCH --time=12:00:00
# Outputs ----------------------------------
#SBATCH --output=/cbica/projects/nibs/logs/t1w_registration_%A-%a.out
#SBATCH --error=/cbica/projects/nibs/logs/t1w_registration_%A-%a.err
# ------------------------------------------

pwd; hostname; date
set -euo pipefail

BIDS_DIR="/cbica/projects/nibs/dset"
DERIVS_DIR="/cbica/projects/nibs/derivatives"
QSIPREP_DIR="${DERIVS_DIR}/qsiprep"
SMRIPREP_DIR="${DERIVS_DIR}/smriprep"
OUT_ROOT="${DERIVS_DIR}/t1w_registration"
TMP_ROOT="/cbica/comp_space/nibs/t1w_registration"
LOGS_DIR="/cbica/projects/nibs/logs"

mkdir -p "${OUT_ROOT}"
mkdir -p "${TMP_ROOT}"
mkdir -p "${LOGS_DIR}/jobs"

# Parse participants.tsv line corresponding to this array index.
subject=$( sed -n -E "$((${SLURM_ARRAY_TASK_ID} + 1))s/sub-(\S*)\>.*/\1/gp" "${BIDS_DIR}/participants.tsv" )

if [[ -z "${subject}" ]]; then
  echo "No subject found for SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID}"
  exit 1
fi

FIXED="${QSIPREP_DIR}/sub-${subject}/anat/sub-${subject}_space-ACPC_desc-preproc_T1w.nii.gz"
MOVING="${SMRIPREP_DIR}/sub-${subject}/anat/sub-${subject}_acq-MPRAGE_rec-refaced_run-01_desc-preproc_T1w.nii.gz"
FIXED_MASK="${QSIPREP_DIR}/sub-${subject}/anat/sub-${subject}_space-ACPC_desc-brain_mask.nii.gz"
MOVING_MASK="${SMRIPREP_DIR}/sub-${subject}/anat/sub-${subject}_acq-MPRAGE_rec-refaced_run-01_desc-brain_mask.nii.gz"

OUT_DIR="${OUT_ROOT}/sub-${subject}/anat"
mkdir -p "${OUT_DIR}"

TMP_SUBJ_DIR="${TMP_ROOT}/sub-${subject}"
mkdir -p "${TMP_SUBJ_DIR}"

WORK_PREFIX="${TMP_SUBJ_DIR}/sub-${subject}_acpc_to_t1w_"
WARPED_IMAGE="${TMP_SUBJ_DIR}/sub-${subject}_space-ACPC_desc-preproc_from-T1w_warped.nii.gz"

if [[ ! -f "${FIXED}" ]]; then
  echo "Missing fixed image: ${FIXED}"
  exit 1
fi

if [[ ! -f "${MOVING}" ]]; then
  echo "Missing moving image: ${MOVING}"
  exit 1
fi

mask_arg="--masks [ NULL, NULL ]"
if [[ -f "${FIXED_MASK}" && -f "${MOVING_MASK}" ]]; then
  mask_arg="--masks [ ${FIXED_MASK}, ${MOVING_MASK} ]"
elif [[ -f "${FIXED_MASK}" ]]; then
  echo "Moving mask missing, using fixed mask only: ${MOVING_MASK}"
  mask_arg="--masks [ ${FIXED_MASK}, NULL ]"
elif [[ -f "${MOVING_MASK}" ]]; then
  echo "Fixed mask missing, using moving mask only: ${FIXED_MASK}"
  mask_arg="--masks [ NULL, ${MOVING_MASK} ]"
fi

cmd="antsRegistration \
  --collapse-output-transforms 1 \
  --dimensionality 3 \
  --float 0 \
  --initial-moving-transform [ ${FIXED}, ${MOVING}, 1 ] \
  --initialize-transforms-per-stage 0 \
  --interpolation BSpline \
  --output [ ${WORK_PREFIX}, ${WARPED_IMAGE} ] \
  --transform Rigid[ 0.1 ] \
  --metric Mattes[ ${FIXED}, ${MOVING}, 1, 32, Random, 0.25 ] \
  --convergence [ 1000x500x250x100, 1e-06, 10 ] \
  --smoothing-sigmas 3.0x2.0x1.0x0.0mm \
  --shrink-factors 8x4x2x1 \
  --use-histogram-matching 0 \
  ${mask_arg} \
  --winsorize-image-intensities [ 0.002, 0.998 ] \
  --write-composite-transform 1"

echo "Running task ${SLURM_ARRAY_TASK_ID} for sub-${subject}"
echo "Commandline: ${cmd}"
eval "${cmd}"
exitcode=$?

if [[ ! -f "${WORK_PREFIX}Composite.h5" ]]; then
  echo "Expected output missing: ${WORK_PREFIX}Composite.h5"
  exit 1
fi

if [[ ! -f "${WORK_PREFIX}InverseComposite.h5" ]]; then
  echo "Expected output missing: ${WORK_PREFIX}InverseComposite.h5"
  exit 1
fi

# Fixed is ACPC and moving is native T1w.
# Composite maps T1w -> ACPC, InverseComposite maps ACPC -> T1w.
mv "${WORK_PREFIX}Composite.h5" \
  "${OUT_DIR}/sub-${subject}_from-T1w_to-ACPC_mode-image_xfm.h5"
mv "${WORK_PREFIX}InverseComposite.h5" \
  "${OUT_DIR}/sub-${subject}_from-ACPC_to-T1w_mode-image_xfm.h5"

echo "sub-${subject}   ${SLURM_ARRAY_TASK_ID}    ${exitcode}" \
  >> "${LOGS_DIR}/jobs/${SLURM_JOB_NAME}.${SLURM_ARRAY_JOB_ID}.tsv"
echo "Finished task ${SLURM_ARRAY_TASK_ID} with exit code ${exitcode}"
exit ${exitcode}
