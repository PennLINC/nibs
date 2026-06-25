#!/bin/bash
#SBATCH --job-name=a2009s_acpc
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --array=1-26
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=8gb
#SBATCH --time=06:00:00
# Outputs ----------------------------------
#SBATCH --output=/cbica/projects/nibs/logs/a2009s_to_acpc_%A-%a.out
#SBATCH --error=/cbica/projects/nibs/logs/a2009s_to_acpc_%A-%a.err
# ------------------------------------------

pwd; hostname; date
set -euo pipefail

BIDS_DIR="/cbica/projects/nibs/dset"
DERIVS_DIR="/cbica/projects/nibs/derivatives"
SMRIPREP_DIR="${DERIVS_DIR}/smriprep"
QSIPREP_DIR="${DERIVS_DIR}/qsiprep"
XFM_DIR="${DERIVS_DIR}/t1w_registration"
LOGS_DIR="/cbica/projects/nibs/logs"

mkdir -p "${LOGS_DIR}/jobs"

# Parse participants.tsv line corresponding to this array index.
subject=$( sed -n -E "$((${SLURM_ARRAY_TASK_ID} + 1))s/sub-(\S*)\>.*/\1/gp" "${BIDS_DIR}/participants.tsv" )

if [[ -z "${subject}" ]]; then
  echo "No subject found for SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID}"
  exit 1
fi

FS_MGZ="${SMRIPREP_DIR}/sourcedata/freesurfer/sub-${subject}/mri/aparc.a2009s+aseg.mgz"
OUT_DIR="${XFM_DIR}/sub-${subject}/anat"
OUT_T1W_DSEG="${OUT_DIR}/sub-${subject}_space-T1w_desc-a2009s_dseg.nii.gz"
OUT_ACPC_DSEG="${OUT_DIR}/sub-${subject}_space-ACPC_desc-a2009s_dseg.nii.gz"
REF_ACPC="${QSIPREP_DIR}/sub-${subject}/anat/sub-${subject}_space-ACPC_desc-preproc_T1w.nii.gz"
T1W_TO_ACPC_XFM="${XFM_DIR}/sub-${subject}/anat/sub-${subject}_from-T1w_to-ACPC_mode-image_xfm.h5"

mkdir -p "${OUT_DIR}"

if [[ ! -f "${FS_MGZ}" ]]; then
  echo "Missing FreeSurfer segmentation: ${FS_MGZ}"
  exit 1
fi

if [[ ! -f "${REF_ACPC}" ]]; then
  echo "Missing ACPC reference image: ${REF_ACPC}"
  exit 1
fi

if [[ ! -f "${T1W_TO_ACPC_XFM}" ]]; then
  echo "Missing T1w->ACPC transform: ${T1W_TO_ACPC_XFM}"
  exit 1
fi

cmd_convert="mri_convert \"${FS_MGZ}\" \"${OUT_T1W_DSEG}\""
cmd_apply="antsApplyTransforms \
  --dimensionality 3 \
  --input \"${OUT_T1W_DSEG}\" \
  --reference-image \"${REF_ACPC}\" \
  --output \"${OUT_ACPC_DSEG}\" \
  --transform \"${T1W_TO_ACPC_XFM}\" \
  --interpolation NearestNeighbor"

echo "Running task ${SLURM_ARRAY_TASK_ID} for sub-${subject}"
echo "Convert command: ${cmd_convert}"
eval "${cmd_convert}"

echo "Transform command: ${cmd_apply}"
eval "${cmd_apply}"
exitcode=$?

echo "sub-${subject}   ${SLURM_ARRAY_TASK_ID}    ${exitcode}" \
  >> "${LOGS_DIR}/jobs/${SLURM_JOB_NAME}.${SLURM_ARRAY_JOB_ID}.tsv"
echo "Finished task ${SLURM_ARRAY_TASK_ID} with exit code ${exitcode}"
exit ${exitcode}
