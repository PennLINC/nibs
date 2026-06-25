#!/bin/bash
#SBATCH --job-name=warp_bundles
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --array=1-26
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=8gb
#SBATCH --time=12:00:00
# Outputs ----------------------------------
#SBATCH --output=/cbica/projects/nibs/logs/warp_bundles_%A-%a.out
#SBATCH --error=/cbica/projects/nibs/logs/warp_bundles_%A-%a.err
# ------------------------------------------

pwd; hostname; date
set -euo pipefail

BIDS_DIR="/cbica/projects/nibs/dset"
DERIVS_DIR="/cbica/projects/nibs/derivatives"
IN_ROOT="${DERIVS_DIR}/qsirecon/derivatives/qsirecon-DSIStudio"
OUT_ROOT="${DERIVS_DIR}/warped_bundles"
XFM_ROOT="${DERIVS_DIR}/t1w_registration"
LOGS_DIR="/cbica/projects/nibs/logs"

mkdir -p "${LOGS_DIR}/jobs"

# Parse participants.tsv line corresponding to this array index.
subject=$( sed -n -E "$((${SLURM_ARRAY_TASK_ID} + 1))s/sub-(\S*)\>.*/\1/gp" "${BIDS_DIR}/participants.tsv" )

if [[ -z "${subject}" ]]; then
  echo "No subject found for SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID}"
  exit 1
fi

SUBJ_IN_DIR="${IN_ROOT}/sub-${subject}"
if [[ ! -d "${SUBJ_IN_DIR}" ]]; then
  echo "Input subject directory not found: ${SUBJ_IN_DIR}"
  exit 1
fi

echo "Running task ${SLURM_ARRAY_TASK_ID} for sub-${subject}"
overall_exitcode=0

shopt -s nullglob
session_dirs=( "${SUBJ_IN_DIR}"/ses-* )
shopt -u nullglob

if [[ ${#session_dirs[@]} -eq 0 ]]; then
  echo "No session directories found under ${SUBJ_IN_DIR}"
  exit 1
fi

for session_dir in "${session_dirs[@]}"; do
  session=$(basename "${session_dir}")
  in_dwi="${session_dir}/dwi"
  out_dwi="${OUT_ROOT}/sub-${subject}/${session}/dwi"
  acpc_to_t1w_xfm="${XFM_ROOT}/sub-${subject}/anat/sub-${subject}_from-ACPC_to-T1w_mode-image_xfm.h5"
  mkdir -p "${out_dwi}"

  if [[ ! -d "${in_dwi}" ]]; then
    echo "Skipping ${session}: missing dwi directory ${in_dwi}"
    continue
  fi

  if [[ ! -f "${acpc_to_t1w_xfm}" ]]; then
    echo "Skipping ${session}: missing transform ${acpc_to_t1w_xfm}"
    continue
  fi

  ref_candidates=( "${in_dwi}"/*space-ACPC*param-ha*.nii.gz )
  if [[ ${#ref_candidates[@]} -eq 0 ]]; then
    echo "Skipping ${session}: no ACPC HA reference found in ${in_dwi}"
    continue
  fi
  ref="${ref_candidates[0]}"

  files=( "${in_dwi}"/sub-*_bundle-*_streamlines.tck.gz )
  if [[ ${#files[@]} -eq 0 ]]; then
    echo "Skipping ${session}: no bundle streamlines found in ${in_dwi}"
    continue
  fi

  args=()
  for f in "${files[@]}"; do
    base=$(basename "${f}")
    group=${base##*_bundle-}
    group=${group%_streamlines.tck.gz}
    args+=( --input-group-name "${group}" )
  done

  merged_file="${out_dwi}/sub-${subject}_${session}_space-ACPC_bundle-merged_streamlines.trx"
  warped_file="${out_dwi}/sub-${subject}_${session}_space-T1w_bundle-merged_streamlines.trx"

  echo "Session ${session}"
  echo "Reference: ${ref}"
  echo "Merged output: ${merged_file}"
  echo "Warped output: ${warped_file}"

  if [[ -f "${merged_file}" ]]; then
    echo "Merged tractogram already exists, skipping concatenation: ${merged_file}"
  else
    cmd_concat=( trx concatenate "${files[@]}" --reference "${ref}" "${args[@]}" --output "${merged_file}" )
    "${cmd_concat[@]}" || overall_exitcode=$?
  fi

  if [[ ${overall_exitcode} -ne 0 ]]; then
    continue
  fi

  if [[ -f "${warped_file}" ]]; then
    echo "Warped tractogram already exists, skipping warp: ${warped_file}"
  else
    # antsApplyTransformsToTRX uses point-transform direction conventions.
    # Use the inverse of the ACPC->T1w image transform for streamline vertices.
    cmd_warp=(
      antsApplyTransformsToTRX
      -i "${merged_file}"
      -o "${warped_file}"
      -t "[${acpc_to_t1w_xfm},1]"
    )
    "${cmd_warp[@]}" || overall_exitcode=$?
  fi

  if [[ ${overall_exitcode} -ne 0 ]]; then
    continue
  fi

  group_export_pattern="${out_dwi}/sub-${subject}_${session}_acq-HBCD75_run-01_space-T1w_model-gqi_bundle-{groupname}_streamlines.tck.gz"
  cmd_subset=(
    trx subset
    "${warped_file}"
    --group-export "${group_export_pattern}"
  )
  "${cmd_subset[@]}" || overall_exitcode=$?
done

echo "sub-${subject}   ${SLURM_ARRAY_TASK_ID}    ${overall_exitcode}" \
  >> "${LOGS_DIR}/jobs/${SLURM_JOB_NAME}.${SLURM_ARRAY_JOB_ID}.tsv"
echo "Finished task ${SLURM_ARRAY_TASK_ID} with exit code ${overall_exitcode}"
exit ${overall_exitcode}
