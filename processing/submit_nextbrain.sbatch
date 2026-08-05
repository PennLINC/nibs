#!/bin/bash
#SBATCH --job-name=nextbrain
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --array=1-26
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=8gb
#SBATCH --time=48:00:00

# Outputs ----------------------------------
#SBATCH --output=/cbica/projects/nibs/logs/nextbrain_%A-%a.out
#SBATCH --error=/cbica/projects/nibs/logs/nextbrain_%A-%a.err
# ------------------------------------------

set -euo pipefail

pwd
hostname
date

#============== Configuration ==============#

BIDS_DIR="/cbica/projects/nibs/dset"
SMRIPREP_DIR="/cbica/projects/nibs/derivatives/smriprep"
NEXTBRAIN_DIR="/cbica/projects/nibs/derivatives/nextbrain"
LOGS_DIR="/cbica/projects/nibs/logs"

FS_CONTAINER="${HOME}/apptainer/freesurfer-8.2.0.sif"
ATLAS_DIR="${HOME}/apptainer/atlas_simplified"
SUPERSYNTH_MODEL="${HOME}/apptainer/SuperSynth_August_2025.pth"
FS_LICENSE="/cbica/projects/nibs/tokens/freesurfer_license.txt"

# Location expected by NextBrain inside the FreeSurfer container.
ATLAS_CONTAINER_DIR="/usr/local/freesurfer/8.2.0-1/python/packages/ERC_bayesian_segmentation/atlas_simplified"
SUPERSYNTH_CONTAINER_MODEL="/usr/local/freesurfer/8.2.0-1/models/SuperSynth_August_2025.pth"

mkdir -p "${NEXTBRAIN_DIR}"
mkdir -p "${LOGS_DIR}"

#============== Helper functions ==============#

die() {
    echo "ERROR: $*" >&2
    exit 1
}

# Add all native-space preprocessed T1w images in a given anat directory
# to the array named by the first argument.
#
# Files containing a BIDS space entity are explicitly excluded, including:
#   *_space-MNI152NLin2009cAsym_desc-preproc_T1w.nii.gz
#
# Additional BIDS entities such as acq-, rec-, and run- are allowed.
find_native_t1w_in_anat_dir() {
    local array_name="$1"
    local anat_dir="$2"
    local candidate

    [[ -d "${anat_dir}" ]] || return 0

    while IFS= read -r candidate; do
        [[ -n "${candidate}" ]] || continue
        eval "${array_name}+=(\"\${candidate}\")"
    done < <(
        find "${anat_dir}" \
            -maxdepth 1 \
            -type f \
            -name "sub-${subject}_*desc-preproc_T1w.nii.gz" \
            ! -name "*_space-*" \
            -print 2>/dev/null |
            sort
    )
}

run_hemisphere() {
    local hemisphere="$1"

    echo
    echo "============================================================"
    echo "Running NextBrain hemisphere: ${hemisphere}"
    echo "Subject:                   sub-${subject}"
    echo "Input T1w:                 ${input_t1w}"
    echo "Output directory:          ${output_dir}"
    echo "============================================================"
    echo

    apptainer exec \
        --home "${HOME}" \
        --cleanenv \
        --containall \
        --writable-tmpfs \
        --env "FS_LICENSE=/license.txt" \
        --env "OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}" \
        --env "ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS=${SLURM_CPUS_PER_TASK}" \
        --env "MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}" \
        -B "${SMRIPREP_DIR}:/smriprep:ro" \
        -B "${NEXTBRAIN_DIR}:/nextbrain" \
        -B "${ATLAS_DIR}:${ATLAS_CONTAINER_DIR}:ro" \
        -B "${SUPERSYNTH_MODEL}:${SUPERSYNTH_CONTAINER_MODEL}:ro" \
        -B "${FS_LICENSE}:/license.txt:ro" \
        "${FS_CONTAINER}" \
        mri_histo_atlas_segment_fireants \
            --i "${container_input_t1w}" \
            --o "${container_output_dir}" \
            --device cpu \
            --threads "${SLURM_CPUS_PER_TASK}" \
            --side "${hemisphere}" \
            --mode invivo \
            --skip_bf
}

#============== Validate required files ==============#

[[ -f "${BIDS_DIR}/participants.tsv" ]] ||
    die "Missing participants file: ${BIDS_DIR}/participants.tsv"

[[ -f "${FS_CONTAINER}" ]] ||
    die "Missing FreeSurfer container: ${FS_CONTAINER}"

[[ -d "${ATLAS_DIR}" ]] ||
    die "Missing NextBrain atlas directory: ${ATLAS_DIR}"

[[ -f "${SUPERSYNTH_MODEL}" ]] ||
    die "Missing SuperSynth model: ${SUPERSYNTH_MODEL}"

[[ -f "${FS_LICENSE}" ]] ||
    die "Missing FreeSurfer license: ${FS_LICENSE}"

#============== Determine subject ==============#

# SLURM task 1 corresponds to the first participant after the TSV header.
participant_line=$((SLURM_ARRAY_TASK_ID + 1))

subject=$(
    awk -v line_number="${participant_line}" '
        NR == line_number {
            subject_id = $1
            sub(/^sub-/, "", subject_id)
            print subject_id
        }
    ' "${BIDS_DIR}/participants.tsv"
)

[[ -n "${subject}" ]] ||
    die "Could not determine a subject for SLURM array task ${SLURM_ARRAY_TASK_ID}"

echo "SLURM array task: ${SLURM_ARRAY_TASK_ID}"
echo "Selected subject: sub-${subject}"

#============== Locate the native-space T1w ==============#

subject_anat_dir="${SMRIPREP_DIR}/sub-${subject}/anat"
subject_t1ws=()

find_native_t1w_in_anat_dir subject_t1ws "${subject_anat_dir}"

if [[ ${#subject_t1ws[@]} -gt 1 ]]; then
    echo "Found multiple subject-level native-space preprocessed T1w images:" >&2
    printf '  %s\n' "${subject_t1ws[@]}" >&2
    die "Refusing to guess which subject-level T1w should be used."
fi

if [[ ${#subject_t1ws[@]} -eq 1 ]]; then
    # Subjects with multiple T1w inputs generally have a subject-level
    # sMRIPrep anatomical reference.
    input_t1w="${subject_t1ws[0]}"
    output_dir="${NEXTBRAIN_DIR}/sub-${subject}/anat"
else
    # Subjects with a single T1w may retain the session directory in their
    # sMRIPrep outputs.
    session_t1ws=()
    candidate=""

    while IFS= read -r candidate; do
        [[ -n "${candidate}" ]] || continue

        filename=$(basename "${candidate}")

        # Exclude normalized files carrying any BIDS space entity.
        if [[ "${filename}" == *"_space-"* ]]; then
            continue
        fi

        session_t1ws+=("${candidate}")
    done < <(
        find "${SMRIPREP_DIR}/sub-${subject}" \
            -type f \
            -path "*/ses-*/anat/*" \
            -name "sub-${subject}_*desc-preproc_T1w.nii.gz" \
            -print 2>/dev/null |
            sort
    )

    if [[ ${#session_t1ws[@]} -eq 0 ]]; then
        die "Could not find a native-space desc-preproc_T1w for sub-${subject}.

Checked subject-level directory:
  ${SMRIPREP_DIR}/sub-${subject}/anat/

Checked session-level directories:
  ${SMRIPREP_DIR}/sub-${subject}/ses-*/anat/

Files containing '_space-' were intentionally excluded."
    fi

    if [[ ${#session_t1ws[@]} -gt 1 ]]; then
        echo "Found multiple session-level native-space preprocessed T1w images:" >&2
        printf '  %s\n' "${session_t1ws[@]}" >&2
        die "Refusing to guess which session-level T1w should be used."
    fi

    input_t1w="${session_t1ws[0]}"

    session=$(
        printf '%s\n' "${input_t1w}" |
            sed -n 's#^.*/\(ses-[^/]*\)/anat/.*#\1#p'
    )

    [[ -n "${session}" ]] ||
        die "Could not extract the session identifier from: ${input_t1w}"

    output_dir="${NEXTBRAIN_DIR}/sub-${subject}/${session}/anat"
fi

mkdir -p "${output_dir}"

#============== Convert host paths to container paths ==============#

# The whole sMRIPrep derivative directory is bound to /smriprep.
container_input_t1w="/smriprep/${input_t1w#"${SMRIPREP_DIR}/"}"

# The whole NextBrain derivative directory is bound to /nextbrain.
container_output_dir="/nextbrain/${output_dir#"${NEXTBRAIN_DIR}/"}"

echo
echo "Selected native-space input:"
echo "  ${input_t1w}"
echo
echo "Container input:"
echo "  ${container_input_t1w}"
echo
echo "Output directory:"
echo "  ${output_dir}"
echo

# Record exactly which anatomical reference was used.
printf '%s\n' "${input_t1w}" > "${output_dir}/source_t1w.txt"

#============== Run NextBrain ==============#

# Run sequentially so the second hemisphere can reuse preprocessing products
# created during the first hemisphere.
run_hemisphere left
run_hemisphere right

exitcode=$?

#============== Record completion ==============#

echo -e "sub-${subject}\t${SLURM_ARRAY_TASK_ID}\t${exitcode}" \
    >> "${LOGS_DIR}/${SLURM_JOB_NAME}.${SLURM_ARRAY_JOB_ID}.tsv"

echo
echo "Finished sub-${subject} with exit code ${exitcode}"
date

exit "${exitcode}"