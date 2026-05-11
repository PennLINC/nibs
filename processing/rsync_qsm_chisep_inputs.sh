#!/usr/bin/env bash
set -euo pipefail

REMOTE="${REMOTE:-tsalo@bblsub2:/home/tsalo/nibs/derivatives}"
PROJECT_ROOT="${PROJECT_ROOT:-/cbica/projects/nibs}"
INCLUDE_SOFTWARE=0
SUBJECTS=()

usage() {
    cat <<'USAGE'
Usage:
  rsync_qsm_chisep_inputs.sh [options] [<subject> ...]

Copy the files needed to run processing/process_qsm_chisep.py and
processing/process_qsm_chisep.m for one or more subjects while preserving
paths relative to PROJECT_ROOT.

Arguments:
  subject                 BIDS subject label, with or without "sub-"
                          default: all subjects under PROJECT_ROOT/dset

Options:
  -r, --remote REMOTE     rsync destination
                          default: tsalo@bblsub2:/home/tsalo/nibs/derivatives
  -p, --project-root DIR  source project root
                          default: /cbica/projects/nibs
  --include-software      also copy MATLAB toolbox folders referenced by
                          process_qsm_chisep.m under software/
  -n, --dry-run           show what would be copied
  -h, --help              show this help

Environment:
  REMOTE                  override the default remote destination
  PROJECT_ROOT            override the default source project root

Example:
  ./processing/rsync_qsm_chisep_inputs.sh
  ./processing/rsync_qsm_chisep_inputs.sh sub-60526
  ./processing/rsync_qsm_chisep_inputs.sh --dry-run 60526 60522
USAGE
}

RSYNC_OPTS=(-avh --relative --prune-empty-dirs)

while [[ $# -gt 0 ]]; do
    case "$1" in
        -r|--remote)
            REMOTE="$2"
            shift 2
            ;;
        -p|--project-root)
            PROJECT_ROOT="$2"
            shift 2
            ;;
        --include-software)
            INCLUDE_SOFTWARE=1
            shift
            ;;
        -n|--dry-run)
            RSYNC_OPTS+=(--dry-run)
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        -*)
            echo "Unknown option: $1" >&2
            usage >&2
            exit 2
            ;;
        *)
            SUBJECTS+=("${1#sub-}")
            shift
            ;;
    esac
done

if [[ ! -d "$PROJECT_ROOT" ]]; then
    echo "PROJECT_ROOT does not exist: $PROJECT_ROOT" >&2
    exit 1
fi

if [[ ${#SUBJECTS[@]} -eq 0 ]]; then
    while IFS= read -r -d '' subject_dir; do
        SUBJECTS+=("$(basename "$subject_dir" | sed 's/^sub-//')")
    done < <(compgen -G "$PROJECT_ROOT/dset/sub-*" | while IFS= read -r path; do [[ -d "$path" ]] && printf '%s\0' "$path"; done)

    if [[ ${#SUBJECTS[@]} -eq 0 ]]; then
        echo "No subjects found under $PROJECT_ROOT/dset." >&2
        exit 1
    fi
fi

tmp_filelist="$(mktemp)"
cleanup() {
    rm -f "$tmp_filelist"
}
trap cleanup EXIT

add_if_exists() {
    local rel_path="$1"
    if [[ -e "$PROJECT_ROOT/$rel_path" ]]; then
        printf '%s\0' "$rel_path" >> "$tmp_filelist"
    else
        echo "Warning: missing $rel_path" >&2
    fi
}

add_matches() {
    local pattern="$1"
    local matched=0
    while IFS= read -r -d '' path; do
        printf '%s\0' "${path#"$PROJECT_ROOT/"}" >> "$tmp_filelist"
        matched=1
    done < <(compgen -G "$PROJECT_ROOT/$pattern" | while IFS= read -r path; do printf '%s\0' "$path"; done)
    if [[ "$matched" -eq 0 ]]; then
        echo "Warning: no matches for $pattern" >&2
    fi
}

# Code and config required by process_qsm_chisep.py.
add_if_exists "processing/process_qsm_chisep.py"
add_if_exists "processing/process_qsm_chisep.m"
add_if_exists "processing/utils.py"
add_if_exists "configuration/config.py"
add_if_exists "configuration/paths.yml"
add_if_exists "configuration/nibs_bids_config.json"

# MATLAB toolbox folders hard-coded in process_qsm_chisep.m. These can be
# large, so copy them only when the destination machine does not already have
# compatible installations.
if [[ "$INCLUDE_SOFTWARE" -eq 1 ]]; then
    add_if_exists "software/Chisep_Toolbox_v1.2"
    add_if_exists "software/NIfTI_20140122"
    add_if_exists "software/STISuite_V3.0"
    add_if_exists "software/MEDI"
    add_if_exists "software/SEGUE_28012021"
    add_if_exists "software/mritools"
fi

for subject in "${SUBJECTS[@]}"; do
    # Raw BIDS file used for sessions, input metadata, and output NIfTI template.
    add_matches "dset/sub-${subject}/ses-*/anat/sub-${subject}_ses-*_acq-QSM_run-*_echo-1_part-mag_MEGRE.nii.gz"

    # R2' and R2* maps collected from derivatives/qsm.
    add_matches "derivatives/qsm/sub-${subject}/ses-*/anat/sub-${subject}_ses-*_run-*_space-MEGRE_desc-MEGRE+E12345_R2primemap.nii.gz"
    add_matches "derivatives/qsm/sub-${subject}/ses-*/anat/sub-${subject}_ses-*_run-*_space-MEGRE_desc-MEGRE+E2345_R2primemap.nii.gz"
    add_matches "derivatives/qsm/sub-${subject}/ses-*/anat/sub-${subject}_ses-*_run-*_space-MEGRE_desc-MEGRE+E12345_R2starmap.nii.gz"
    add_matches "derivatives/qsm/sub-${subject}/ses-*/anat/sub-${subject}_ses-*_run-*_space-MEGRE_desc-MEGRE+E2345_R2starmap.nii.gz"

    # SEPIA work inputs consumed directly by process_qsm_chisep.m.
    add_matches "work/qsm-E12345+sepia/sub-${subject}/ses-*/anat/sub-${subject}_ses-*_part-mag_desc-concat_MEGRE.nii.gz"
    add_matches "work/qsm-E12345+sepia/sub-${subject}/ses-*/anat/sub-${subject}_ses-*_part-phase_desc-concat_MEGRE.nii.gz"
    add_matches "work/qsm-E12345+sepia/sub-${subject}/ses-*/anat/sub-${subject}_ses-*_header.mat"
    add_matches "work/qsm-E2345+sepia/sub-${subject}/ses-*/anat/sub-${subject}_ses-*_part-mag_desc-concat_MEGRE.nii.gz"
    add_matches "work/qsm-E2345+sepia/sub-${subject}/ses-*/anat/sub-${subject}_ses-*_part-phase_desc-concat_MEGRE.nii.gz"
    add_matches "work/qsm-E2345+sepia/sub-${subject}/ses-*/anat/sub-${subject}_ses-*_header.mat"
done

if [[ ! -s "$tmp_filelist" ]]; then
    echo "No files were found to rsync." >&2
    exit 1
fi

echo "Copying chi-sep inputs from $PROJECT_ROOT to $REMOTE"
rsync "${RSYNC_OPTS[@]}" --files-from="$tmp_filelist" --from0 "$PROJECT_ROOT/" "$REMOTE/"
