#!/usr/bin/env bash
#SBATCH --job-name=fs_vogt_parc
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --output=/cbica/projects/nibs/logs/myelin_atlas/%x_%A_%a.out
#SBATCH --error=/cbica/projects/nibs/logs/myelin_atlas/%x_%A_%a.err

set -euo pipefail
set +x

# -----------------------------
# User configuration
# -----------------------------
export SUBJECTS_DIR="/cbica/projects/nibs/derivatives/smriprep/sourcedata/freesurfer"
ATLAS_SRCDIR="/cbica/projects/nibs/code/processing/myelin_atlas/MYATLAS_package_new"
OUTDIR="/cbica/projects/nibs/derivatives/myelin_atlas"
PARCEL="vogt_vogt"             # annot basename (produces lh.<SUBID>.<PARCEL>.annot etc.)
CTAB="${ATLAS_SRCDIR}/${PARCEL}_new.ctab"

# Singularity/Apptainer settings (FreeSurfer inside fMRIPrep image)
SIF="${HOME}/apptainer/fmriprep-25.0.0.sif"
FS_LICENSE_PATH="/cbica/projects/nibs/tokens/freesurfer_license.txt"

# label names to transfer from COLIN27_FS -> subject (as you had them)
label_name=( 1 2 3 4 5 6 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50 51 52 53 54 55 56 57 58 59 60 61 62 63 64 65 66 67 67III 67IV 68I 68II 68III 69 70med 70I 70II 71m 71I 71II 72 73I 73II 73III 74I 74II 75med 75sup 75if 76s 76i 77 78 79 80 81 82 83I 83II 83III 83IV 84 85I 85II 85III 85IV 86 87 88a 88p 89a 89m 89p 89ip 89t 90a 90m 90p 90ip 90t 90o 91 92 93 94 95 96 97 98 99 100 101 102 103 104 105 106 107 108 109 110 111l 111m 112l 112m 113 114 115 116 117 118 119 BA18 BA17 120 121 122 123 124 125 126 127 128 129 130 131 132 133 134 135 136 137 138 139 140 141 142 143 144 145 146 147 148 149 150 151 152 153 154 155 156 157 158 159 160 161 162 163 164 165 166 167 168 169 170 171 172 173 174 175 176 177 178 179 180 181 182 )
HEMIS=(lh rh)

# -----------------------------
# Helpers
# -----------------------------

# Run FreeSurfer command inside Apptainer with license+binds
fs() {
  singularity exec -e \
    -B "${SUBJECTS_DIR}" \
    -B "${OUTDIR}" \
    -B "${ATLAS_SRCDIR}" \
    -B "${FS_LICENSE_PATH}:/LICENSE" \
    --env FS_LICENSE=/LICENSE \
    --env SUBJECTS_DIR="${SUBJECTS_DIR}" \
    "${SIF}" "$@"
}

# Build subject list from SUBJECTS_DIR/sub-*
build_subject_list() {
  mapfile -t SUBS < <(find "${SUBJECTS_DIR}" -maxdepth 1 -mindepth 1 -type d -name 'sub-*' -printf '%f\n' | sort)
  if (( ${#SUBS[@]} == 0 )); then
    echo "No subjects found in ${SUBJECTS_DIR}/sub-*"; exit 1
  fi
}

# Ensure output dirs
init_dirs() {
  mkdir -p "${OUTDIR}/logs"
}

# One subject end-to-end
process_subject() {
  local SUBID="$1"
  echo "=== Processing ${SUBID} ==="

  # BIDS-safe subject label (exactly one 'sub-')
  local BIDS_SUB="sub-${SUBID#sub-}"

  # 1) If annot doesn’t exist, transfer labels from COLIN27_FS and build annot per hemi
  for HEMI in "${HEMIS[@]}"; do
    local ANNOT_PATH="${SUBJECTS_DIR}/${SUBID}/label/${HEMI}.${BIDS_SUB}.${PARCEL}.annot"
    if [[ ! -e "${ANNOT_PATH}" ]]; then
      echo "[${SUBID}/${HEMI}] No annot found — creating from labels..."
      local LABEL_DIR="${SUBJECTS_DIR}/${SUBID}/label/${PARCEL}"
      mkdir -p "${LABEL_DIR}"

      # Transfer each label from the atlas (COLIN27_FS) to this subject
      for label in "${label_name[@]}"; do
        local SRC_LABEL="${ATLAS_SRCDIR}/label/${HEMI}.colin27.${PARCEL}.label/${HEMI}.${label}.label"
        local TRG_LABEL="${LABEL_DIR}/${HEMI}.${label}.label"
        if [[ ! -e "${SRC_LABEL}" ]]; then
          echo "WARNING: missing source label ${SRC_LABEL} — skipping"
          continue
        fi
        mri_label2label \
          --srclabel "${SRC_LABEL}" \
          --srcsubject COLIN27_FS \
          --trglabel "${TRG_LABEL}" \
          --trgsubject "${SUBID}" \
          --regmethod surface \
          --hemi "${HEMI}"
      done

      # Build .annot
      fs mris_label2annot \
        --s "${SUBID}" \
        --h "${HEMI}" \
        --ctab "${CTAB}" \
        --a "${BIDS_SUB}.${PARCEL}" \
        --ldir "${LABEL_DIR}" \
        --nhits ${SUBJECTS_DIR}/${SUBID}/mri/myelin_atlas_overlapped_vertex.mgh \
        --no-unknown
    else
      echo "[${SUBID}/${HEMI}] Annot exists — skipping build: ${ANNOT_PATH}"
    fi
  done

  # 2) Create a voxel parcellation from the surface annots (both hemispheres)
  local DSEG_DIR="${OUTDIR}/${BIDS_SUB}/anat"
  mkdir -p "${DSEG_DIR}"
  local DSEG_PATH="${DSEG_DIR}/${BIDS_SUB}_space-T1w_label-vogt_dseg.nii.gz"

  if [[ -e "${DSEG_PATH}" ]]; then
    echo "[${SUBID}] Voxel parcellation exists — ${DSEG_PATH}"
  else
    echo "[${SUBID}] Creating voxel parcellation with mri_aparc2aseg..."
    mri_aparc2aseg \
      --s "${SUBID}" \
      --annot "${BIDS_SUB}.${PARCEL}" \
      --o "${DSEG_PATH}"
  fi

  echo "=== Done ${SUBID} ==="
}

# -----------------------------
# Entry point
# -----------------------------
init_dirs
build_subject_list

# Choose subject based on SLURM array or CLI arg, else run all
if [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
  IDX="${SLURM_ARRAY_TASK_ID}"
  if (( IDX < 0 || IDX >= ${#SUBS[@]} )); then
    echo "Array index ${IDX} out of range (0..$(( ${#SUBS[@]} - 1 )))"; exit 1
  fi
  process_subject "${SUBS[$IDX]}"
elif [[ $# -ge 1 ]]; then
  process_subject "$1"
else
  echo "No array index and no subject arg provided — processing ALL (${#SUBS[@]}) subjects serially."
  for SUB in "${SUBS[@]}"; do
    process_subject "${SUB}"
  done
fi
