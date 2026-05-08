#!/bin/bash
# Load correct MATLAB versions
module load matlab/R2023A

# Check if two arguments are provided (subjectID and sessionID)
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <subjectID> <sessionID>"
    exit 1
fi

subjectID=$1
sessionID=$2

# Raw MEGRE echo-1 magnitude file (used only for niftiinfo header)
input_file="/cbica/projects/nibs/dset/sub-${subjectID}/ses-${sessionID}/anat/sub-${subjectID}_ses-${sessionID}_acq-QSM_run-01_echo-1_part-mag_MEGRE.nii.gz"

# SEPIA-preprocessed input folders (part-mag.nii.gz, part-phase.nii.gz, header.mat).
# E12345 and E2345 share the same SEPIA outputs within each echo-set folder; all
# three chi-sep variants per echo set read from the same folder.
sepia_e12345="/cbica/projects/nibs/work/qsm-E12345+sepia/sub-${subjectID}/ses-${sessionID}/anat"
sepia_e2345="/cbica/projects/nibs/work/qsm-E2345+sepia/sub-${subjectID}/ses-${sessionID}/anat"

# Chi-sep output folders (one per combination)
out_e12345_r2p="/cbica/projects/nibs/work/qsm-E12345+chisep+r2p/sub-${subjectID}/ses-${sessionID}/anat"
out_e2345_r2p="/cbica/projects/nibs/work/qsm-E2345+chisep+r2p/sub-${subjectID}/ses-${sessionID}/anat"
out_e12345_r2primenet="/cbica/projects/nibs/work/qsm-E12345+chisep+r2primenet/sub-${subjectID}/ses-${sessionID}/anat"
out_e2345_r2primenet="/cbica/projects/nibs/work/qsm-E2345+chisep+r2primenet/sub-${subjectID}/ses-${sessionID}/anat"
out_e12345_r2s="/cbica/projects/nibs/work/qsm-E12345+chisep+r2s/sub-${subjectID}/ses-${sessionID}/anat"
out_e2345_r2s="/cbica/projects/nibs/work/qsm-E2345+chisep+r2s/sub-${subjectID}/ses-${sessionID}/anat"

# Precomputed R2' maps (used only for the r2p combinations)
r2prime="/cbica/projects/nibs/dset/sub-${subjectID}/ses-${sessionID}/anat/sub-${subjectID}_ses-${sessionID}_run-01_space-MEGRE_desc-MEGRE+E12345_R2primemap.nii.gz"
r2prime_e2345="/cbica/projects/nibs/dset/sub-${subjectID}/ses-${sessionID}/anat/sub-${subjectID}_ses-${sessionID}_run-01_space-MEGRE_desc-MEGRE+E2345_R2primemap.nii.gz"

# Precomputed R2* maps (used only for the r2p combinations, to supply R2* to the
# CSF-mask step while the precomputed R2' map is used for chi-sep itself)
r2star="/cbica/projects/nibs/dset/sub-${subjectID}/ses-${sessionID}/anat/sub-${subjectID}_ses-${sessionID}_run-01_space-MEGRE_desc-MEGRE+E12345_R2starmap.nii.gz"
r2star_e2345="/cbica/projects/nibs/dset/sub-${subjectID}/ses-${sessionID}/anat/sub-${subjectID}_ses-${sessionID}_run-01_space-MEGRE_desc-MEGRE+E2345_R2starmap.nii.gz"

# Specify the directory where the MATLAB function is located
MATLAB_SCRIPT_DIR="/cbica/projects/nibs/software"

# We run chi-sep six ways:
# 1. E12345+chisep+r2p        echo_start=1  have_r2prime=1  is_scaling=0
# 2. E2345+chisep+r2p         echo_start=2  have_r2prime=1  is_scaling=0
# 3. E12345+chisep+r2primenet echo_start=1  have_r2prime=0  is_scaling=0
# 4. E2345+chisep+r2primenet  echo_start=2  have_r2prime=0  is_scaling=0
# 5. E12345+chisep+r2s        echo_start=1  have_r2prime=0  is_scaling=1
# 6. E2345+chisep+r2s         echo_start=2  have_r2prime=0  is_scaling=1

process_qsm_chisep() {
    local label="$1"
    local sepia_folder="$2"
    local r2p_path="$3"
    local outputa="$4"
    local echo_start="$5"
    local have_r2prime="$6"
    local is_scaling="$7"
    local r2s_path="$8"

    echo "Running chi-sep: ${label}"
    matlab -nodisplay -nosplash -r \
        "addpath(genpath('${MATLAB_SCRIPT_DIR}')); \
         try; \
           process_qsm_chisep('${input_file}','${sepia_folder}','${r2p_path}','${outputa}',${echo_start},${have_r2prime},${is_scaling},'${r2s_path}'); \
         catch e; \
           disp(e.message); \
         end; \
         exit;"
    if [ $? -ne 0 ]; then
        echo "Error processing ${label} for sub-${subjectID} ses-${sessionID}"
    else
        echo "Successfully processed ${label} for sub-${subjectID} ses-${sessionID}"
    fi
}

# 1. E12345+chisep+r2p
process_qsm_chisep "E12345+chisep+r2p" "$sepia_e12345" "$r2prime" "$out_e12345_r2p" 1 1 0 "$r2star"

# 2. E2345+chisep+r2p
process_qsm_chisep "E2345+chisep+r2p" "$sepia_e2345" "$r2prime_e2345" "$out_e2345_r2p" 2 1 0 "$r2star_e2345"

# 3. E12345+chisep+r2primenet
process_qsm_chisep "E12345+chisep+r2primenet" "$sepia_e12345" "" "$out_e12345_r2primenet" 1 0 0 ""

# 4. E2345+chisep+r2primenet
process_qsm_chisep "E2345+chisep+r2primenet" "$sepia_e2345" "" "$out_e2345_r2primenet" 2 0 0 ""

# 5. E12345+chisep+r2s
process_qsm_chisep "E12345+chisep+r2s" "$sepia_e12345" "" "$out_e12345_r2s" 1 0 1 ""

# 6. E2345+chisep+r2s
process_qsm_chisep "E2345+chisep+r2s" "$sepia_e2345" "" "$out_e2345_r2s" 2 0 1 ""
