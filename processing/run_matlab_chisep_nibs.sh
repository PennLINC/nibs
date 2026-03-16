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

# Define the directories for input and output
input_file="/cbica/projects/nibs/dset/sub-${subjectID}/ses-${sessionID}/anat/sub-${subjectID}_ses-${sessionID}_acq-QSM_run-01_echo-1_part-mag_MEGRE.nii.gz"

# Create the output directory if it doesn't exist
output_folder="/cbica/projects/nibs/work/qsm-E12345+chisep+r2p/sub-${subjectID}/ses-${sessionID}/anat"
output_folder_e2345="/cbica/projects/nibs/work/qsm-E2345+chisep+r2p/sub-${subjectID}/ses-${sessionID}/anat"
r2map="/cbica/projects/nibs/dset/sub-${subjectID}/ses-${sessionID}/anat/sub-${subjectID}_ses-${sessionID}_run-01_space-MEGRE_desc-MESE_R2map.nii.gz"
r2prime="/cbica/projects/nibs/dset/sub-${subjectID}/ses-${sessionID}/anat/sub-${subjectID}_ses-${sessionID}_run-01_space-MEGRE_desc-MESE_R2primemap.nii.gz"
r2prime_e2345="/cbica/projects/nibs/dset/sub-${subjectID}/ses-${sessionID}/anat/sub-${subjectID}_ses-${sessionID}_run-01_space-MEGRE_desc-MEGRE+E2345_R2primemap.nii.gz"
r2star="/cbica/projects/nibs/dset/sub-${subjectID}/ses-${sessionID}/anat/sub-${subjectID}_ses-${sessionID}_run-01_space-MEGRE_desc-MEGRE_R2starmap.nii.gz"
r2star_e2345="/cbica/projects/nibs/dset/sub-${subjectID}/ses-${sessionID}/anat/sub-${subjectID}_ses-${sessionID}_run-01_space-MEGRE_desc-MEGRE+E2345_R2starmap.nii.gz"

# We run chi-sep six ways:
# 1. E12345+chisep+r2p
# 2. E2345+chisep+r2p
# 3. E12345+chisep+r2primenet
# 4. E2345+chisep+r2primenet
# 5. E12345+chisep+r2s
# 6. E2345+chisep+r2s

# Specify the directory where the MATLAB function is located
MATLAB_SCRIPT_DIR="/cbica/projects/nibs/software"
echo "matlab -nodisplay -nosplash -r \"addpath(genpath('$MATLAB_SCRIPT_DIR')); disp('MATLAB Path:'); disp(path); try; Chisep_script_shell_nibr2prime('$input_file', '$output_folder','$r2prime', '$output_folder_e2345'); catch e; disp(e.message); end; exit;\""

matlab -nodisplay -nosplash -r "addpath(genpath('$MATLAB_SCRIPT_DIR')); try; Chisep_script_shell_nibr2prime('$input_file', '$output_folder','$r2prime', '$output_folder_e2345'); catch e; disp(e.message); end; exit;"

# Check if MATLAB succeeded and handle errors
if [ $? -ne 0 ]; then
    echo "Error processing subject $subjectID session $sessionID"
else
    echo "Successfully processed subject $subjectID session $sessionID"
fi
