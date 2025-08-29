#!/bin/bash
# Unload and load correct MATLAB versions
#module unload matlab/2022a
# XXX: I only see matlab/2023a, not 2025a
module load matlab/2025a
module load spm/12

# Check if two arguments are provided (subjectID and sessionID)
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <subjectID> <sessionID>"
    exit 1
fi

subjectID=$1
sessionID=$2

# Define the directories for input and output
input_folder="/home/tsalo/nibs/dset/sub-${subjectID}/ses-${sessionID}/anat"
baseoutput_folder="/home/tsalo/nibs/derivatives/qsm/sub-${subjectID}/ses-${sessionID}"

# Create the output directory if it doesn't exist
mkdir -p "${baseoutput_folder}"
output_folder="${baseoutput_folder}/output/Sepia"
# Specify the directory where the MATLAB function is located
MATLAB_SCRIPT_DIR="/home/tsalo/nibs/sepia"
# Run the MATLAB script
matlab -nodisplay -nosplash -r "addpath(genpath('$MATLAB_SCRIPT_DIR')); try; updated2sepiaIO('$input_folder', '$output_folder'); catch e; disp(e.message); end; exit;"

# Check if MATLAB succeeded and handle errors
if [ $? -ne 0 ]; then
    echo "Error processing subject $subjectID session $sessionID"
else
    echo "Successfully processed subject $subjectID session $sessionID"
fi
