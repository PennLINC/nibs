#!/bin/bash
# Unload and load correct MATLAB versions
#module unload matlab/2022a
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
input_folder="/project/ftdc_volumetric/fw_bids/sub-${subjectID}/ses-${sessionID}/anat"
baseoutput_folder="/project/ftdc_pipeline/data/qsmxt_3t/QSM_sepia/sepia_results_noref/sub-${subjectID}/ses-${sessionID}"

# Create the output directory if it doesn't exist
mkdir -p "$baseoutput_folder/output"
output_folder="/project/ftdc_pipeline/data/qsmxt_3t/QSM_sepia/sepia_results_noref/sub-${subjectID}/ses-${sessionID}/output/Sepia"
# Specify the directory where the MATLAB function is located
MATLAB_SCRIPT_DIR="/project/ftdc_misc/spandey/sepia"
# Run the MATLAB script
matlab -nodisplay -nosplash -r "addpath(genpath('$MATLAB_SCRIPT_DIR')); try; updated2sepiaIO('$input_folder', '$output_folder'); catch e; disp(e.message); end; exit;"

# Check if MATLAB succeeded and handle errors
if [ $? -ne 0 ]; then
    echo "Error processing subject $subjectID session $sessionID"
else
    echo "Successfully processed subject $subjectID session $sessionID"
fi
