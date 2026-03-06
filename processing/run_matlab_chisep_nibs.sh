#!/bin/bash
# Unload and load correct MATLAB versions
#module purge
module unload matlab/2022a
module load matlab/2023a
#module load spm/12
#module unload gcc/8.5.0
#module load gcc/10.2.0
module list

# Check if two arguments are provided (subjectID and sessionID)
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <subjectID> <sessionID>"
    exit 1
fi

subjectID=$1
sessionID=$2
# /project/nibs_data/dset/sub-24037/ses-01/anat/sub-24037_ses-01_run-01_space-MEGRE_desc-MESE_R2primemap.nii.gz
# Define the directories for input and output
input_folder="/project/nibs_data/dset/sub-${subjectID}/ses-${sessionID}/anat/sub-${subjectID}_ses-${sessionID}_acq-QSM_run-01_echo-1_part-mag_MEGRE.nii.gz"  
# commented this as this file was not available and looks like it has been moved and access has been denied
# input_folder="/project/ftdc_pipeline/spandey/sepia_results/sub-${subjectID}/ses-${sessionID}/output/Sepia_part-mag.nii.gz"
#filename="/project/nibs_data/dset/sub-${subjectID}/ses-${sessionID}/anat/outputE2345/sub-${subjectID}_ses-${sessionID}_total_r2s.nii"
# Create the output directory if it doesn't exist
output_folder="/project/nibs_data/dset/sub-${subjectID}/ses-${sessionID}/anat/output"
output_foldera="/project/nibs_data/dset/sub-${subjectID}/ses-${sessionID}/anat/outputE2345"
r2prime="/project/nibs_data/dset/sub-${subjectID}/ses-${sessionID}/anat/sub-${subjectID}_ses-${sessionID}_run-01_space-MEGRE_desc-MESE_R2primemap.nii.gz"
# Specify the directory where the MATLAB function is located
MATLAB_SCRIPT_DIR="/project/ftdc_misc/spandey"
echo "matlab -nodisplay -nosplash -r \"addpath(genpath('$MATLAB_SCRIPT_DIR')); disp('MATLAB Path:'); disp(path); try; Chisep_script_shell_nibr2prime('$input_folder', '$output_folder','$r2prime', '$output_foldera'); catch e; disp(e.message); end; exit;\""
# if [  ! -e ${output_foldera}/sub-${subjectID}_ses-${sessionID}_total_r2p.nii ]; then
#     # Command(s) to run if the file does NOT exist
#     matlab -nodisplay -nosplash -r "addpath(genpath('$MATLAB_SCRIPT_DIR')); try; Chisep_script_shell_nibr2prime('$input_folder', '$output_folder','$r2prime', '$output_foldera'); catch e; disp(e.message); end; exit;"
# fi
matlab -nodisplay -nosplash -r "addpath(genpath('$MATLAB_SCRIPT_DIR')); try; Chisep_script_shell_nibr2prime('$input_folder', '$output_folder','$r2prime', '$output_foldera'); catch e; disp(e.message); end; exit;"
# Run the MATLAB script with addpath and the function call
#matlab -nodisplay -nosplash -r "addpath(genpath('$MATLAB_SCRIPT_DIR')); disp('MATLAB Path:'); disp(path); try; updatedsepiaIO('$input', '$output'); catch e; disp(e.message); end; exit;"
# matlab -nodisplay -nosplash -r "addpath(genpath('$MATLAB_SCRIPT_DIR')); try; updated2sepiaIO_nibs('$input_folder', '$output_folder'); catch e; disp(e.message); end; exit;"

# MATLAB_SCRIPT_DIR="/project/ftdc_misc/spandey"

# # Run the MATLAB script
# matlab -nodisplay -nosplash -r "addpath(genpath('$MATLAB_SCRIPT_DIR')); try; Chisep_scriptshell_nibs('$input_folder', '$output_folder'); catch e; disp(e.message); end; exit;"

# Check if MATLAB succeeded and handle errors
if [ $? -ne 0 ]; then
    echo "Error processing subject $subjectID session $sessionID"
else
    echo "Successfully processed subject $subjectID session $sessionID"
fi
