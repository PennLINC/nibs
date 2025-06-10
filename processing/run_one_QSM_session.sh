#!/bin/bash
# Use SLURM bash shell


#SBATCH --mem=8G
#SBATCH --tmp=10G
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00

# Export arguments as environment variables
export SUBJECT=$1
export SESSION=$2

# get rid of botched run
rm /cbica/projects/pennlinc_qsm/output/SEPIA/sub-${SUBJECT}/ses-${SESSION}/sub-${SUBJECT}_ses-${SESSION}__*nii.gz

# Run MATLAB, treating inputs as strings
module load matlab
if [ ! -e /cbica/projects/pennlinc_qsm/output/SEPIA/sub-${SUBJECT}/ses-${SESSION}/sub-${SUBJECT}_ses-${SESSION}_Chimap.nii.gz ]; then
	matlab -nodisplay -r "addpath('/cbica/projects/pennlinc_qsm/scripts/tools/'); run_one_QSM_session('$SUBJECT', '$SESSION'); exit"
fi

# Transform the QSM NIFTI image by warping it to T1 space, and also apply brain mask to QSM, all in T1 space
QSM_path=/cbica/projects/pennlinc_qsm/output/SEPIA/sub-${SUBJECT}/ses-${SESSION}/sub-${SUBJECT}_ses-${SESSION}_Chimap.nii.gz
QSM_result=/cbica/projects/pennlinc_qsm/output/skullStripAndRegistration/sub-${SUBJECT}/ses-${SESSION}/QSM/sub-${SUBJECT}_ses-${SESSION}_QSM_regT1.nii.gz
# Apply transform to QSM
	antsApplyTransforms \
		-d 3 \
		-t /cbica/projects/pennlinc_qsm/output/skullStripAndRegistration/sub-${SUBJECT}/ses-${SESSION}/mag/sub-${SUBJECT}_ses-${SESSION}_reg_0GenericAffine.mat \
		-i ${QSM_path} -r /cbica/projects/pennlinc_qsm/output/skullStripAndRegistration/sub-${SUBJECT}/ses-${SESSION}/mag/sub-${SUBJECT}_ses-${SESSION}_reg_Warped.nii.gz\
		-o ${QSM_result}

# Apply brain mask to QSM to eliminate nonzero boxes surrounding the brain
	3dcalc -a ${QSM_result} -b /cbica/projects/pennlinc_qsm/output/skullStripAndRegistration/sub-${SUBJECT}/ses-${SESSION}/anat/sub-${SUBJECT}_ses-${SESSION}_T1_BrainExtractionMask.nii.gz -expr "a*b" \
	-prefix /cbica/projects/pennlinc_qsm/output/skullStripAndRegistration/sub-${SUBJECT}/ses-${SESSION}/QSM/sub-${SUBJECT}_ses-${SESSION}_QSM_regT1_masked.nii.gz -overwrite
