#! /bin/bash

# These processes are done sequentially to extract skull-stripped T1 and magnitude images
# 1) average the magnitude images
# 2) register the averaged magnitude to original T1 image
# 3) extract T1 brain, thus we get a skull-stripped T1 image
# 4) extract the average magnitude image brain by multiplying it with T1 brain mask
# 5) Warp T1 Mask from T1 space into the magnitude space by inverse transform the T1 mask using mat file from 2)
# 6) Apply the mask in magnitude space to magnitude images by doing magnitude * T1 brain mask that is in magnitude space

# initialization
T1=$1
mag=$2
output_dir=$3
subid=$4
sesid=$5
sumfile=$6
ext_template=/cbica/projects/pennlinc_qsm/data/OASIS/T_template0.nii.gz
ext_prob_mask=/cbica/projects/pennlinc_qsm/data/OASIS/T_template0_BrainCerebellumProbabilityMask.nii.gz
ext_reg_mask=/cbica/projects/pennlinc_qsm/data/OASIS/T_template0_BrainCerebellumRegistrationMask.nii.gz
ants_dir=/cbica/projects/pennlinc_qsm/software/ANTs
output_anat=${output_dir}/anat
output_mag=${output_dir}/mag
mkdir -p ${output_anat}
mkdir -p ${output_mag}
mkdir -p ${output_dir}/QSM

# 1) average magnitude
avg_mag=${output_mag}/${subid}_${sesid}_mag_avg.nii.gz
if [ ! -e ${output_mag}/${subid}_${sesid}_mag_avg.nii.gz ]; then
	3dTstat -mean -prefix ${avg_mag} ${mag}.nii.gz
fi

# 2) register 1) to T1
if [ ! -e ${output_mag}/${subid}_${sesid}_reg_Warped.nii.gz ]; then
	antsRegistration --dimensionality 3 --float 0 --output [${output_mag}/${subid}_${sesid}_reg_, \
	${output_mag}/${subid}_${sesid}_reg_Warped.nii.gz, ${output_mag}/${subid}_${sesid}_reg_invWarped.nii.gz] \
	--interpolation BSpline \
	--winsorize-image-intensities [0.005,0.995] --use-histogram-matching 0 \
	--initial-moving-transform  [${T1}.nii.gz, ${avg_mag}, 1] \
	--transform Rigid[0.1] --metric MI[${T1}.nii.gz, ${avg_mag}, 1, 32, Regular, 0.25] \
	--convergence [1000x500x250x100, 1e-6, 10] --shrink-factors 8x4x2x1 --smoothing-sigmas 3x2x1x0vox
fi

# 3) extract T1
# Version with ANTs BrainExtraction tool
# extract_T1=${output_anat}/${subid}_${sesid}_T1_
# if [ ! -e ${output_anat}/${subid}_${sesid}_T1_BrainExtractionBrain.nii.gz ]; then
# 	extract_T1=${output_anat}/${subid}_${sesid}_T1_
# 	${ants_dir}/Scripts/antsBrainExtraction.sh -d 3 -a ${T1}.nii.gz -e ${ext_template} -m ${ext_prob_mask} -o ${extract_T1} -f ${ext_reg_mask}
# fi
# Version with SynthStrip (added April 24th, 2023)
extract_T1=${output_anat}/${subid}_${sesid}_T1_
if [ ! -e ${output_anat}/${subid}_${sesid}_T1_BrainExtractionBrain.nii.gz ]; then
	extract_T1=${output_anat}/${subid}_${sesid}_T1_
	python3 /cbica/projects/pennlinc_qsm/software/synthstrip/synthstrip-singularity \
		-i ${T1}.nii.gz \
		-o ${extract_T1}BrainExtractionBrain.nii.gz \
		-m ${extract_T1}BrainExtractionMask.nii.gz \
		--no-csf --border 2
fi

# 4) extract average mag
extract_avg_mag=${output_mag}/${subid}_${sesid}_mag_brain
if [ ! -e ${output_mag}/${subid}_${sesid}_mag_brain.nii.gz ]; then
	extract_avg_mag=${output_mag}/${subid}_${sesid}_mag_brain
	3dcalc -a ${output_mag}/${subid}_${sesid}_reg_Warped.nii.gz \
	-b ${extract_T1}BrainExtractionMask.nii.gz -expr "a*b" \
	-prefix ${extract_avg_mag}.nii.gz
fi

### Extra steps for getting custom mask in magnitude space
#### 5) Warp T1 Mask from T1 space into the magnitude space
#### Uses T1 Mask as input (moving image) 
#### and average magnitude as fixed image to get warped into that space
#### Note that the -t [mat, 1] is using the mat file but perform inverse transformation
if [ ! -e ${output_dir}/anat/${subid}_${sesid}_T1BrainMask_in_mag_space.nii.gz ]; then
	antsApplyTransforms \
		-d 3 \
		-t [${output_mag}/${subid}_${sesid}_reg_0GenericAffine.mat, 1] \
		-i ${extract_T1}BrainExtractionMask.nii.gz \
		-r ${avg_mag} \
		-n NearestNeighbor \
		-o ${output_dir}/anat/${subid}_${sesid}_T1BrainMask_in_mag_space.nii.gz
fi


#### 6) Apply the mask in magnitude space to magnitude images
if [ ! -e ${output_mag}/${subid}_${sesid}_masked_mag_in_mag_space.nii.gz ]; then
	3dcalc -a ${mag}.nii.gz \
	-b ${output_dir}/anat/${subid}_${sesid}_T1BrainMask_in_mag_space.nii.gz \
	-expr "a*b" \
	-prefix ${output_mag}/${subid}_${sesid}_masked_mag_in_mag_space.nii.gz
fi

# Put in the summary of number of files in the directory in the summary file
numAnat=$(ls -l ${output_anat}/*.nii.gz | wc -l)
numMag=$(ls -l ${output_mag}/*.nii.gz | wc -l)
# get only the numerical id (for mag and qsm path)
subid_num=$(echo "${subid}" | grep -o -E '[0-9]+')
sesid_num=$(echo "${sesid}" | grep -o -E '[0-9]+')

echo "${subid_num},${sesid_num},${numAnat},${numMag}">>${sumfile}
