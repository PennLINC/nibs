#! /bin/bash

# This script runs the MEDI toolbox by first checking if the QSM dicoms is missing
# then, check if the given session has already been processed
module load slurm
# get current date
date=$(date '+%Y-%m-%d')
logfile=/cbica/projects/pennlinc_qsm/scripts/03_1_logfile_${date}.txt
sumfile=/cbica/projects/pennlinc_qsm/scripts/03_1_summary_${date}.txt

echo -n >${logfile}
echo -e "Starting to run the SEPIA toolbox">>${logfile}

echo -n >${sumfile}
#echo -e "bblid,sesid,number of dicoms,number of NIFTIs,masked QSM present">>${sumfile} # COME BACK TO DIS PLS

# loop through all raw data sessions
for session in $(ls -d /cbica/projects/pennlinc_qsm/data/bids_directory/sub-*/*); do
	sessionID=$(basename $session)
	sessionID="${sessionID#ses-}"
	subID=$(basename $(dirname $session))
	subID="${subID#sub-}"
	#qsm_dicoms=${session}/all_dicoms


	out_dir=/cbica/projects/pennlinc_qsm/output/SEPIA/sub-${subID}/ses-${sessionID}

	# check if the given session has already been processed
	if [ -e ${out_dir}/sub-${subID}_ses-${sessionID}_Chimap.nii.gz ]; then
		echo -e "${session} already processed">>${logfile}
		continue
	fi

	out_dir_for_transformation=/cbica/projects/pennlinc_qsm/output/skullStripAndRegistration/sub-${subID}/ses-${sessionID}
	mask=${out_dir_for_transformation}/anat/sub-${subID}_ses-${sessionID}_T1BrainMask_in_mag_space.nii.gz
	if [ -e ${mask} ]; then

		echo -e "Running SEPIA for sub-${subID} ses-${sessionID}">>${logfile}
		/cbica/software/external/slurm/current/bin/sbatch -e /cbica/projects/pennlinc_qsm/scripts/logs/${subID}_${sessionID}.err -o /cbica/projects/pennlinc_qsm/scripts/logs/${subID}_${sessionID}.out \
		/cbica/projects/pennlinc_qsm/scripts/tools/process_qsm_02a.sh ${subID} ${sessionID}
	else
		# did not have the custom mask to pass into MEDI toolbox
		echo -e "Does not have custom mask ${mask} to pass into MEDI for sub-${subID} ses-${sessionID}">>${logfile}
		echo "${subID},${sessionID},0,0,0">>${sumfile}
	fi
done
