#! /bin/bash

# This is the overall pipeline to skull-strip and make a template from T1 and magnitude image
# It loop through all data in bids_directory

# get current date
date=$(date '+%Y-%m-%d')
logfile=/cbica/projects/pennlinc_qsm/scripts/02_1_logfile_${date}.txt
sumfile=/cbica/projects/pennlinc_qsm/scripts/02_1_summary_${date}.txt

echo -n >${logfile}
echo -e "Starting to skull-strip and align the images">>${logfile}

echo -n >${sumfile}
echo -e "bblid,sesid,number of anat,number of mag">>${sumfile} # COME BACK TO DIS PLS

# read the whole directory in loop
for f in $(ls -d /cbica/projects/pennlinc_qsm/data/bids_directory/sub*); do
	bblid=$(basename $f);
	for ses in $(ls -d /cbica/projects/pennlinc_qsm/data/bids_directory/${bblid}/s*); do
		sesid=$(basename $ses);
		# automatically get rid of potential _2 in order to collect everything in original one
		sesid=${sesid%_2}

		# make directory for storing outputs (bblid = sub-12345, sesid = ses-12345)
		# output_dir=/cbica/projects/pennlinc_qsm/output/templates/${bblid}/${sesid} # changed the target output dir
		output_dir=/cbica/projects/pennlinc_qsm/output/skullStripAndRegistration/${bblid}/${sesid} # changed the target output dir

		# get only the numerical id (for mag and qsm path)
		bblid_num=$(echo "${bblid}" | grep -o -E '[0-9]+')
		sesid_num=$(echo "${sesid}" | grep -o -E '[0-9]+')

		# find T1 and mag full paths to input into the single subj function (save as variables)
		T1_path=/cbica/projects/pennlinc_qsm/data/bids_directory/${bblid}/${sesid}/anat/${bblid}_${sesid}_T1w
		mag_path=/cbica/projects/pennlinc_qsm/data/bids_directory/${bblid}/${sesid}/qsm/${bblid}_${sesid}_mag

		# skull-strip T1 and magnitude
		# if [ ! -e ${output_dir}/QSM/${bblid}_${sesid}_masked_mag_in_mag_space.nii.gz ]; then
		if [ ! -e ${output_dir}/mag/${bblid}_${sesid}_masked_mag_in_mag_space.nii.gz ]; then
			echo -e "Perform skull-strip for ${bblid_num} ${sesid_num}">>${logfile}
			qsub -N SS_${bblid}_${sesid} -l h_vmem=22.5G -l s_vmem=22G \
			/cbica/projects/pennlinc_qsm/scripts/tools/proc_one_mag_and_T1_sub.sh \
			${T1_path} ${mag_path} ${output_dir} ${bblid} ${sesid} ${sumfile}
		else
			echo -e "Already processed ${bblid_num} ${sesid_num}">>${logfile}

			# Put in the summary of number of files in the directory in the summary file
			numAnat=$(ls -l ${output_dir}/anat/*.nii.gz | wc -l)
			numMag=$(ls -l ${output_dir}/mag/*.nii.gz | wc -l)

			echo "${bblid_num},${sesid_num},${numAnat},${numMag}">>${sumfile}

			continue
		fi
	done
done

