#!/bin/bash

# Function to submit reface job
submit_reface_job() {
    local input_file=$1

    # Create a temporary script for this specific job
    tmp_script=$(mktemp)
    echo "#!/bin/bash" > "${tmp_script}"
    echo "#SBATCH --job-name=reface" >> "${tmp_script}"
    echo "#SBATCH --time=12:00:00" >> "${tmp_script}"
    echo "#SBATCH --mem=16G" >> "${tmp_script}"
    echo "#SBATCH --output=logs/reface_%j.out" >> "${tmp_script}"
    echo "#SBATCH --error=logs/reface_%j.err" >> "${tmp_script}"

    echo "module load afni/2022_05_03" >> "${tmp_script}"
    echo "@afni_refacer_run \\" >> "${tmp_script}"
    echo "    -input \"${input_file}\" \\" >> "${tmp_script}"
    echo "    -mode_reface \\" >> "${tmp_script}"
    echo "    -no_images \\" >> "${tmp_script}"
    echo "    -overwrite \\" >> "${tmp_script}"
    echo "    -prefix \"${input_file}\"" >> "${tmp_script}"

    # Submit the job
    sbatch "${tmp_script}"
    rm -f "${tmp_script}"
}

# Create logs directory if it doesn't exist
mkdir -p logs

# Process T1w files
t1w_files=$(find /cbica/projects/nibs/dset/sub-*/ses-*/anat/*T1w.nii.gz)
for t1w_file in $t1w_files
do
    echo "Submitting job for: $t1w_file"
    submit_reface_job "${t1w_file}"
done

# Process T2w files
t2w_files=$(find /cbica/projects/nibs/dset/sub-*/ses-*/anat/*T2w.nii.gz)
for t2w_file in $t2w_files
do
    echo "Submitting job for: $t2w_file"
    submit_reface_job "${t2w_file}"
done

# Process MP2RAGE files
mp2rage_files=$(find /cbica/projects/nibs/dset/sub-*/ses-*/anat/*part-mag*MP2RAGE.nii.gz)
for mp2rage_file in $mp2rage_files
do
    echo "Submitting job for: $mp2rage_file"
    submit_reface_job "${mp2rage_file}"
done
