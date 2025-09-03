# Submit an LSF array job to run process_qsm_prep.py for all subjects.
# Submit with bsub -J "qsm_chisep[1-10]" process_qsm_chisep.sh

module load matlab/2023a

subjects=("24037" "60501" "60505" "60514" "60516" "60518" "60519" "60520" "60522" "60526")

# Select one subject based on the index
subject=${subjects[$((LSB_JOBINDEX-1))]}

# Find all folders for the subject
folders=($(find /home/tsalo/nibs/data -name "sub${subject}*"))

# Process each folder
for folder in "${folders[@]}"; do
    echo "Processing folder ${folder}"
    mag_file="${folder}/python_mag.nii"
    pha_file="${folder}/python_phase.nii"
    r2_file="${folder}/python_r2.nii"
    sepia_head_file="/home/tsalo/nibs/code/nibs/sepia_header.mat"
    chisep_r2p_dir="${folder}/chisep_r2p"
    mkdir -p ${chisep_r2p_dir}
    matlab -nodisplay -nosplash -r "addpath('/home/tsalo/nibs/code/nibs'); run_chisep_script('${mag_file}', '${pha_file}', '${sepia_head_file}', '${chisep_r2p_dir}', '${r2_file}'); exit;"
    chisep_r2p_iron_file="${chisep_r2p_dir}/Paramagnetic.nii.gz"
    if [ ! -f ${chisep_r2p_iron_file} ]; then
        echo "Error: chi-separation QSM output file ${chisep_r2p_iron_file} not found"
        exit 1
    fi
    chisep_r2p_myelin_file="${chisep_r2p_dir}/Diamagnetic.nii.gz"
    if [ ! -f ${chisep_r2p_myelin_file} ]; then
        echo "Error: chi-separation QSM output file ${chisep_r2p_myelin_file} not found"
        exit 1
    fi
    chisep_no_r_dir="${folder}/chisep_no_r2p"
    mkdir -p ${chisep_no_r_dir}
    matlab -nodisplay -nosplash -r "addpath('/home/tsalo/nibs/code/nibs'); run_chisep_script('${mag_file}', '${pha_file}', '${sepia_head_file}', '${chisep_no_r_dir}'); exit;"
    chisep_no_r2p_iron_file="${chisep_no_r_dir}/Paramagnetic.nii.gz"
    if [ ! -f ${chisep_no_r2p_iron_file} ]; then
        echo "Error: chi-separation QSM output file ${chisep_no_r2p_iron_file} not found"
        exit 1
    fi
    chisep_no_r2p_myelin_file="${chisep_no_r2p_dir}/Diamagnetic.nii.gz"
    if [ ! -f ${chisep_no_r2p_myelin_file} ]; then
        echo "Error: chi-separation QSM output file ${chisep_no_r2p_myelin_file} not found"
        exit 1
    fi
done