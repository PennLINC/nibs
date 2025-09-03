# Submit an LSF array job to run process_qsm_prep.py for all subjects.
# Submit with bsub -J "qsm_chisep[1-10]" process_qsm_chisep.sh

module load matlab/2023a

subjects=("24037" "60501" "60505" "60514" "60516" "60518" "60519" "60520" "60522" "60526")

# Select one subject based on the index
subject=${subjects[$((LSB_JOBINDEX-1))]}

python process_qsm_chisep.py --subject-id ${subject}
