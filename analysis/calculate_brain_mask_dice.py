"""Create study-wide brain mask.

Create a brain mask for each subject and session in the study,
then limit it to the intersection of all the masks.
"""
import os
from glob import glob

import ants
import numpy as np

patterns = {
    "qsirecon": "qsirecon/derivatives/qsirecon-DSIStudio/{subject}/{session}/dwi/{subject}_{session}_acq-HBCD75_run-01_space-MNI152NLin2009cAsym_model-tensor_param-md_dwimap.nii.gz",
    "ihmt": "ihmt/{subject}/{session}/anat/{subject}_{session}_run-01_space-ihMTRAGEref_desc-brain_mask.nii.gz",
    "pymp2rage": "pymp2rage/{subject}/{session}/anat/{subject}_{session}_run-01_part-mag_space-T1w_desc-brain_mask.nii.gz",
    "qsm": "qsm/{subject}/{session}/anat/{subject}_{session}_acq-QSM_run-01_echo-1_part-mag_space-MEGRE_desc-brain_mask.nii.gz",
}
smriprep = "smriprep/{subject}/anat/{subject}_acq-MPRAGE_rec-refaced_run-01_space-MNI152NLin2009cAsym_desc-brain_mask.nii.gz"
smriprep_backup = "smriprep/{subject}/{session}/anat/{subject}_{session}_acq-MPRAGE_rec-refaced_run-01_space-MNI152NLin2009cAsym_desc-brain_mask.nii.gz"
mod_transforms = {
    "qsirecon": None,
    "ihmt": [
        "smriprep/{subject}/anat/{subject}_acq-MPRAGE_rec-refaced_run-01_from-T1w_to-MNI152NLin2009cAsym_mode-image_xfm.h5",
        "ihmt/{subject}/{session}/anat/{subject}_{session}_run-01_from-ihMTRAGEref_to-T1w_mode-image_xfm.mat",
    ],
    "pymp2rage": [
        "smriprep/{subject}/anat/{subject}_acq-MPRAGE_rec-refaced_run-01_from-T1w_to-MNI152NLin2009cAsym_mode-image_xfm.h5",
    ],
    "smriprep": None,
    "qsm": [
        "smriprep/{subject}/anat/{subject}_acq-MPRAGE_rec-refaced_run-01_from-T1w_to-MNI152NLin2009cAsym_mode-image_xfm.h5",
        "qsm/{subject}/{session}/anat/{subject}_{session}_run-01_from-MEGRE_to-T1w_mode-image_xfm.mat",
    ],
}

if __name__ == "__main__":
    bids_dir = '/cbica/projects/nibs/dset'
    deriv_dir = '/cbica/projects/nibs/derivatives'
    work_dir = '/cbica/projects/nibs/work/brain_mask'
    os.makedirs(work_dir, exist_ok=True)
    subject_dirs = sorted(glob(os.path.join(bids_dir, 'sub-*')))
    subjects = [os.path.basename(d) for d in subject_dirs]
    subjects = [s for s in subjects if not s.startswith('sub-PILOT')]
    masks = []
    subses_counter, scalar_counter = 0, 0
    for subject in subjects:
        print(subject, flush=True)
        session_dirs = sorted(glob(os.path.join(bids_dir, subject, 'ses-*')))
        sessions = [os.path.basename(d) for d in session_dirs]
        for session in sessions:
            print(session, flush=True)
            smriprep_file = os.path.join(deriv_dir, smriprep.format(subject=subject))
            if not os.path.exists(smriprep_file):
                print(f'{smriprep_file} does not exist', flush=True)
                smriprep_file = os.path.join(deriv_dir, smriprep_backup.format(subject=subject, session=session))
                if not os.path.exists(smriprep_file):
                    print(f'{smriprep_file} does not exist', flush=True)
                    continue

            smriprep_mask = ants.image_read(smriprep_file)
            if subses_counter == 0:
                smriprep_sum_mask = smriprep_mask
            else:
                smriprep_sum_mask = smriprep_sum_mask + smriprep_mask

            subses_counter += 1
            scalar_counter += 1

    ants.image_write(smriprep_sum_mask, os.path.join(work_dir, 'smriprep_brain_mask.nii.gz'))
    #ants.image_write(sum_mask, os.path.join(work_dir, 'scalar_brain_mask.nii.gz'))
    print(subses_counter, scalar_counter, flush=True)
