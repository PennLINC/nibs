"""Create study-wide brain mask.

Create a brain mask for each subject and session in the study,
then limit it to the intersection of all the masks.
"""
import os
from glob import glob

import ants

patterns = {
    "qsirecon": "qsirecon/derivatives/qsirecon-DSIStudio/{subject}/{session}/dwi/{subject}_{session}_acq-HBCD75_run-01_space-MNI152NLin2009cAsym_model-tensor_param-md_dwimap.nii.gz",
    "ihmt": "ihmt/{subject}/{session}/anat/{subject}_{session}_run-01_space-ihMTRAGEref_desc-brain_mask.nii.gz",
    "pymp2rage": "pymp2rage/{subject}/{session}/anat/{subject}_{session}_run-01_space-T1w_desc-brain_mask.nii.gz",
    "t1wt2w_ratio": "t1wt2w_ratio/{subject}/{session}/anat/{subject}_{session}_run-01_space-MNI152NLin2009cAsym_desc-brain_mask.nii.gz",
    "smriprep": "smriprep/{subject}/{session}/anat/{subject}_{session}_run-01_space-MNI152NLin2009cAsym_desc-brain_mask.nii.gz",
    "qsm": "qsm/{subject}/{session}/anat/{subject}_{session}_acq-QSM_run-01_echo-1_part-mag_space-MEGRE_desc-brain_mask.nii.gz",
}
transforms = {
    "qsirecon": None,
    "ihmt": [
        "smriprep/{subject}/anat/{subject}_acq-MPRAGE_rec-refaced_run-01_from-T1w_to-MNI152NLin2009cAsym_mode-image_xfm.h5",
        "ihmt/{subject}/{session}/anat/{subject}_{session}_run-01_from-ihMTRAGEref_to-T1w_mode-image_xfm.mat",
    ],
    "pymp2rage": [
        "smriprep/{subject}/anat/{subject}_acq-MPRAGE_rec-refaced_run-01_from-T1w_to-MNI152NLin2009cAsym_mode-image_xfm.h5",
    ],
    "t1wt2w_ratio": None,
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
    counter = 0
    for subject in subjects:
        session_dirs = sorted(glob(os.path.join(bids_dir, subject, 'ses-*')))
        sessions = [os.path.basename(d) for d in session_dirs]
        for session in sessions:
            for modality, pattern in patterns.items():
                in_file = os.path.join(deriv_dir, pattern.format(subject=subject, session=session))
                if not os.path.exists(in_file):
                    print(f'{in_file} does not exist')
                    continue

                mask = ants.image_read(in_file)
                if modality == "qsirecon":
                    # Binarize MD image
                    mask = (mask > 0).astype(int)
                    # Use QSIRecon MD image as target image for resampling
                    target_img = ants.image_read(in_file)

                transforms = transforms[modality]
                if transforms is not None:
                    transforms[0] = os.path.join(deriv_dir, transforms[0].format(subject=subject))
                    if len(transforms) > 1:
                        transforms[1] = os.path.join(deriv_dir, transforms[1].format(subject=subject, session=session))

                    mask = ants.apply_transforms(
                        fixed=target_img,
                        moving=mask,
                        transformlist=transforms,
                        interpolator='nearestNeighbor',
                    )
                elif modality != "qsirecon":
                    mask = mask.resample_image_to_target(target_img, interp_type='nearestNeighbor')

                if counter == 0:
                    sum_mask = mask
                else:
                    sum_mask = sum_mask + mask

                counter += 1

    ants.image_write(sum_mask, os.path.join(work_dir, 'brain_mask.nii.gz'))