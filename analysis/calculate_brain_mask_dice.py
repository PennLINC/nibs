"""Create study-wide brain mask.

Create a brain mask for each subject and session in the study,
then limit it to the intersection of all the masks.
"""

import os
from glob import glob

import ants
import numpy as np
import yaml

patterns = {
    'qsirecon': 'qsirecon/derivatives/qsirecon-DSIStudio/{subject}/{session}/dwi/{subject}_{session}_acq-HBCD75_run-01_space-MNI152NLin2009cAsym_model-tensor_param-md_dwimap.nii.gz',
    'ihmt': 'ihmt/{subject}/{session}/anat/{subject}_{session}_run-01_space-T1w_desc-brain_mask.nii.gz',
    'pymp2rage': 'pymp2rage/{subject}/{session}/anat/{subject}_{session}_run-01_part-mag_space-T1w_desc-brain_mask.nii.gz',
}
smriprep = 'smriprep/{subject}/anat/{subject}_acq-MPRAGE_rec-refaced_run-01_space-MNI152NLin2009cAsym_desc-brain_mask.nii.gz'
smriprep_backup = 'smriprep/{subject}/{session}/anat/{subject}_{session}_acq-MPRAGE_rec-refaced_run-01_space-MNI152NLin2009cAsym_desc-brain_mask.nii.gz'
mod_transforms = {
    'qsirecon': None,
    'ihmt': [
        'smriprep/{subject}/anat/{subject}_acq-MPRAGE_rec-refaced_run-01_from-T1w_to-MNI152NLin2009cAsym_mode-image_xfm.h5',
    ],
    'pymp2rage': [
        'smriprep/{subject}/anat/{subject}_acq-MPRAGE_rec-refaced_run-01_from-T1w_to-MNI152NLin2009cAsym_mode-image_xfm.h5',
    ],
}


def dice(input1, input2):
    r"""Calculate Dice coefficient between two arrays.

    Computes the Dice coefficient (also known as Sorensen index) between two binary images.

    The metric is defined as

    .. math::

        DC=\frac{2|A\cap B|}{|A|+|B|}

    , where :math:`A` is the first and :math:`B` the second set of samples (here: binary objects).
    This method was first proposed in :footcite:t:`dice1945measures` and
    :footcite:t:`sorensen1948method`.

    Parameters
    ----------
    input1/input2 : :obj:`numpy.ndarray`
        Numpy arrays to compare.
        Can be any type but will be converted into binary:
        False where 0, True everywhere else.

    Returns
    -------
    coef : :obj:`float`
        The Dice coefficient between ``input1`` and ``input2``.
        It ranges from 0 (no overlap) to 1 (perfect overlap).

    References
    ----------
    .. footbibliography::
    """
    input1 = np.atleast_1d(input1.astype(bool))
    input2 = np.atleast_1d(input2.astype(bool))

    intersection = np.count_nonzero(input1 & input2)

    size_i1 = np.count_nonzero(input1)
    size_i2 = np.count_nonzero(input2)

    if (size_i1 + size_i2) == 0:
        coef = 0
    else:
        coef = (2 * intersection) / (size_i1 + size_i2)

    return coef


if __name__ == '__main__':
    _cfg_path = os.path.join(os.path.dirname(__file__), '..', 'paths.yaml')
    with open(_cfg_path) as f:
        _cfg = yaml.safe_load(f)
    _root = _cfg['project_root']

    bids_dir = os.path.join(_root, _cfg['bids_dir'])
    deriv_dir = os.path.join(_root, 'derivatives')
    work_dir = os.path.join(_root, _cfg['work_dir'], 'brain_mask')
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
            print(f'\t{session}', flush=True)
            smriprep_file = os.path.join(deriv_dir, smriprep.format(subject=subject))
            if not os.path.exists(smriprep_file):
                print(f'{smriprep_file} does not exist', flush=True)
                smriprep_file = os.path.join(
                    deriv_dir, smriprep_backup.format(subject=subject, session=session)
                )
                if not os.path.exists(smriprep_file):
                    print(f'{smriprep_file} does not exist', flush=True)
                    continue

            smriprep_mask = ants.image_read(smriprep_file)
            ants.image_write(
                smriprep_mask, os.path.join(work_dir, f'{subject}_{session}_smriprep.nii.gz')
            )
            if subses_counter == 0:
                smriprep_sum_mask = smriprep_mask
            else:
                smriprep_sum_mask = smriprep_sum_mask + smriprep_mask

            subses_counter += 1

            for modality, pattern in patterns.items():
                in_file = os.path.join(deriv_dir, pattern.format(subject=subject, session=session))
                if not os.path.exists(in_file):
                    print(f'{in_file} does not exist')
                    continue

                mask = ants.image_read(in_file)
                if modality == 'qsirecon':
                    # Binarize MD image
                    mask = (mask > 0).astype('uint32')

                transforms = mod_transforms[modality]
                if transforms is not None:
                    transforms[0] = os.path.join(deriv_dir, transforms[0].format(subject=subject))
                    if len(transforms) > 1:
                        transforms[1] = os.path.join(
                            deriv_dir, transforms[1].format(subject=subject, session=session)
                        )

                    mask = ants.apply_transforms(
                        fixed=smriprep_mask,
                        moving=mask,
                        transformlist=transforms,
                        interpolator='nearestNeighbor',
                    )
                else:
                    mask = mask.resample_image_to_target(
                        smriprep_mask, interp_type='nearestNeighbor'
                    )

                dsi = dice(smriprep_mask.numpy(), mask.numpy())
                print(f'\t\t{modality}: {dsi:.4f}', flush=True)
                ants.image_write(
                    mask, os.path.join(work_dir, f'{subject}_{session}_{modality}.nii.gz')
                )

                if scalar_counter == 0:
                    sum_mask = mask
                else:
                    sum_mask = sum_mask + mask

                scalar_counter += 1

    ants.image_write(smriprep_sum_mask, os.path.join(work_dir, 'smriprep_brain_mask.nii.gz'))
    ants.image_write(sum_mask, os.path.join(work_dir, 'scalar_brain_mask.nii.gz'))
    print(subses_counter, scalar_counter, flush=True)
