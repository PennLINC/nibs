import json
import os

import matplotlib.pyplot as plt
import nibabel as nb
from pymp2rage import MP2RAGE


if __name__ == '__main__':
    in_dir = '/Users/taylor/Documents/datasets/nibs/dset/sub-02/ses-01/anat'
    out_dir = '/Users/taylor/Documents/datasets/nibs/derivatives/pymp2rage/sub-02/ses-01/anat'
    os.makedirs(out_dir, exist_ok=True)

    with open(os.path.join(in_dir, 'sub-02_ses-01_run-01_inv-1_MP2RAGE.json'), 'rb') as fobj:
        inv1_metadata = json.load(fobj)

    with open(os.path.join(in_dir, 'sub-02_ses-01_run-01_inv-2_MP2RAGE.json'), 'rb') as fobj:
        inv2_metadata = json.load(fobj)

    inversion_times = [
        inv1_metadata['InversionTime'],
        inv2_metadata['InversionTime'],
    ]
    flip_angles = [
        inv1_metadata['FlipAngle'],
        inv2_metadata['FlipAngle'],
    ]
    repetition_times = [
        inv1_metadata['RepetitionTimeExcitation'],
        inv2_metadata['RepetitionTimeExcitation'],
    ]
    n_slices = inv1_metadata['NumberShots']

    mp2rage = MP2RAGE(
        MPRAGE_tr=inv1_metadata['RepetitionTimePreparation'],
        invtimesAB=inversion_times,
        flipangleABdegree=flip_angles,
        B0=inv1_metadata['MagneticFieldStrength'],
        nZslices=n_slices,
        FLASH_tr=repetition_times,
        inv1=os.path.join(in_dir, 'sub-02_ses-01_run-01_inv-1_MP2RAGE.nii.gz'),
        inv2=os.path.join(in_dir, 'sub-02_ses-01_run-01_inv-2_MP2RAGE.nii.gz'),
    )
    t1map = mp2rage.t1map
    t1map_arr = t1map.get_fdata()
    t1map_arr = t1map_arr / 1000  # Convert from milliseconds to seconds
    t1map = nb.Nifti1Image(t1map_arr, t1map.affine, t1map.header)
    t1map.to_filename(os.path.join(out_dir, 'sub-02_ses-01_run-01_T1map.nii.gz'))
    mp2rage.t1w_uni.to_filename(os.path.join(out_dir, 'sub-02_ses-01_run-01_T1w.nii.gz'))

    plt.figure(figsize=(15, 6))
    mp2rage.plot_B1_effects()
    plt.savefig(os.path.join(out_dir, 'B1_effects.png'))

    # Correct for B1+ inhomogeneity (not yet implemented)
    # mp2rage.correct_for_B1('sub-02_ses-01_B1map.nii.gz')
    # mp2rage.t1_b1_corrected.to_filename(
    #     os.path.join(out_dir, 'sub-02_ses-01_run-01_desc-B1corrected_T1map.nii.gz')
    # )
    # mp2rage.t1w_uni_b1_corrected.to_filename(
    #     os.path.join(out_dir, 'sub-02_ses-01_run-01_desc-B1corrected_T1w.nii.gz')
    # )
