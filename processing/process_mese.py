"""Calculate T2* maps from MESE data.

This is still just a draft.
I need to calculate SDC from the first echo and apply that to the T2* map.
Plus we need proper output names.
"""

import json
import os

import nibabel as nb
from ants import apply_transforms, image_read, image_write, registration
from bids.layout import BIDSLayout, Query
from nilearn import image
from pymp2rage import MP2RAGE


def collect_run_data(layout, bids_filters):
    queries = {
        'inv1_magnitude': {
            'part': ['mag', Query.NONE],
            'inv': 1,
            'suffix': 'MP2RAGE',
            'extension': ['.nii', '.nii.gz'],
        },
        'inv1_phase': {
            'part': 'phase',
            'inv': 1,
            'suffix': 'MP2RAGE',
            'extension': ['.nii', '.nii.gz'],
        },
        'inv2_magnitude': {
            'part': ['mag', Query.NONE],
            'inv': 2,
            'suffix': 'MP2RAGE',
            'extension': ['.nii', '.nii.gz'],
        },
        'inv2_phase': {
            'part': 'phase',
            'inv': 2,
            'suffix': 'MP2RAGE',
            'extension': ['.nii', '.nii.gz'],
        },
        'b1_famp': {
            'datatype': 'fmap',
            'acquisition': 'famp',
            'suffix': 'TB1TFL',
            'extension': ['.nii', '.nii.gz'],
        },
        'b1_anat': {
            'datatype': 'fmap',
            'acquisition': 'anat',
            'suffix': 'TB1TFL',
            'extension': ['.nii', '.nii.gz'],
        },
    }

    run_data = {}
    for key, query in queries.items():
        query = {**query, **bids_filters}
        files = layout.get(**query)
        if len(files) > 1:
            raise ValueError(f'Expected 1 file for {key}, got {len(files)}')
        elif len(files) == 0:
            print(f'Expected 1 file for {key}, got {len(files)}')
            run_data[key] = None
            continue

        file = files[0]
        run_data[key] = file.path

    return run_data


def fit_monoexponential(in_files, echo_times):
    import numpy as np
    from tedana import io, decay

    data_cat, ref_img = io.load_data(in_files, n_echos=len(echo_times))

    # Fit model on all voxels, using all echoes
    mask = np.ones(data_cat.shape[0], dtype=int)
    masksum = mask * len(echo_times)

    t2s_limited, s0_limited, _, _ = decay.fit_monoexponential(
        data_cat=data_cat,
        echo_times=echo_times,
        adaptive_mask=masksum,
        report=False,
    )

    t2s_s = t2s_limited / 1000
    t2s_s[np.isinf(t2s_s)] = 0.5
    s0_limited[np.isinf(s0_limited)] = 0

    t2s_s_img = io.new_nii_like(ref_img, t2s_s)
    s0_img = io.new_nii_like(ref_img, s0_limited)
    return t2s_s_img, s0_img


def process_run(layout, run_data, out_dir, temp_dir):
    inv1_metadata = layout.get_metadata(run_data['inv1_magnitude'])
    inv2_metadata = layout.get_metadata(run_data['inv2_magnitude'])
    b1map_metadata = layout.get_metadata(run_data['b1_famp'])

    # Rescale b1_famp to percentage of flip angle
    scalar = b1map_metadata['FlipAngle'] * 10
    # scalar = 90 * 10  # original scalar from Manuel, but I think he had the wrong FA
    b1map_rescaled = image.math_img(f'img / {scalar}', img=run_data['b1_famp'])
    b1map_rescaled_path = os.path.join(temp_dir, os.path.basename(run_data['b1_famp']))
    b1map_rescaled.to_filename(b1map_rescaled_path)

    # Register b1_famp to inv1_magnitude using b1_anat with ANTs
    fixed_img = image_read(run_data['inv1_magnitude'])
    moving_img = image_read(run_data['b1_anat'])
    reg_output = registration(
        fixed=fixed_img,
        moving=moving_img,
        type_of_transform='Rigid',
    )
    fwd_transform = reg_output['fwdtransforms']
    b1map_rescaled_img = image_read(b1map_rescaled_path)
    b1map_rescaled_reg = apply_transforms(
        fixed=fixed_img,
        moving=b1map_rescaled_img,
        transformlist=fwd_transform,
        interpolator='gaussian',
    )
    b1map_rescaled_reg_path = os.path.join(out_dir, os.path.basename(run_data['b1_famp']))
    image_write(b1map_rescaled_reg, b1map_rescaled_reg_path)

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
        inv1=run_data['inv1_magnitude'],
        inv2=run_data['inv2_magnitude'],
        inv1ph=run_data['inv1_phase'],
        inv2ph=run_data['inv2_phase'],
    )
    t1map = mp2rage.t1map
    t1map_arr = t1map.get_fdata()
    t1map_arr = t1map_arr / 1000  # Convert from milliseconds to seconds
    t1map = nb.Nifti1Image(t1map_arr, t1map.affine, t1map.header)
    t1map.to_filename(os.path.join(out_dir, 'sub-02_ses-01_run-01_T1map.nii.gz'))
    mp2rage.t1w_uni.to_filename(os.path.join(out_dir, 'sub-02_ses-01_run-01_T1w.nii.gz'))

    # Correct for B1+ inhomogeneity
    mp2rage.correct_for_B1(b1map_rescaled_reg_path)
    t1map = mp2rage.t1map_b1_corrected
    t1map_arr = t1map.get_fdata()
    t1map_arr = t1map_arr / 1000  # Convert from milliseconds to seconds
    t1map = nb.Nifti1Image(t1map_arr, t1map.affine, t1map.header)
    t1map.to_filename(os.path.join(out_dir, 'sub-02_ses-01_run-01_desc-B1corrected_T1map.nii.gz'))
    mp2rage.t1w_uni_b1_corrected.to_filename(
        os.path.join(out_dir, 'sub-02_ses-01_run-01_desc-B1corrected_T1w.nii.gz')
    )


if __name__ == '__main__':
    code_dir = '/Users/taylor/Documents/linc/nibs'
    in_dir = '/Users/taylor/Documents/datasets/nibs/dset'
    out_dir = '/Users/taylor/Documents/datasets/nibs/derivatives/pymp2rage'
    os.makedirs(out_dir, exist_ok=True)
    temp_dir = '/Users/taylor/Documents/datasets/nibs/derivatives/pymp2rage/temp'
    os.makedirs(temp_dir, exist_ok=True)

    dataset_description = {
        'Name': 'NIBS MP2RAGE Derivatives',
        'BIDSVersion': '1.10.0',
        'DatasetType': 'derivative',
        'DatasetLinks': {
            'raw': in_dir,
        },
        'GeneratedBy': [
            {
                'Name': 'Custom code',
                'Description': 'Custom Python code combining ANTsPy and pymp2rage.',
                'CodeURL': 'https://github.com/PennLINC/nibs',
            }
        ],
    }
    with open(os.path.join(out_dir, 'dataset_description.json'), 'w') as f:
        json.dump(dataset_description, f, sort_keys=True, indent=4)

    layout = BIDSLayout(in_dir, config=os.path.join(code_dir, 'nibs_bids_config.json'))
    subjects = layout.get_subjects(suffix='MP2RAGE')
    for subject in subjects:
        print(f'Processing subject {subject}')
        sessions = layout.get_sessions(subject=subject, suffix='MP2RAGE')
        for session in sessions:
            print(f'Processing session {session}')
            run_data = collect_run_data(layout, {'subject': subject, 'session': session})
            process_run(layout, run_data, out_dir, temp_dir)

    print('DONE!')
