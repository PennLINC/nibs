import json
import os
import shutil

import ants
import nibabel as nb
from bids.layout import BIDSLayout, Query
from nilearn import image
from pymp2rage import MP2RAGE

from utils import get_filename


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
        query = {**bids_filters, **query}
        files = layout.get(**query)
        if len(files) > 1:
            raise ValueError(f'Expected 1 file for {key}, got {len(files)}: {query}')
        elif len(files) == 0:
            print(f'Expected 1 file for {key}, got {len(files)}: {query}')
            run_data[key] = None
            continue

        file = files[0]
        run_data[key] = file.path

    return run_data


def process_run(layout, run_data, out_dir, temp_dir):
    """Process a single MP2RAGE run.

    Parameters
    ----------
    layout : BIDSLayout
        BIDSLayout object.
    run_data : dict
        Dictionary of run data.
    out_dir : str
        Directory to write output files.
    temp_dir : str
        Directory to write temporary files.
    """
    name_source = run_data['inv1_magnitude']
    inv1_metadata = layout.get_metadata(run_data['inv1_magnitude'])
    inv2_metadata = layout.get_metadata(run_data['inv2_magnitude'])
    b1map_metadata = layout.get_metadata(run_data['b1_famp'])

    # Rescale b1_famp to percentage of flip angle
    scalar = b1map_metadata['FlipAngle'] * 10
    # scalar = 90 * 10  # original scalar from Manuel, but I think he had the wrong FA
    b1map_rescaled = image.math_img(f'img / {scalar}', img=run_data['b1_famp'])
    b1map_rescaled_file = os.path.join(temp_dir, os.path.basename(run_data['b1_famp']))
    b1map_rescaled.to_filename(b1map_rescaled_file)

    # Register b1_famp to inv1_magnitude using b1_anat with ANTs
    fixed_img = ants.image_read(run_data['inv1_magnitude'])
    moving_img = ants.image_read(run_data['b1_anat'])
    reg_output = ants.registration(
        fixed=fixed_img,
        moving=moving_img,
        type_of_transform='Rigid',
    )
    if len(reg_output['fwdtransforms']) != 1:
        print(
            f'Expected 1 transform, got {len(reg_output["fwdtransforms"])}: '
            f'{reg_output["fwdtransforms"]}'
        )
    fwd_transform = reg_output['fwdtransforms'][0]

    # Write the transform to a file
    fwd_transform_file = get_filename(
        name_source=name_source,
        layout=layout,
        out_dir=out_dir,
        entities={
            'datatype': 'fmap',
            'from': 'B1map',
            'to': 'T1map',
            'mode': 'image',
            'suffix': 'xfm',
            'extension': '.txt',
        },
    )
    shutil.copyfile(fwd_transform, fwd_transform_file)

    # Write the transform to a file
    inv_transform = reg_output['invtransforms'][0]
    inv_transform_file = get_filename(
        name_source=name_source,
        layout=layout,
        out_dir=out_dir,
        entities={
            'datatype': 'fmap',
            'from': 'T1map',
            'to': 'B1map',
            'mode': 'image',
            'suffix': 'xfm',
            'extension': '.txt',
        },
    )
    shutil.copyfile(inv_transform, inv_transform_file)

    # Apply the transform to b1_famp
    b1map_rescaled_img = ants.image_read(b1map_rescaled_file)
    b1map_rescaled_reg = ants.apply_transforms(
        fixed=fixed_img,
        moving=b1map_rescaled_img,
        transformlist=fwd_transform,
        interpolator='gaussian',
    )
    b1map_rescaled_reg_file = get_filename(
        name_source=name_source,
        layout=layout,
        out_dir=out_dir,
        entities={'datatype': 'fmap', 'space': 'T1map', 'suffix': 'B1map'},
        dismiss_entities=['inv', 'part'],
    )
    ants.image_write(b1map_rescaled_reg, b1map_rescaled_reg_file)

    # Apply the transform to b1_anat
    b1_anat_img = ants.image_read(run_data['b1_anat'])
    b1_anat_reg = ants.apply_transforms(
        fixed=fixed_img,
        moving=b1_anat_img,
        transformlist=fwd_transform,
        interpolator='gaussian',
    )
    b1_anat_reg_file = get_filename(
        name_source=name_source,
        layout=layout,
        out_dir=out_dir,
        entities={'datatype': 'fmap', 'space': 'T1map', 'suffix': 'B1anat'},
        dismiss_entities=['inv', 'part'],
    )
    ants.image_write(b1_anat_reg, b1_anat_reg_file)

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
    t1map_file = get_filename(
        name_source=name_source,
        layout=layout,
        out_dir=out_dir,
        entities={'suffix': 'T1map'},
        dismiss_entities=['inv', 'part'],
    )
    t1map.to_filename(t1map_file)

    t1w_uni_file = get_filename(
        name_source=name_source,
        layout=layout,
        out_dir=out_dir,
        entities={'suffix': 'T1w'},
        dismiss_entities=['inv', 'part'],
    )
    mp2rage.t1w_uni.to_filename(t1w_uni_file)

    # Correct for B1+ inhomogeneity
    mp2rage.correct_for_B1(b1map_rescaled_reg_file)

    t1map = mp2rage.t1map_b1_corrected
    t1map_arr = t1map.get_fdata()
    t1map_arr = t1map_arr / 1000  # Convert from milliseconds to seconds
    t1map = nb.Nifti1Image(t1map_arr, t1map.affine, t1map.header)
    t1map_b1_corrected_file = get_filename(
        name_source=name_source,
        layout=layout,
        out_dir=out_dir,
        entities={'suffix': 'T1map', 'desc': 'B1corrected'},
        dismiss_entities=['inv', 'part'],
    )
    t1map.to_filename(t1map_b1_corrected_file)
    t1w_uni_b1_corrected_file = get_filename(
        name_source=name_source,
        layout=layout,
        out_dir=out_dir,
        entities={'suffix': 'T1w', 'desc': 'B1corrected'},
        dismiss_entities=['inv', 'part'],
    )
    mp2rage.t1w_uni_b1_corrected.to_filename(t1w_uni_b1_corrected_file)


if __name__ == '__main__':
    code_dir = '/Users/taylor/Documents/linc/nibs'
    in_dir = '/Users/taylor/Documents/datasets/nibs/dset'
    out_dir = '/Users/taylor/Documents/datasets/nibs/derivatives/pymp2rage'
    os.makedirs(out_dir, exist_ok=True)
    temp_dir = '/Users/taylor/Documents/datasets/nibs/work/pymp2rage'
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

    layout = BIDSLayout(
        in_dir,
        config=os.path.join(code_dir, 'nibs_bids_config.json'),
        validate=False,
    )
    subjects = layout.get_subjects(suffix='MP2RAGE')
    for subject in subjects:
        print(f'Processing subject {subject}')
        sessions = layout.get_sessions(subject=subject, suffix='MP2RAGE')
        for session in sessions:
            print(f'Processing session {session}')
            inv1_magnitude_files = layout.get(
                subject=subject,
                session=session,
                inv=1,
                part=['mag', Query.NONE],
                suffix='MP2RAGE',
                extension=['.nii', '.nii.gz'],
            )
            for inv1_magnitude_file in inv1_magnitude_files:
                entities = inv1_magnitude_file.get_entities()
                entities.pop('inv')
                if 'part' in entities:
                    entities.pop('part')

                run_data = collect_run_data(layout, entities)
                process_run(layout, run_data, out_dir, temp_dir)

    print('DONE!')
