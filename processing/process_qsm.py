"""Process QSM data.

Steps:
1.  Average the magnitude images.
2.  Calculate R2* map.
3.  Coregister the averaged magnitude to the preprocessed T1w image from sMRIPrep.
4.  Extract the average magnitude image brain by applying the sMRIPrep brain mask.
5.  Warp T1w mask from T1w space into the QSM space by applying the inverse of the coregistration
    transform.
6.  Apply the mask in QSM space to magnitude images.
7.  Run SEPIA QSM estimation by calling the MATLAB script.
8.  Run chi-separation QSM estimation by calling the MATLAB script.
9.  Warp QSM derivatives to MNI152NLin2009cAsym space.

Notes:

- This doesn't apply the X-separation method or use the T2* map.
- Remember to name the QSM files with the suffix "Chimap".
- Chimap outputs should be in parts per million (ppm).
- The R2* map is calculated using the monoexponential fit.
"""
import json
import os

import ants
from bids.layout import BIDSLayout, Query
from nilearn import image

from utils import coregister_to_t1, fit_monoexponential, get_filename


def collect_run_data(layout, bids_filters):
    queries = {
        # SWI images from raw BIDS dataset
        'megre_mag': {
            'datatype': 'swi',
            'acquisition': 'QSM',
            'part': 'mag',
            'echo': Query.ANY,
            'suffix': 'swi',
            'extension': ['.nii', '.nii.gz'],
        },
        'megre_phase': {
            'datatype': 'swi',
            'acquisition': 'QSM',
            'part': 'phase',
            'echo': Query.ANY,
            'suffix': 'swi',
            'extension': ['.nii', '.nii.gz'],
        },
        # T1w-space R2 map from MESE pipeline
        'r2_map': {
            'datatype': 'anat',
            'space': 'T1w',
            'suffix': 'R2map',
            'extension': '.nii.gz',
        },
        # T1w-space T1w image from sMRIPrep
        't1w': {
            'datatype': 'anat',
            'space': Query.NONE,
            'resolution': Query.NONE,
            'desc': 'preproc',
            'suffix': 'T1w',
            'extension': ['.nii', '.nii.gz'],
        },
        # sMRIPrep T1w-space brain mask
        't1w_mask': {
            'datatype': 'anat',
            'space': Query.NONE,
            'resolution': Query.NONE,
            'desc': 'brain',
            'suffix': 'mask',
            'extension': ['.nii', '.nii.gz'],
        },
        # MNI-space T1w image from sMRIPrep
        't1w_mni': {
            'datatype': 'anat',
            'space': 'MNI152NLin2009cAsym',
            'desc': 'preproc',
            'suffix': 'T1w',
            'extension': ['.nii', '.nii.gz'],
        },
        # Normalization transform from sMRIPrep
        't1w2mni_xfm': {
            'datatype': 'anat',
            'from': 'T1w',
            'to': 'MNI152NLin2009cAsym',
            'mode': 'image',
            'suffix': 'xfm',
            'extension': '.h5',
        },
    }

    run_data = {}
    for key, query in queries.items():
        query = {**query, **bids_filters}
        files = layout.get(**query)
        if key.startswith('megre_'):
            if len(files) != 4:
                raise ValueError(f'Expected 4 files for {key}, got {len(files)}')
            else:
                run_data[key] = files
                continue

        elif len(files) > 1:
            raise ValueError(f'Expected 1 file for {key}, got {len(files)}')
        elif len(files) == 0:
            print(f'Expected 1 file for {key}, got {len(files)}')
            run_data[key] = None
            continue

        file = files[0]
        run_data[key] = file.path

    if len(run_data['megre_mag']) != len(run_data['megre_phase']):
        raise ValueError('Expected same number of magnitude and phase images')

    return run_data


def process_run(layout, run_data, out_dir, temp_dir):
    """Process a single run of QSM data.

    Parameters
    ----------
    layout : BIDSLayout
        BIDSLayout object for the dataset.
    run_data : dict
        Dictionary containing the paths to the MESE data.
    out_dir : str
        Path to the output directory.
    temp_dir : str
        Path to the temporary directory.
        Not currently used.
    """
    name_source = run_data['megre_mag'][0]

    # Calculate T2*, R2*, and S0 maps
    megre_metadata = [layout.get_metadata(f) for f in run_data['megre_mag']]
    echo_times = [m['EchoTime'] * 1000 for m in megre_metadata]
    t2s_img, r2s_img, s0_img = fit_monoexponential(
        in_files=run_data['megre_mag'],
        echo_times=echo_times,
    )
    t2s_filename = get_filename(
        name_source=name_source,
        layout=layout,
        out_dir=out_dir,
        entities={
            'datatype': 'anat',
            'space': 'QSM',
            'suffix': 'T2starmap',
            'extension': '.nii.gz',
        },
        dismiss_entities=['echo', 'part'],
    )
    t2s_img.to_filename(t2s_filename)

    r2s_filename = get_filename(
        name_source=name_source,
        layout=layout,
        out_dir=out_dir,
        entities={
            'datatype': 'anat',
            'space': 'QSM',
            'suffix': 'R2starmap',
            'extension': '.nii.gz',
        },
        dismiss_entities=['echo', 'part'],
    )
    r2s_img.to_filename(r2s_filename)

    s0_filename = get_filename(
        name_source=name_source,
        layout=layout,
        out_dir=out_dir,
        entities={
            'datatype': 'anat',
            'space': 'QSM',
            'suffix': 'S0map',
            'extension': '.nii.gz',
        },
        dismiss_entities=['echo', 'part'],
    )
    s0_img.to_filename(s0_filename)

    # Average the magnitude images
    mean_mag_img = image.mean_img(run_data['megre_mag'])
    mean_mag_filename = get_filename(
        name_source=name_source,
        layout=layout,
        out_dir=out_dir,
        entities={'space': 'QSM', 'desc': 'mean', 'suffix': 'swi'},
        dismiss_entities=['echo'],
    )
    mean_mag_img.to_filename(mean_mag_filename)

    # Coregister MEGRE data to preprocessed T1w
    coreg_transform = coregister_to_t1(
        name_source=name_source,
        layout=layout,
        in_file=mean_mag_filename,
        t1_file=run_data['t1w'],
        source_space='QSM',
        target_space='T1w',
    )

    # Warp R2 map from T1w space to QSM space
    r2_qsm_filename = get_filename(
        name_source=name_source,
        layout=layout,
        out_dir=out_dir,
        entities={'space': 'QSM', 'suffix': 'R2map'},
        dismiss_entities=['echo', 'part'],
    )
    r2_qsm_img = ants.apply_transforms(
        fixed=ants.image_read(mean_mag_filename),
        moving=ants.image_read(run_data['r2_map']),
        transformlist=[coreg_transform],
        whichtoinvert=[True],
    )
    ants.image_write(r2_qsm_img, r2_qsm_filename)

    # Calculate R2' (R2 - R2*)
    # R2' is used in chi-separation QSM estimation.
    r2_prime_filename = get_filename(
        name_source=name_source,
        layout=layout,
        out_dir=out_dir,
        entities={'space': 'QSM', 'suffix': 'R2prime'},
        dismiss_entities=['echo', 'part'],
    )
    r2_prime_img = r2_qsm_img - r2s_img
    ants.image_write(r2_prime_img, r2_prime_filename)

    # Now run the chi-separation QSM estimation
    ...

    # Warp T1w-space T2*map, R2*map, and S0map to MNI152NLin2009cAsym using normalization
    # transform from sMRIPrep and coregistration transform to sMRIPrep's T1w space.
    for file_ in [t2s_filename, r2s_filename, s0_filename]:
        suffix = os.path.basename(file_).split('_')[1].split('.')[0]
        out_file = get_filename(
            name_source=name_source,
            layout=layout,
            out_dir=out_dir,
            entities={'space': 'MNI152NLin2009cAsym', 'suffix': suffix},
        )
        reg_img = ants.apply_transforms(
            fixed=ants.image_read(run_data['t1w_mni']),
            moving=ants.image_read(file_),
            transformlist=[run_data['t1w2mni_xfm'], coreg_transform],
        )
        ants.image_write(reg_img, out_file)


if __name__ == '__main__':
    code_dir = '/Users/taylor/Documents/linc/nibs'
    in_dir = '/Users/taylor/Documents/datasets/nibs/dset'
    smriprep_dir = '/Users/taylor/Documents/datasets/nibs/derivatives/smriprep'
    out_dir = '/Users/taylor/Documents/datasets/nibs/derivatives/qsm'
    os.makedirs(out_dir, exist_ok=True)
    temp_dir = '/Users/taylor/Documents/datasets/nibs/work/qsm'
    os.makedirs(temp_dir, exist_ok=True)

    dataset_description = {
        'Name': 'NIBS QSM Derivatives',
        'BIDSVersion': '1.10.0',
        'DatasetType': 'derivative',
        'DatasetLinks': {
            'raw': in_dir,
            'smriprep': smriprep_dir,
        },
        'GeneratedBy': [
            {
                'Name': 'Custom code',
                'Description': 'Custom Python code combining ANTsPy and tedana.',
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
        derivatives=[smriprep_dir],
    )
    subjects = layout.get_subjects(suffix='swi')
    for subject in subjects:
        print(f'Processing subject {subject}')
        sessions = layout.get_sessions(subject=subject, suffix='swi')
        for session in sessions:
            print(f'Processing session {session}')
            megre_files = layout.get(
                subject=subject,
                session=session,
                acquisition='QSM',
                echo=1,
                part='mag',
                suffix='swi',
                extension=['.nii', '.nii.gz'],
            )
            for megre_file in megre_files:
                entities = megre_file.get_entities()
                entities.pop('echo')
                entities.pop('part')
                entities.pop('acquisition')
                run_data = collect_run_data(layout, entities)
                process_run(layout, run_data, out_dir, temp_dir)

    print('DONE!')
