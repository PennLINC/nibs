"""Process QSM data.

Steps:

0.  Load matlab/R2020B.
1.  Run SEPIA QSM estimation by calling the MATLAB script.

Notes:

- Remember to name the QSM files with the suffix "Chimap".
- Chimap outputs should be in parts per million (ppm).
"""

from __future__ import annotations

import argparse
import os
import subprocess
from pprint import pformat

import nibabel as nb
from bids.layout import BIDSLayout, Query
from nilearn import image
from scipy.io import loadmat, savemat

from utils import get_filename, load_config

CFG = load_config()
CODE_DIR = CFG['code_dir']


def collect_run_data(layout: object, bids_filters: dict) -> dict[str, str]:
    """Collect multi-echo GRE images for SEPIA QSM estimation.

    Parameters
    ----------
    layout : bids.BIDSLayout
        BIDSLayout indexing the dataset and derivatives.
    bids_filters : dict
        BIDS entity filters (e.g., subject, session, run) to narrow the query.

    Returns
    -------
    run_data : dict
        Mapping of descriptive keys to resolved file paths.
    """
    queries = {
        # SWI images from raw BIDS dataset
        'megre_mag': {
            'datatype': 'anat',
            'acquisition': 'QSM',
            'part': 'mag',
            'echo': Query.ANY,
            'space': Query.NONE,
            'desc': Query.NONE,
            'suffix': 'MEGRE',
            'extension': ['.nii', '.nii.gz'],
        },
        'megre_phase': {
            'datatype': 'anat',
            'acquisition': 'QSM',
            'part': 'phase',
            'echo': Query.ANY,
            'space': Query.NONE,
            'desc': Query.NONE,
            'suffix': 'MEGRE',
            'extension': ['.nii', '.nii.gz'],
        },
        'mask': {
            'datatype': 'anat',
            'acquisition': 'QSM',
            'part': 'mag',
            'echo': 1,
            'space': 'MEGRE',
            'desc': 'brain',
            'suffix': 'mask',
            'extension': ['.nii', '.nii.gz'],
        },
    }

    run_data = {}
    for key, query in queries.items():
        query = {**bids_filters, **query}
        files = layout.get(**query)
        if key.startswith('megre_'):
            if len(files) != 5:
                raise ValueError(f'Expected 5 files for {key}, got {len(files)}')
            else:
                run_data[key] = sorted([f.path for f in files])
                continue

        elif len(files) != 1:
            raise ValueError(f'Expected 1 file for {key}, got {len(files)} with query {query}')

        file = files[0]
        run_data[key] = file.path

    if len(run_data['megre_mag']) != len(run_data['megre_phase']):
        raise ValueError('Expected same number of magnitude and phase images')

    print(f'Collected run data:\n{pformat(run_data, indent=4)}', flush=True)
    return run_data


def process_run(layout, run_data, out_dir, temp_dir, subject_id, session):
    """Process a single run of QSM data.

    Parameters
    ----------
    layout : BIDSLayout
        BIDSLayout object for the dataset.
    run_data : dict
        Dictionary containing the paths to the QSM data.
    out_dir : str
        Path to the output directory.
    temp_dir : str
        Path to the temporary directory.
    subject_id : str
        BIDS subject label (without 'sub-' prefix).
    session : str
        BIDS session label (without 'ses-' prefix).
    """
    name_source = run_data['megre_mag'][0]

    # Create concatenated versions of files
    for version in ['E12345', 'E2345']:
        if version == 'E12345':
            mag_concat_img = image.concat_imgs(run_data['megre_mag'])
            phase_concat_img = image.concat_imgs(run_data['megre_phase'])
        else:
            mag_concat_img = image.concat_imgs(run_data['megre_mag'][1:])
            phase_concat_img = image.concat_imgs(run_data['megre_phase'][1:])

        sepia_work_dir = os.path.join(
            CFG['work_dir'], f'qsm-{version}+sepia', f'sub-{subject_id}', f'ses-{session}', 'anat'
        )
        os.makedirs(sepia_work_dir, exist_ok=True)
        mag_concat_file = os.path.join(
            sepia_work_dir, f'sub-{subject_id}_ses-{session}_part-mag_desc-concat_MEGRE.nii.gz'
        )
        phase_concat_file = os.path.join(
            sepia_work_dir, f'sub-{subject_id}_ses-{session}_part-phase_desc-concat_MEGRE.nii.gz'
        )
        mag_concat_img.to_filename(mag_concat_file)
        phase_concat_img.to_filename(phase_concat_file)


def _get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--subject-id',
        type=lambda label: label.removeprefix('sub-'),
        required=True,
    )
    return parser


def _main(argv=None):
    """Run the process_qsm_sepia workflow."""
    options = _get_parser().parse_args(argv)
    kwargs = vars(options)
    main(**kwargs)


def main(subject_id):
    in_dir = CFG['bids_dir']
    smriprep_dir = CFG['derivatives']['smriprep']
    mese_dir = CFG['derivatives']['mese']
    out_dir = CFG['derivatives']['qsm']
    os.makedirs(out_dir, exist_ok=True)
    temp_dir = os.path.join(CFG['work_dir'], 'sepia')
    os.makedirs(temp_dir, exist_ok=True)

    layout = BIDSLayout(
        in_dir,
        config=os.path.join(CODE_DIR, 'configuration', 'nibs_bids_config.json'),
        validate=False,
        derivatives=[smriprep_dir, mese_dir, out_dir],
    )

    print(f'Processing subject {subject_id}', flush=True)
    sessions = layout.get_sessions(subject=subject_id, suffix='MEGRE')
    for session in sessions:
        print(f'Processing session {session}', flush=True)
        megre_files = layout.get(
            subject=subject_id,
            session=session,
            acquisition='QSM',
            echo=1,
            part='mag',
            suffix='MEGRE',
            extension=['.nii', '.nii.gz'],
        )
        for megre_file in megre_files:
            entities = megre_file.get_entities()
            entities.pop('echo')
            entities.pop('part')
            entities.pop('acquisition')
            try:
                run_data = collect_run_data(layout, entities)
            except ValueError as e:
                print(f'Failed {megre_file}', flush=True)
                print(e, flush=True)
                continue
            fname = os.path.basename(megre_file.path).split('.')[0]
            run_temp_dir = os.path.join(temp_dir, fname.replace('-', '').replace('_', ''))
            os.makedirs(run_temp_dir, exist_ok=True)
            process_run(layout, run_data, out_dir, run_temp_dir, subject_id, session)

    print('DONE!', flush=True)


if __name__ == '__main__':
    _main()
