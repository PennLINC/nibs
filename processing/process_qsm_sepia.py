"""Process QSM data.

Steps:

0.  Load matlab/R2020B.
1.  Run SEPIA QSM estimation by calling the MATLAB script.

Notes:

- Remember to name the QSM files with the suffix "Chimap".
- Chimap outputs should be in parts per million (ppm).
"""

import argparse
import os
import subprocess
from pprint import pprint

import nibabel as nb
from bids.layout import BIDSLayout, Query
from nilearn import image
from scipy.io import loadmat, savemat

from utils import get_filename

CODE_DIR = '/cbica/projects/nibs/code'


def collect_run_data(layout, bids_filters):
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

    pprint(run_data)

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
    """
    name_source = run_data['megre_mag'][0]
    header_file = os.path.join(CODE_DIR, 'processing', 'sepia_header.mat')
    header_struct = loadmat(header_file)
    header_struct['B0_dir'] = header_struct['B0_dir'].astype(float)
    header_struct['B0'] = header_struct['B0'].astype(float)

    # Create concatenated versions of files
    for version in ['E12345', 'E2345']:
        if version == 'E12345':
            mag_concat_img = image.concat_imgs(run_data['megre_mag'])
            phase_concat_img = image.concat_imgs(run_data['megre_phase'])
        else:
            mag_concat_img = image.concat_imgs(run_data['megre_mag'][1:])
            phase_concat_img = image.concat_imgs(run_data['megre_phase'][1:])
            header_struct['TE'] = header_struct['TE'][:, 1:]

        out_header_file = os.path.join(temp_dir, f'python_header_{version}.mat')
        savemat(out_header_file, header_struct)
        mag_concat_file = os.path.join(temp_dir, f'python_mag_{version}.nii.gz')
        phase_concat_file = os.path.join(temp_dir, f'python_phase_{version}.nii.gz')
        mag_concat_img.to_filename(mag_concat_file)
        phase_concat_img.to_filename(phase_concat_file)

        # Run SEPIA QSM estimation
        sepia_dir = os.path.join(temp_dir, f'sepia_{version}')
        sepia_script = os.path.join(CODE_DIR, 'processing', 'process_qsm_sepia.m')
        with open(sepia_script, 'r') as fobj:
            base_sepia_script = fobj.read()

        modified_sepia_script = (
            base_sepia_script.replace("{{ phase_file }}", mag_concat_file)
            .replace("{{ mag_file }}", phase_concat_file)
            .replace("{{ output_dir }}", sepia_dir)
            .replace("{{ header_file }}", out_header_file)
            .replace("{{ mask_file }}", run_data['mask'])
        )

        out_sepia_script = os.path.join(temp_dir, f'process_qsm_sepia_{version}.m')
        with open(out_sepia_script, "w") as fobj:
            fobj.write(modified_sepia_script)

        subprocess.run(
            [
                "matlab",
                "-nodisplay",
                "-nosplash",
                "-nodesktop",
                "-r",
                f"run('{out_sepia_script}'); exit;",
            ],
        )
        sepia_chimap_file = f'{sepia_dir}_Chimap.nii.gz'
        if not os.path.isfile(sepia_chimap_file):
            raise FileNotFoundError(f'SEPIA QSM output file {sepia_chimap_file} not found')

        sepia_chimap_img = nb.load(sepia_chimap_file)
        sepia_chimap_filename = get_filename(
            name_source=name_source,
            layout=layout,
            out_dir=out_dir,
            entities={'space': 'MEGRE', 'desc': f'{version}+sepia', 'suffix': 'Chimap'},
            dismiss_entities=['acquisition', 'echo', 'part', 'inv', 'reconstruction'],
        )
        sepia_chimap_img.to_filename(sepia_chimap_filename)


def _get_parser():
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
    in_dir = '/cbica/projects/nibs/dset'
    smriprep_dir = '/cbica/projects/nibs/derivatives/smriprep'
    mese_dir = '/cbica/projects/nibs/derivatives/mese'
    out_dir = '/cbica/projects/nibs/derivatives/qsm'
    os.makedirs(out_dir, exist_ok=True)
    temp_dir = '/cbica/projects/nibs/work/sepia'
    os.makedirs(temp_dir, exist_ok=True)

    layout = BIDSLayout(
        in_dir,
        config=os.path.join(CODE_DIR, 'nibs_bids_config.json'),
        validate=False,
        derivatives=[smriprep_dir, mese_dir, out_dir],
    )

    print(f'Processing subject {subject_id}')
    sessions = layout.get_sessions(subject=subject_id, suffix='MEGRE')
    for session in sessions:
        print(f'Processing session {session}')
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
                print(f'Failed {megre_file}')
                print(e)
                continue
            fname = os.path.basename(megre_file.path).split('.')[0]
            run_temp_dir = os.path.join(temp_dir, fname.replace('-', '').replace('_', ''))
            os.makedirs(run_temp_dir, exist_ok=True)
            process_run(layout, run_data, out_dir, run_temp_dir)

    print('DONE!')


if __name__ == '__main__':
    _main()
