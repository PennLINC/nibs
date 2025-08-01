"""Process QSM data.

Steps:

0.  Load matlab/R2020B.
1.  Run SEPIA QSM estimation by calling the MATLAB script.

Notes:

- Remember to name the QSM files with the suffix "Chimap".
- Chimap outputs should be in parts per million (ppm).
"""
import os
import subprocess
from pprint import pprint

import nibabel as nb
from bids.layout import BIDSLayout, Query

from utils import get_filename


def collect_run_data(layout, bids_filters):
    queries = {
        # SWI images from raw BIDS dataset
        'megre_mag': {
            'datatype': 'anat',
            'acquisition': 'QSM',
            'part': 'mag',
            'echo': Query.ANY,
            'suffix': 'MEGRE',
            'extension': ['.nii', '.nii.gz'],
        },
        'megre_phase': {
            'datatype': 'anat',
            'acquisition': 'QSM',
            'part': 'phase',
            'echo': Query.ANY,
            'suffix': 'MEGRE',
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

    # Collect MATLAB-compatible NIfTIs for QSM
    matlab_mask_filename = os.path.join(temp_dir, 'python_mask.nii')
    matlab_mag_filename = os.path.join(temp_dir, 'python_mag.nii')
    matlab_phase_filename = os.path.join(temp_dir, 'python_phase.nii')

    # Run SEPIA QSM estimation
    sepia_dir = os.path.join(temp_dir, 'sepia')
    os.makedirs(sepia_dir, exist_ok=True)
    sepia_script = os.path.join(code_dir, 'processing', 'process_qsm_sepia.m')
    with open(sepia_script, 'r') as fobj:
        base_sepia_script = fobj.read()

    modified_sepia_script = (
        base_sepia_script.replace("{{ phase_file }}", matlab_phase_filename)
        .replace("{{ mag_file }}", matlab_mag_filename)
        .replace("{{ mask_file }}", matlab_mask_filename)
        .replace("{{ output_dir }}", os.path.join(sepia_dir, 'sepia'))
    )

    out_sepia_script = os.path.join(temp_dir, 'process_qsm_sepia.m')
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
    sepia_chimap_file = os.path.join(sepia_dir, 'sepia_Chimap.nii.gz')
    if not os.path.isfile(sepia_chimap_file):
        raise FileNotFoundError(f'SEPIA QSM output file {sepia_chimap_file} not found')

    sepia_chimap_img = nb.load(sepia_chimap_file)
    sepia_chimap_filename = get_filename(
        name_source=name_source,
        layout=layout,
        out_dir=out_dir,
        entities={'space': 'MEGRE', 'desc': 'SEPIA', 'suffix': 'Chimap'},
        dismiss_entities=['echo', 'part', 'inv', 'reconstruction'],
    )
    sepia_chimap_img.to_filename(sepia_chimap_filename)


if __name__ == '__main__':
    code_dir = '/cbica/projects/nibs/code'
    in_dir = '/cbica/projects/nibs/dset'
    smriprep_dir = '/cbica/projects/nibs/derivatives/smriprep'
    mese_dir = '/cbica/projects/nibs/derivatives/mese'
    out_dir = '/cbica/projects/nibs/derivatives/qsm'
    os.makedirs(out_dir, exist_ok=True)
    temp_dir = '/cbica/projects/nibs/work/qsm'
    os.makedirs(temp_dir, exist_ok=True)

    layout = BIDSLayout(
        in_dir,
        config=os.path.join(code_dir, 'nibs_bids_config.json'),
        validate=False,
        derivatives=[smriprep_dir, mese_dir, out_dir],
    )
    subjects = layout.get_subjects(suffix='MEGRE')
    for subject in subjects:
        print(f'Processing subject {subject}')
        sessions = layout.get_sessions(subject=subject, suffix='MEGRE')
        for session in sessions:
            print(f'Processing session {session}')
            megre_files = layout.get(
                subject=subject,
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
                run_temp_dir = os.path.join(temp_dir, os.path.basename(megre_file.path).split('.')[0])
                os.makedirs(run_temp_dir, exist_ok=True)
                process_run(layout, run_data, out_dir, run_temp_dir)

    print('DONE!')
