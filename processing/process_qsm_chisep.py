"""Process QSM data.

Steps:

0.  Load matlab/R2023B.
1.  Run chi-separation QSM estimation by calling the MATLAB script.

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
    matlab_r2s_filename = os.path.join(temp_dir, 'python_r2s.nii')
    matlab_r2p_filename = os.path.join(temp_dir, 'python_r2p.nii')

    # Now run the chi-separation QSM estimation with R2' map
    chisep_r2p_dir = os.path.join(temp_dir, 'chisep_r2p', 'chisep_output')
    os.makedirs(chisep_r2p_dir, exist_ok=True)
    chisep_script = os.path.join(CODE_DIR, 'processing', 'process_qsm_chisep.m')
    with open(chisep_script, 'r') as fobj:
        base_chisep_script = fobj.read()

    modified_chisep_script = (
        base_chisep_script.replace("{{ mag_file }}", matlab_mag_filename)
        .replace("{{ phase_file }}", matlab_phase_filename)
        .replace("{{ mask_file }}", matlab_mask_filename)
        .replace("{{ r2s_file }}", matlab_r2s_filename)
        .replace("{{ r2p_file }}", matlab_r2p_filename)
        .replace("{{ output_dir }}", os.path.join(temp_dir, 'chisep_r2p'))
    )

    out_chisep_script = os.path.join(temp_dir, 'process_qsm_chisep_r2p.m')
    with open(out_chisep_script, "w") as fobj:
        fobj.write(modified_chisep_script)

    subprocess.run(
        [
            "matlab",
            "-nodisplay",
            "-nosplash",
            "-nodesktop",
            "-r",
            f"run('{out_chisep_script}'); exit;",
        ],
    )
    chisep_r2p_iron_file = os.path.join(chisep_r2p_dir, 'iron.nii.gz')
    if not os.path.isfile(chisep_r2p_iron_file):
        raise FileNotFoundError(f'chi-separation QSM output file {chisep_r2p_iron_file} not found')

    chisep_r2p_iron_img = nb.load(chisep_r2p_iron_file)
    chisep_r2p_iron_filename = get_filename(
        name_source=name_source,
        layout=layout,
        out_dir=out_dir,
        entities={'space': 'MEGRE', 'desc': 'chisepR2p', 'suffix': 'ironw'},
        dismiss_entities=['echo', 'part', 'inv', 'reconstruction'],
    )
    chisep_r2p_iron_img.to_filename(chisep_r2p_iron_filename)

    chisep_r2p_myelin_file = os.path.join(chisep_r2p_dir, 'myelin.nii.gz')
    if not os.path.isfile(chisep_r2p_myelin_file):
        raise FileNotFoundError(f'chi-separation QSM output file {chisep_r2p_myelin_file} not found')

    chisep_r2p_myelin_img = nb.load(chisep_r2p_myelin_file)
    chisep_r2p_myelin_filename = get_filename(
        name_source=name_source,
        layout=layout,
        out_dir=out_dir,
        entities={'space': 'MEGRE', 'desc': 'chisepR2p', 'suffix': 'myelinw'},
        dismiss_entities=['echo', 'part', 'inv', 'reconstruction'],
    )
    chisep_r2p_myelin_img.to_filename(chisep_r2p_myelin_filename)

    # Run X-separation QSM estimation without R2' map
    chisep_no_r_dir = os.path.join(temp_dir, 'chisep_no_r2p', 'chisep_output')
    os.makedirs(chisep_no_r_dir, exist_ok=True)
    chisep_script = os.path.join(CODE_DIR, 'processing', 'process_qsm_chisep.m')
    with open(chisep_script, 'r') as fobj:
        base_chisep_script = fobj.read()

    modified_chisep_script = (
        base_chisep_script.replace("{{ mag_file }}", matlab_mag_filename)
        .replace("{{ phase_file }}", matlab_phase_filename)
        .replace("{{ mask_file }}", matlab_mask_filename)
        .replace("{{ r2s_file }}", matlab_r2s_filename)
        .replace("{{ r2p_file }}", 'None')
        .replace("{{ output_dir }}", os.path.join(temp_dir, 'chisep_no_r2p'))
    )

    out_chisep_script = os.path.join(temp_dir, 'process_qsm_chisep_no_r2p.m')
    with open(out_chisep_script, "w") as fobj:
        fobj.write(modified_chisep_script)

    subprocess.run(
        [
            "matlab",
            "-nodisplay",
            "-nosplash",
            "-nodesktop",
            "-r",
            f"run('{out_chisep_script}'); exit;",
        ],
    )

    chisep_no_r2p_iron_file = os.path.join(chisep_no_r_dir, 'iron.nii.gz')
    if not os.path.isfile(chisep_no_r2p_iron_file):
        raise FileNotFoundError(f'chi-separation QSM output file {chisep_no_r2p_iron_file} not found')

    chisep_no_r2p_iron_img = nb.load(chisep_no_r2p_iron_file)
    chisep_no_r2p_iron_filename = get_filename(
        name_source=name_source,
        layout=layout,
        out_dir=out_dir,
        entities={'space': 'MEGRE', 'desc': 'chisep', 'suffix': 'ironw'},
        dismiss_entities=['echo', 'part', 'inv', 'reconstruction'],
    )
    chisep_no_r2p_iron_img.to_filename(chisep_no_r2p_iron_filename)

    chisep_no_r2p_myelin_file = os.path.join(chisep_no_r_dir, 'myelin.nii.gz')
    if not os.path.isfile(chisep_no_r2p_myelin_file):
        raise FileNotFoundError(f'chi-separation QSM output file {chisep_no_r2p_myelin_file} not found')

    chisep_no_r2p_myelin_img = nb.load(chisep_no_r2p_myelin_file)
    chisep_no_r2p_myelin_filename = get_filename(
        name_source=name_source,
        layout=layout,
        out_dir=out_dir,
        entities={'space': 'MEGRE', 'desc': 'chisep', 'suffix': 'myelinw'},
        dismiss_entities=['echo', 'part', 'inv', 'reconstruction'],
    )
    chisep_no_r2p_myelin_img.to_filename(chisep_no_r2p_myelin_filename)


def _get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--subject-id',
        type=lambda label: label.removeprefix('sub-'),
        required=True,
    )
    return parser


def _main(argv=None):
    """Run the process_qsm_chisep workflow."""
    options = _get_parser().parse_args(argv)
    kwargs = vars(options)
    main(**kwargs)


def main(subject_id):
    in_dir = '/cbica/projects/nibs/dset'
    smriprep_dir = '/cbica/projects/nibs/derivatives/smriprep'
    mese_dir = '/cbica/projects/nibs/derivatives/mese'
    out_dir = '/cbica/projects/nibs/derivatives/qsm'
    os.makedirs(out_dir, exist_ok=True)
    temp_dir = '/cbica/projects/nibs/work/qsm'
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
            run_temp_dir = os.path.join(temp_dir, os.path.basename(megre_file.path).split('.')[0])
            os.makedirs(run_temp_dir, exist_ok=True)
            process_run(layout, run_data, out_dir, run_temp_dir)

    print('DONE!')


if __name__ == '__main__':
    _main()
