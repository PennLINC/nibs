"""Process QSM data with SEPIA and chi-separation.

Steps:

0.  Load matlab (e.g., ``module load matlab/R2023A``).
1.  For each echo set (E12345, E2345):
    a.  Concatenate the magnitude and phase MEGRE echoes and build the SEPIA header.
    b.  Run SEPIA QSM estimation once on the concatenated data.
    c.  Run chi-separation once for each parameter combination, using the
        processed magnitude and phase images from SEPIA.

Notes:

- Must be run after process_qsm_prep.py (brain mask and R2*/R2' maps).
- Requires SEPIA and the chi-sep MATLAB toolbox along with their dependencies.
- Chimap outputs are in parts per million (ppm).
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

from utils import get_filename, load_config, run_command

CFG = load_config()
CODE_DIR = CFG['code_dir']

# Each echo set selects which MEGRE echoes feed SEPIA and chi-separation:
# E12345 uses all five echoes, E2345 drops the first echo.
ECHO_SETS = ['E12345', 'E2345']

# Chi-separation parameter combinations run within each echo set.
# (label, is_scaling, have_r2prime) -- the R2*/R2' paths are filled in per echo
# set because only the R2' variant reads precomputed maps.
CHISEP_COMBOS = [
    ('chisep+r2p', 0, 1),
    ('chisep+r2primenet', 0, 0),
    ('chisep+r2s', 1, 0),
]


def collect_run_data(layout: object, bids_filters: dict) -> dict[str, str]:
    """Collect inputs for SEPIA and chi-separation processing.

    Parameters
    ----------
    layout : bids.BIDSLayout
        BIDSLayout indexing the dataset and derivatives.
    bids_filters : dict
        BIDS entity filters (e.g., subject, session, run) to narrow the query.

    Returns
    -------
    run_data : dict
        Mapping of descriptive keys to resolved file paths. ``megre_mag`` and
        ``megre_phase`` map to sorted lists of five echo paths; all other keys
        map to a single path.
    """
    queries = {
        # Multi-echo GRE magnitude/phase from the raw BIDS dataset (for SEPIA).
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
        # Brain mask in MEGRE space (from process_qsm_prep.py).
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
        # R2*/R2' maps per echo set (for the chi-separation R2' variant).
        'r2s_e12345': {
            'datatype': 'anat',
            'space': 'MEGRE',
            'desc': 'MEGRE+E12345',
            'suffix': 'R2starmap',
            'extension': ['.nii', '.nii.gz'],
        },
        'r2p_e12345': {
            'datatype': 'anat',
            'space': 'MEGRE',
            'desc': 'MEGRE+E12345',
            'suffix': 'R2primemap',
            'extension': ['.nii', '.nii.gz'],
        },
        'r2s_e2345': {
            'datatype': 'anat',
            'space': 'MEGRE',
            'desc': 'MEGRE+E2345',
            'suffix': 'R2starmap',
            'extension': ['.nii', '.nii.gz'],
        },
        'r2p_e2345': {
            'datatype': 'anat',
            'space': 'MEGRE',
            'desc': 'MEGRE+E2345',
            'suffix': 'R2primemap',
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
            run_data[key] = sorted([f.path for f in files])
            continue

        if len(files) != 1:
            raise ValueError(f'Expected 1 file for {key}, got {len(files)} with query {query}')

        run_data[key] = files[0].path

    if len(run_data['megre_mag']) != len(run_data['megre_phase']):
        raise ValueError('Expected same number of magnitude and phase images')

    print(f'Collected run data:\n{pformat(run_data, indent=4)}', flush=True)
    return run_data


def run_sepia(
    subject_id: str,
    session: str,
    version: str,
    sepia_work_dir: str,
    mag_file: str,
    phase_file: str,
    header_file: str,
    mask_file: str,
) -> str:
    """Run SEPIA QSM estimation on one echo set and return the Chimap path.

    Parameters
    ----------
    subject_id, session, version : str
        BIDS subject/session labels and echo-set label (e.g., ``E12345``).
    sepia_work_dir : str
        Directory holding the concatenated inputs and receiving SEPIA outputs.
    mag_file, phase_file, header_file, mask_file : str
        Concatenated magnitude/phase NIfTIs, SEPIA header, and brain mask.

    Returns
    -------
    sepia_chimap_file : str
        Path to the SEPIA Chimap output.
    """
    out_prefix = os.path.join(sepia_work_dir, f'sub-{subject_id}_ses-{session}_desc-{version}_')

    sepia_script = os.path.join(CODE_DIR, 'processing', 'process_qsm_sepia.m')
    with open(sepia_script) as fobj:
        base_sepia_script = fobj.read()

    modified_sepia_script = (
        base_sepia_script.replace('{{ phase_file }}', phase_file)
        .replace('{{ mag_file }}', mag_file)
        .replace('{{ output_dir }}', out_prefix)
        .replace('{{ header_file }}', header_file)
        .replace('{{ mask_file }}', mask_file)
    )

    out_sepia_script = os.path.join(sepia_work_dir, f'process_qsm_sepia_{version}.m')
    with open(out_sepia_script, 'w') as fobj:
        fobj.write(modified_sepia_script)

    result = subprocess.run(
        [
            'matlab',
            '-nodisplay',
            '-nosplash',
            '-nodesktop',
            '-r',
            f"run('{out_sepia_script}'); exit;",
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f'MATLAB exited with code {result.returncode}\n'
            f'stdout:\n{result.stdout}\n'
            f'stderr:\n{result.stderr}'
        )

    sepia_chimap_file = f'{out_prefix}Chimap.nii.gz'
    if not os.path.isfile(sepia_chimap_file):
        raise FileNotFoundError(f'SEPIA QSM output file {sepia_chimap_file} not found')

    return sepia_chimap_file


def run_chisep(
    run_data: dict,
    version: str,
    example_nifti: str,
    sepia_work_dir: str,
    subject_id: str,
    session: str,
) -> None:
    """Run chi-separation for every parameter combination of one echo set.

    Parameters
    ----------
    run_data : dict
        Input file paths from :func:`collect_run_data`.
    version : str
        Echo-set label (e.g., ``E12345``); selects the matching R2*/R2' maps.
    example_nifti : str
        Raw MEGRE echo-1 magnitude NIfTI, used for the output NIfTI header.
    sepia_work_dir : str
        SEPIA working directory holding the concatenated MEGRE data and header
        that chi-separation reads as input.
    subject_id, session : str
        BIDS subject/session labels.
    """
    work_dir = CFG['work_dir']
    matlab_script_dir = os.path.join(CODE_DIR, 'processing')

    version_key = version.lower()
    r2s_path = run_data[f'r2s_{version_key}']
    r2p_path = run_data[f'r2p_{version_key}']

    for label, is_scaling, have_r2prime in CHISEP_COMBOS:
        r2s = r2s_path if have_r2prime else ''
        r2p = r2p_path if have_r2prime else ''

        out_dir = os.path.join(
            work_dir, f'qsm-{version}+{label}', f'sub-{subject_id}', f'ses-{session}', 'anat'
        )
        os.makedirs(out_dir, exist_ok=True)

        cmd = [
            'matlab',
            '-nodisplay',
            '-nosplash',
            '-r',
            (
                'try; '
                f"addpath(genpath('{matlab_script_dir}')); "
                f"process_qsm_chisep('{example_nifti}','{sepia_work_dir}','{out_dir}',"
                f"{is_scaling},{have_r2prime},'{r2s}','{r2p}'); "
                'exit(0); '
                "catch ME; disp(getReport(ME, 'extended', 'hyperlinks', 'off')); exit(1); end;"
            ),
        ]
        print(f'Running chi-sep: {version} {label}', flush=True)
        run_command(cmd)
        print(f'Finished chi-sep: {version} {label}', flush=True)


def process_run(layout, run_data, out_dir, subject_id, session):
    """Run SEPIA and chi-separation for each echo set of a single run.

    Parameters
    ----------
    layout : BIDSLayout
        BIDSLayout object for the dataset.
    run_data : dict
        Dictionary of input file paths from :func:`collect_run_data`.
    out_dir : str
        Path to the QSM derivatives directory (for the copied Chimap).
    subject_id : str
        BIDS subject label (without 'sub-' prefix).
    session : str
        BIDS session label (without 'ses-' prefix).
    """
    name_source = run_data['megre_mag'][0]
    example_nifti = run_data['megre_mag'][0]

    header_file = os.path.join(CODE_DIR, 'processing', 'sepia_header.mat')
    header_struct = loadmat(header_file)
    header_struct['B0_dir'] = header_struct['B0_dir'].astype(float)
    header_struct['B0'] = header_struct['B0'].astype(float)

    for version in ECHO_SETS:
        if version == 'E12345':
            mag_imgs = run_data['megre_mag']
            phase_imgs = run_data['megre_phase']
        else:
            # Drop the first echo for both the images and the header TE vector.
            mag_imgs = run_data['megre_mag'][1:]
            phase_imgs = run_data['megre_phase'][1:]
            header_struct['TE'] = header_struct['TE'][:, 1:]

        sepia_work_dir = os.path.join(
            CFG['work_dir'], f'qsm-{version}+sepia', f'sub-{subject_id}', f'ses-{session}', 'anat'
        )
        os.makedirs(sepia_work_dir, exist_ok=True)

        # Write the concatenated MEGRE inputs and header with the names that both
        # SEPIA and chi-separation read from the SEPIA working directory.
        prefix = os.path.join(sepia_work_dir, f'sub-{subject_id}_ses-{session}_')
        mag_concat_file = f'{prefix}part-mag_desc-concat_MEGRE.nii.gz'
        phase_concat_file = f'{prefix}part-phase_desc-concat_MEGRE.nii.gz'
        header_concat_file = f'{prefix}header.mat'
        image.concat_imgs(mag_imgs).to_filename(mag_concat_file)
        image.concat_imgs(phase_imgs).to_filename(phase_concat_file)
        savemat(header_concat_file, header_struct)

        # Run SEPIA once for this echo set.
        sepia_chimap_file = run_sepia(
            subject_id=subject_id,
            session=session,
            version=version,
            sepia_work_dir=sepia_work_dir,
            mag_file=mag_concat_file,
            phase_file=phase_concat_file,
            header_file=header_concat_file,
            mask_file=run_data['mask'],
        )

        # Copy the SEPIA Chimap into the QSM derivatives.
        sepia_chimap_filename = get_filename(
            name_source=name_source,
            layout=layout,
            out_dir=out_dir,
            entities={'space': 'MEGRE', 'desc': f'{version}+sepia', 'suffix': 'Chimap'},
            dismiss_entities=['acquisition', 'echo', 'part', 'inv', 'reconstruction'],
        )
        nb.load(sepia_chimap_file).to_filename(sepia_chimap_filename)

        # Run chi-separation for every parameter combination, using the SEPIA
        # working directory (concatenated MEGRE data + header) as input.
        run_chisep(
            run_data=run_data,
            version=version,
            example_nifti=example_nifti,
            sepia_work_dir=sepia_work_dir,
            subject_id=subject_id,
            session=session,
        )


def _get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--subject-id',
        type=lambda label: label.removeprefix('sub-'),
        required=True,
    )
    return parser


def _main(argv=None):
    """Run the process_qsm workflow."""
    options = _get_parser().parse_args(argv)
    kwargs = vars(options)
    main(**kwargs)


def main(subject_id):
    in_dir = CFG['bids_dir']
    smriprep_dir = CFG['derivatives']['smriprep']
    mese_dir = CFG['derivatives']['mese']
    out_dir = CFG['derivatives']['qsm']
    os.makedirs(out_dir, exist_ok=True)

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
        if not megre_files:
            print(f'No MEGRE files found for subject {subject_id} and session {session}')
            continue

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

            process_run(layout, run_data, out_dir, subject_id, session)

    print('DONE!', flush=True)


if __name__ == '__main__':
    _main()
