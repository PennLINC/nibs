"""Process QSM data using chi-separation.

Steps:

1.  Run chi-separation on SEPIA-preprocessed QSM data for six parameter combinations.

Notes:

- Must be run after process_qsm_sepia.py.
- Requires the chi-sep MATLAB toolbox and its dependencies.
"""

from __future__ import annotations

import argparse
import os
from pprint import pformat

from bids.layout import BIDSLayout, Query

from utils import run_command

PROJECT_ROOT = '/home/tsalo/nibs'
CFG = {
    'project_root': PROJECT_ROOT,
    'bids_dir': os.path.join(PROJECT_ROOT, 'dset'),
    'code_dir': os.path.join(PROJECT_ROOT, 'code', 'nibs'),
    'work_dir': os.path.join(PROJECT_ROOT, 'work'),
    'derivatives': {
        'qsm': os.path.join(PROJECT_ROOT, 'derivatives', 'qsm'),
    },
}
CODE_DIR = CFG['code_dir']


def collect_run_data(layout: object, bids_filters: dict) -> dict[str, str]:
    """Collect input files for chi-separation processing.

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
        'megre_echo1_mag': {
            'datatype': 'anat',
            'acquisition': 'QSM',
            'part': 'mag',
            'echo': 1,
            'space': Query.NONE,
            'desc': Query.NONE,
            'suffix': 'MEGRE',
            'extension': ['.nii', '.nii.gz'],
        },
        'r2prime_e12345': {
            'datatype': 'anat',
            'space': 'MEGRE',
            'desc': 'MEGRE+E12345',
            'suffix': 'R2primemap',
            'extension': ['.nii', '.nii.gz'],
        },
        'r2prime_e2345': {
            'datatype': 'anat',
            'space': 'MEGRE',
            'desc': 'MEGRE+E2345',
            'suffix': 'R2primemap',
            'extension': ['.nii', '.nii.gz'],
        },
        'r2star_e12345': {
            'datatype': 'anat',
            'space': 'MEGRE',
            'desc': 'MEGRE+E12345',
            'suffix': 'R2starmap',
            'extension': ['.nii', '.nii.gz'],
        },
        'r2star_e2345': {
            'datatype': 'anat',
            'space': 'MEGRE',
            'desc': 'MEGRE+E2345',
            'suffix': 'R2starmap',
            'extension': ['.nii', '.nii.gz'],
        },
    }

    run_data = {}
    for key, query in queries.items():
        query = {**bids_filters, **query}
        files = layout.get(**query)
        if len(files) != 1:
            raise ValueError(f'Expected 1 file for {key}, got {len(files)} with query {query}')
        run_data[key] = files[0].path

    print(f'Collected run data:\n{pformat(run_data, indent=4)}', flush=True)
    return run_data


def process_run(run_data: dict, subject_id: str, session: str) -> None:
    """Run chi-separation for all six parameter combinations.

    Parameters
    ----------
    run_data : dict
        Dictionary of input file paths from collect_run_data.
    subject_id : str
        BIDS subject label (without 'sub-' prefix).
    session : str
        BIDS session label (without 'ses-' prefix).
    """
    work_dir = CFG['work_dir']
    input_file = run_data['megre_echo1_mag']

    sepia_e12345 = os.path.join(
        work_dir, 'qsm-E12345+sepia', f'sub-{subject_id}', f'ses-{session}', 'anat'
    )
    sepia_e2345 = os.path.join(
        work_dir, 'qsm-E2345+sepia', f'sub-{subject_id}', f'ses-{session}', 'anat'
    )

    def out_dir(variant):
        return os.path.join(
            work_dir, f'qsm-{variant}', f'sub-{subject_id}', f'ses-{session}', 'anat'
        )

    combos = [
        # SEPIA already writes echo-selected concat inputs for each variant, so
        # chi-sep should read each concat file from its first stored volume.
        # (label, sepia_folder, r2p_path, outputa, echo_start, have_r2prime, is_scaling, r2s_path)
        (
            'E12345+chisep+r2p',
            sepia_e12345,
            run_data['r2prime_e12345'],
            out_dir('E12345+chisep+r2p'),
            1,
            1,
            0,
            run_data['r2star_e12345'],
        ),
        (
            'E2345+chisep+r2p',
            sepia_e2345,
            run_data['r2prime_e2345'],
            out_dir('E2345+chisep+r2p'),
            1,
            1,
            0,
            run_data['r2star_e2345'],
        ),
        (
            'E12345+chisep+r2primenet',
            sepia_e12345,
            '',
            out_dir('E12345+chisep+r2primenet'),
            1,
            0,
            0,
            '',
        ),
        (
            'E2345+chisep+r2primenet',
            sepia_e2345,
            '',
            out_dir('E2345+chisep+r2primenet'),
            1,
            0,
            0,
            '',
        ),
        (
            'E12345+chisep+r2s',
            sepia_e12345,
            '',
            out_dir('E12345+chisep+r2s'),
            1,
            0,
            1,
            '',
        ),
        (
            'E2345+chisep+r2s',
            sepia_e2345,
            '',
            out_dir('E2345+chisep+r2s'),
            1,
            0,
            1,
            '',
        ),
    ]
    matlab_script_dir = os.path.join(CODE_DIR, 'processing')
    for (
        label,
        sepia_folder,
        r2p_path,
        outputa,
        echo_start,
        have_r2prime,
        is_scaling,
        r2s_path,
    ) in combos:
        cmd = [
            'matlab',
            '-nodisplay',
            '-nosplash',
            '-r',
            (
                'try; '
                f"addpath(genpath('{matlab_script_dir}')); "
                f"process_qsm_chisep('{input_file}','{sepia_folder}','{r2p_path}','{outputa}'"
                f",{echo_start},{have_r2prime},{is_scaling},'{r2s_path}'); "
                'exit(0); '
                "catch ME; disp(getReport(ME, 'extended', 'hyperlinks', 'off')); exit(1); end;"
            ),
        ]
        print(f'Running chi-sep: {label}', flush=True)
        run_command(cmd)
        print(f'Finished chi-sep: {label}', flush=True)


def _get_parser() -> argparse.ArgumentParser:
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
    in_dir = CFG['bids_dir']
    out_dir = CFG['derivatives']['qsm']
    os.makedirs(out_dir, exist_ok=True)

    layout = BIDSLayout(
        in_dir,
        config=os.path.join(CODE_DIR, 'configuration', 'nibs_bids_config.json'),
        validate=False,
        derivatives=[out_dir],
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
                print(f'Failed {megre_file}')
                print(e)
                continue

            process_run(run_data, subject_id, session)

    print('DONE!', flush=True)


if __name__ == '__main__':
    _main()
