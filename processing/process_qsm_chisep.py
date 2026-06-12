"""Process QSM data using chi-separation.

Steps:

1.  Run chi-separation on SEPIA-preprocessed QSM data for six parameter combinations.

Notes:

- Must be run after process_qsm_sepia.py and process_qsm_prep.py (brain mask).
- Requires the chi-sep MATLAB toolbox and its dependencies.
"""

from __future__ import annotations

import argparse
import os
from pprint import pformat

from bids.layout import BIDSLayout, Query

from utils import load_config, run_command

CFG = load_config()
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
        'r2p_e12345': {
            'datatype': 'anat',
            'space': 'MEGRE',
            'desc': 'MEGRE+E12345',
            'suffix': 'R2primemap',
            'extension': ['.nii', '.nii.gz'],
        },
        'r2p_e2345': {
            'datatype': 'anat',
            'space': 'MEGRE',
            'desc': 'MEGRE+E2345',
            'suffix': 'R2primemap',
            'extension': ['.nii', '.nii.gz'],
        },
        'r2s_e12345': {
            'datatype': 'anat',
            'space': 'MEGRE',
            'desc': 'MEGRE+E12345',
            'suffix': 'R2starmap',
            'extension': ['.nii', '.nii.gz'],
        },
        'r2s_e2345': {
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
    example_nifti = run_data['megre_echo1_mag']

    for version in ['E12345', 'E2345']:
        sepia_work_dir = os.path.join(
            work_dir, f'qsm-{version}+sepia', f'sub-{subject_id}', f'ses-{session}', 'anat'
        )

        def get_out_dir(variant):
            return os.path.join(
                work_dir, f'qsm-{variant}', f'sub-{subject_id}', f'ses-{session}', 'anat'
            )

        combos = [
            # SEPIA already writes echo-selected concat inputs for each variant, so
            # chi-sep reads every stored volume of each concat file.
            # (label, is_scaling, have_r2prime, r2s_path, r2p_path)
            ('chisep+r2p', 0, 1, run_data['r2s_e12345'], run_data['r2p_e12345']),
            ('chisep+r2primenet', 0, 0, '', ''),
            ('chisep+r2s', 1, 0, '', ''),
        ]
        matlab_script_dir = os.path.join(CODE_DIR, 'processing')
        for (label, is_scaling, have_r2prime, r2s_path, r2p_path) in combos:
            out_dir = get_out_dir(f'{version}+{label}')
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
                    f"{is_scaling},{have_r2prime},'{r2s_path}','{r2p_path}'); "
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
