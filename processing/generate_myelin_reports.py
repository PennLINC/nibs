"""Generate myelin reports."""

import argparse
import json
import os

import numpy as np
import yaml
from bids.layout import BIDSLayout, Query
from nilearn import masking
from nireports.assembler.report import Report

from utils import get_filename, plot_scalar_map

CODE_DIR = '/cbica/projects/nibs/code'
QUERY_LOOKUP = {
    'Query.NONE': Query.NONE,
    'Query.ANY': Query.ANY,
}


def collect_run_data(layout, bids_filters):
    with open(os.path.join(CODE_DIR, 'processing', 'myelin_derivatives.yml'), 'r') as fobj:
        queries = yaml.safe_load(fobj)['queries']

    run_data = {}
    for key, query in queries['myelin'].items():
        for k, v in query.items():
            print(k, v)
            if isinstance(v, list):
                new_v = []
                for item in v:
                    new_v.append(QUERY_LOOKUP.get(item, item))
                query[k] = new_v
                print(k, query[k])
            else:
                query[k] = QUERY_LOOKUP.get(v, v)

            print(k, query[k])
            print()

        query = {**bids_filters, **query}
        files = layout.get(**query)
        if len(files) != 1:
            print(f'Expected 1 file for {key}, got {len(files)}: {query}')
            run_data[key] = None
            continue

        file = files[0]
        run_data[key] = file.path

    for key, query in queries['other'].items():
        query = {**bids_filters, **query}
        files = layout.get(**query)
        if len(files) != 1:
            raise ValueError(f'Expected 1 file for {key}, got {len(files)}: {query}')
        run_data[key] = files[0].path

    return run_data


def process_run(layout, run_data, out_dir):
    """Process a single MP2RAGE run.

    Parameters
    ----------
    layout : BIDSLayout
        BIDSLayout object.
    run_data : dict
        Dictionary of run data.
    out_dir : str
        Directory to write output files.
    """
    for name, file_ in run_data.items():
        if name in ['t1w', 'brainmask', 'dseg']:
            continue

        if file_ is None:
            print(f'No {name} file found for {run_data["t1w"]}')
            continue

        data = masking.apply_mask(file_, run_data['brainmask'])
        vmin = np.percentile(data, 2)
        vmin = np.minimum(vmin, 0)
        vmax = np.percentile(data, 98)

        scalar_report = get_filename(
            name_source=file_,
            layout=layout,
            out_dir=out_dir,
            entities={
                'datatype': 'figures',
                'desc': name,
                'suffix': 'scalar',
                'extension': '.svg',
            },
        )
        plot_scalar_map(
            underlay=run_data['t1w'],
            overlay=file_,
            mask=run_data['brainmask'],
            dseg=run_data['dseg'],
            out_file=scalar_report,
            vmin=vmin,
            vmax=vmax,
        )


def _get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--subject-id',
        type=lambda label: label.removeprefix('sub-'),
        required=True,
    )
    return parser


def _main(argv=None):
    """Run the process_mese workflow."""
    options = _get_parser().parse_args(argv)
    kwargs = vars(options)
    main(**kwargs)


def main(subject_id):
    in_dir = '/cbica/projects/nibs/dset'
    smriprep_dir = '/cbica/projects/nibs/derivatives/smriprep'
    ihmt_dir = '/cbica/projects/nibs/derivatives/ihmt'
    mp2rage_dir = '/cbica/projects/nibs/derivatives/pymp2rage'
    qsm_dir = '/cbica/projects/nibs/derivatives/qsm'
    t1wt2w_dir = '/cbica/projects/nibs/derivatives/t1wt2w_ratio'
    qsirecon_dki_dir = '/cbica/projects/nibs/derivatives/qsirecon/derivatives/qsirecon-DIPYDKI'
    qsirecon_dsi_dir = '/cbica/projects/nibs/derivatives/qsirecon/derivatives/qsirecon-DSIStudio'

    out_dir = '/cbica/projects/nibs/derivatives/myelin'
    os.makedirs(out_dir, exist_ok=True)

    bootstrap_file = os.path.join(CODE_DIR, 'processing', 'reports_spec_myelin.yml')
    assert os.path.isfile(bootstrap_file), f'Bootstrap file {bootstrap_file} not found'

    config = os.path.join(CODE_DIR, 'nibs_bids_config.json')
    layout = BIDSLayout(
        in_dir,
        config=[config],
        validate=False,
        derivatives=[
            smriprep_dir,
            ihmt_dir,
            mp2rage_dir,
            qsm_dir,
            t1wt2w_dir,
            qsirecon_dki_dir,
            qsirecon_dsi_dir,
        ],
    )

    print(f'Processing subject {subject_id}')
    sessions = layout.get_sessions(subject=subject_id)
    for session in sessions:
        print(f'Processing session {session}')
        try:
            run_data = collect_run_data(layout, {'subject': subject_id, 'session': session, 'space': 'MNI152NLin2009cAsym'})
        except ValueError as e:
            print(e)
            continue

        process_run(layout, run_data, out_dir)

        report_dir = os.path.join(out_dir, f'sub-{subject_id}', f'ses-{session}')
        robj = Report(
            report_dir,
            run_uuid=None,
            bootstrap_file=bootstrap_file,
            out_filename=f'sub-{subject_id}_ses-{session}.html',
            reportlets_dir=out_dir,
            plugins=None,
            plugin_meta=None,
            subject=subject_id,
            session=session,
        )
        robj.generate_report()

    # Write out dataset_description.json
    dataset_description_file = os.path.join(out_dir, 'dataset_description.json')
    if not os.path.isfile(dataset_description_file):
        dataset_description = {
            'Name': 'NIBS Myelin Derivatives',
            'BIDSVersion': '1.10.0',
            'DatasetType': 'derivative',
            'GeneratedBy': [
                {
                    'Name': 'Custom code',
                    'Description': 'Custom Python code combining ANTsPy and pymp2rage.',
                    'CodeURL': 'https://github.com/PennLINC/nibs',
                }
            ],
        }
        with open(dataset_description_file, 'w') as fobj:
            json.dump(dataset_description, fobj, sort_keys=True, indent=4)

    print('DONE!')


if __name__ == '__main__':
    _main()
