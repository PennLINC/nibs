"""Process QSM data.

Steps:

1.  Warp QSM derivatives to T1w and MNI152NLin2009cAsym spaces.
2.  Generate scalar reports for QSM derivatives.

Notes:

- The R2* map is calculated using the monoexponential fit.
- This must be run after sMRIPrep and process_mese.py.
"""

import argparse
import json
import os
from pprint import pprint

import ants
import numpy as np
from bids.layout import BIDSLayout, Query
from nilearn import masking
from nireports.assembler.report import Report

from utils import (
    get_filename,
    plot_coregistration,
    plot_scalar_map,
)

CODE_DIR = '/cbica/projects/nibs/code'


def collect_run_data(layout, bids_filters):
    queries = {
        # SEPIA Chimap
        'sepia_chimap_e12345': {
            'datatype': 'anat',
            'run': [Query.NONE, Query.ANY],
            'space': 'MEGRE',
            'res': Query.NONE,
            'desc': 'E12345+sepia',
            'suffix': 'Chimap',
            'extension': ['.nii', '.nii.gz'],
        },
        'sepia_chimap_e2345': {
            'datatype': 'anat',
            'run': [Query.NONE, Query.ANY],
            'space': 'MEGRE',
            'res': Query.NONE,
            'desc': 'E2345+sepia',
            'suffix': 'Chimap',
            'extension': ['.nii', '.nii.gz'],
        },
        # Chisep chi map with R2'
        'chisep_chimap_r2p_e12345': {
            'datatype': 'anat',
            'run': [Query.NONE, Query.ANY],
            'space': 'MEGRE',
            'res': Query.NONE,
            'desc': 'E12345+chisep+r2p',
            'suffix': 'Chimap',
            'extension': ['.nii', '.nii.gz'],
        },
        'chisep_chimap_r2p_e2345': {
            'datatype': 'anat',
            'run': [Query.NONE, Query.ANY],
            'space': 'MEGRE',
            'res': Query.NONE,
            'desc': 'E2345+chisep+r2p',
            'suffix': 'Chimap',
            'extension': ['.nii', '.nii.gz'],
        },
        # Chisep iron map with R2'
        'chisep_iron_r2p_e12345': {
            'datatype': 'anat',
            'run': [Query.NONE, Query.ANY],
            'space': 'MEGRE',
            'res': Query.NONE,
            'desc': 'E12345+chisep+r2p',
            'suffix': 'ironw',
            'extension': ['.nii', '.nii.gz'],
        },
        'chisep_iron_r2p_e2345': {
            'datatype': 'anat',
            'run': [Query.NONE, Query.ANY],
            'space': 'MEGRE',
            'res': Query.NONE,
            'desc': 'E2345+chisep+r2p',
            'suffix': 'ironw',
            'extension': ['.nii', '.nii.gz'],
        },
        # Chisep myelin map with R2'
        'chisep_myelin_r2p_e12345': {
            'datatype': 'anat',
            'run': [Query.NONE, Query.ANY],
            'space': 'MEGRE',
            'res': Query.NONE,
            'desc': 'E12345+chisep+r2p',
            'suffix': 'myelinw',
            'extension': ['.nii', '.nii.gz'],
        },
        'chisep_myelin_r2p_e2345': {
            'datatype': 'anat',
            'run': [Query.NONE, Query.ANY],
            'space': 'MEGRE',
            'res': Query.NONE,
            'desc': 'E2345+chisep+r2p',
            'suffix': 'myelinw',
            'extension': ['.nii', '.nii.gz'],
        },
        # Chisep chi map with R2'
        'chisep_chimap_r2primenet_e12345': {
            'datatype': 'anat',
            'run': [Query.NONE, Query.ANY],
            'space': 'MEGRE',
            'res': Query.NONE,
            'desc': 'E12345+chisep+r2primenet',
            'suffix': 'Chimap',
            'extension': ['.nii', '.nii.gz'],
        },
        'chisep_chimap_r2primenet_e2345': {
            'datatype': 'anat',
            'run': [Query.NONE, Query.ANY],
            'space': 'MEGRE',
            'res': Query.NONE,
            'desc': 'E2345+chisep+r2primenet',
            'suffix': 'Chimap',
            'extension': ['.nii', '.nii.gz'],
        },
        # Chisep iron map with R2'
        'chisep_iron_r2primenet_e12345': {
            'datatype': 'anat',
            'run': [Query.NONE, Query.ANY],
            'space': 'MEGRE',
            'res': Query.NONE,
            'desc': 'E12345+chisep+r2primenet',
            'suffix': 'ironw',
            'extension': ['.nii', '.nii.gz'],
        },
        'chisep_iron_r2primenet_e2345': {
            'datatype': 'anat',
            'run': [Query.NONE, Query.ANY],
            'space': 'MEGRE',
            'res': Query.NONE,
            'desc': 'E2345+chisep+r2primenet',
            'suffix': 'ironw',
            'extension': ['.nii', '.nii.gz'],
        },
        # Chisep myelin map with R2'
        'chisep_myelin_r2primenet_e12345': {
            'datatype': 'anat',
            'run': [Query.NONE, Query.ANY],
            'space': 'MEGRE',
            'res': Query.NONE,
            'desc': 'E12345+chisep+r2primenet',
            'suffix': 'myelinw',
            'extension': ['.nii', '.nii.gz'],
        },
        'chisep_myelin_r2primenet_e2345': {
            'datatype': 'anat',
            'run': [Query.NONE, Query.ANY],
            'space': 'MEGRE',
            'res': Query.NONE,
            'desc': 'E2345+chisep+r2primenet',
            'suffix': 'myelinw',
            'extension': ['.nii', '.nii.gz'],
        },
        # Chisep chi map with R2*
        'chisep_chimap_r2s_e12345': {
            'datatype': 'anat',
            'run': [Query.NONE, Query.ANY],
            'space': 'MEGRE',
            'res': Query.NONE,
            'desc': 'E12345+chisep+r2s',
            'suffix': 'Chimap',
            'extension': ['.nii', '.nii.gz'],
        },
        'chisep_chimap_r2s_e2345': {
            'datatype': 'anat',
            'run': [Query.NONE, Query.ANY],
            'space': 'MEGRE',
            'res': Query.NONE,
            'desc': 'E2345+chisep+r2s',
            'suffix': 'Chimap',
            'extension': ['.nii', '.nii.gz'],
        },
        # Chisep iron map with R2*
        'chisep_iron_r2s_e12345': {
            'datatype': 'anat',
            'run': [Query.NONE, Query.ANY],
            'space': 'MEGRE',
            'res': Query.NONE,
            'desc': 'E12345+chisep+r2s',
            'suffix': 'ironw',
            'extension': ['.nii', '.nii.gz'],
        },
        'chisep_iron_r2s_e2345': {
            'datatype': 'anat',
            'run': [Query.NONE, Query.ANY],
            'space': 'MEGRE',
            'res': Query.NONE,
            'desc': 'E2345+chisep+r2s',
            'suffix': 'ironw',
            'extension': ['.nii', '.nii.gz'],
        },
        # Chisep myelin map with R2*
        'chisep_myelin_r2s_e12345': {
            'datatype': 'anat',
            'run': [Query.NONE, Query.ANY],
            'space': 'MEGRE',
            'res': Query.NONE,
            'desc': 'E12345+chisep+r2s',
            'suffix': 'myelinw',
            'extension': ['.nii', '.nii.gz'],
        },
        'chisep_myelin_r2s_e2345': {
            'datatype': 'anat',
            'run': [Query.NONE, Query.ANY],
            'space': 'MEGRE',
            'res': Query.NONE,
            'desc': 'E2345+chisep+r2s',
            'suffix': 'myelinw',
            'extension': ['.nii', '.nii.gz'],
        },
        # Coregistration transform from process_qsm_prep.py
        'megre2t1w_xfm': {
            'datatype': 'anat',
            'run': [Query.NONE, Query.ANY],
            'from': 'MEGRE',
            'to': 'T1w',
            'mode': 'image',
            'suffix': 'xfm',
            'extension': '.mat',
        },
        # T1w-space T1w image from sMRIPrep
        't1w': {
            'datatype': 'anat',
            'session': [Query.NONE, Query.ANY],
            'run': [Query.NONE, Query.ANY],
            'space': Query.NONE,
            'desc': 'preproc',
            'suffix': 'T1w',
            'extension': ['.nii', '.nii.gz'],
        },
        # MNI-space T1w image from sMRIPrep
        't1w_mni': {
            'datatype': 'anat',
            'session': [Query.NONE, Query.ANY],
            'run': [Query.NONE, Query.ANY],
            'space': 'MNI152NLin2009cAsym',
            'desc': 'preproc',
            'suffix': 'T1w',
            'extension': ['.nii', '.nii.gz'],
        },
        # Normalization transform from sMRIPrep
        't1w2mni_xfm': {
            'datatype': 'anat',
            'session': [Query.NONE, Query.ANY],
            'run': [Query.NONE, Query.ANY],
            'from': 'T1w',
            'to': 'MNI152NLin2009cAsym',
            'mode': 'image',
            'suffix': 'xfm',
            'extension': '.h5',
        },
        # MNI-space dseg from sMRIPrep
        'dseg_mni': {
            'datatype': 'anat',
            'session': [Query.NONE, Query.ANY],
            'run': [Query.NONE, Query.ANY],
            'space': 'MNI152NLin2009cAsym',
            'suffix': 'dseg',
            'extension': ['.nii', '.nii.gz'],
        },
        # sMRIPrep MNI-space brain mask
        'mni_mask': {
            'datatype': 'anat',
            'session': [Query.NONE, Query.ANY],
            'run': [Query.NONE, Query.ANY],
            'space': 'MNI152NLin2009cAsym',
            'desc': 'brain',
            'suffix': 'mask',
            'extension': ['.nii', '.nii.gz'],
        },
    }

    run_data = {}
    for key, query in queries.items():
        query = {**bids_filters, **query}
        files = layout.get(**query)
        if (key.startswith('chisep_') or key.startswith('sepia_')) and len(files) == 0:
            print(f'No files found for {key} with query {query}')
            run_data[key] = None
            continue
        elif len(files) != 1:
            raise ValueError(f'Expected 1 file for {key}, got {len(files)} with query {query}')

        file = files[0]
        run_data[key] = file.path

    pprint(run_data)

    return run_data


def process_run(layout, run_data, out_dir):
    """Process a single run of QSM data.

    Parameters
    ----------
    layout : BIDSLayout
        BIDSLayout object for the dataset.
    run_data : dict
        Dictionary containing the paths to the QSM data.
    out_dir : str
        Path to the output directory.
    """
    # Coregister MEGRE data to preprocessed T1w
    coreg_transform = run_data['megre2t1w_xfm']

    # Warp T1w-space T2*map, R2*map, and S0map to MNI152NLin2009cAsym using normalization
    # transform from sMRIPrep and coregistration transform to sMRIPrep's T1w space.
    keys = [
        # SEPIA
        'sepia_chimap_e12345',
        'sepia_chimap_e2345',
        # Chi-separation with input R2' map
        'chisep_chimap_r2p_e12345',
        'chisep_chimap_r2p_e2345',
        'chisep_iron_r2p_e12345',
        'chisep_iron_r2p_e2345',
        'chisep_myelin_r2p_e12345',
        'chisep_myelin_r2p_e2345',
        # Chi-separation with R2' map from r2primenet
        'chisep_chimap_r2primenet_e12345',
        'chisep_chimap_r2primenet_e2345',
        'chisep_iron_r2primenet_e12345',
        'chisep_iron_r2primenet_e2345',
        'chisep_myelin_r2primenet_e12345',
        'chisep_myelin_r2primenet_e2345',
        # Chi-separation with R2* map from ARLO
        'chisep_chimap_r2s_e12345',
        'chisep_chimap_r2s_e2345',
        'chisep_iron_r2s_e12345',
        'chisep_iron_r2s_e2345',
        'chisep_myelin_r2s_e12345',
        'chisep_myelin_r2s_e2345',
    ]
    for key in keys:
        file_ = run_data[key]
        if file_ is None:
            continue

        suffix = os.path.basename(file_).split('_')[-1].split('.')[0]

        # Coregister to T1w
        t1w_file = get_filename(
            name_source=file_,
            layout=layout,
            out_dir=out_dir,
            entities={'space': 'T1w'},
            dismiss_entities=['echo', 'part'],
        )
        reg_img = ants.apply_transforms(
            fixed=ants.image_read(run_data['t1w']),
            moving=ants.image_read(file_),
            transformlist=[coreg_transform],
            interpolator='lanczosWindowedSinc',
        )
        ants.image_write(reg_img, t1w_file)

        # Coregister to MNI152NLin2009cAsym
        mni_file = get_filename(
            name_source=file_,
            layout=layout,
            out_dir=out_dir,
            entities={'space': 'MNI152NLin2009cAsym'},
            dismiss_entities=['echo', 'part'],
        )
        reg_img = ants.apply_transforms(
            fixed=ants.image_read(run_data['t1w_mni']),
            moving=ants.image_read(file_),
            transformlist=[run_data['t1w2mni_xfm'], coreg_transform],
            interpolator='lanczosWindowedSinc',
        )
        ants.image_write(reg_img, mni_file)

        # Plot coregistration
        plot_coregistration(
            name_source=mni_file,
            layout=layout,
            in_file=mni_file,
            t1_file=run_data['t1w_mni'],
            out_dir=out_dir,
            source_space=suffix,
            target_space='MNI152NLin2009cAsym',
        )

        # Plot scalar map
        desc = 'scalar'
        if 'desc-' in mni_file:
            # Append the desc to the target desc
            desc = mni_file.split('desc-')[-1].split('_')[0] + 'scalar'

        scalar_report = get_filename(
            name_source=mni_file,
            layout=layout,
            out_dir=out_dir,
            entities={'datatype': 'figures', 'desc': desc, 'extension': '.svg'},
        )
        data = masking.apply_mask(mni_file, run_data['mni_mask'])
        vmin = np.percentile(data, 2)
        vmin = np.minimum(vmin, 0)
        vmax = np.percentile(data, 98)
        plot_scalar_map(
            underlay=run_data['t1w_mni'],
            overlay=mni_file,
            mask=run_data['mni_mask'],
            dseg=run_data['dseg_mni'],
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
    """Run the process_qsm_post workflow."""
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

    bootstrap_file = os.path.join(CODE_DIR, 'processing', 'reports_spec_qsm.yml')
    assert os.path.isfile(bootstrap_file), f'Bootstrap file {bootstrap_file} not found'

    layout = BIDSLayout(
        in_dir,
        config=os.path.join(CODE_DIR, 'nibs_bids_config.json'),
        validate=False,
        derivatives=[smriprep_dir, out_dir],
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
            'Name': 'NIBS QSM Derivatives',
            'BIDSVersion': '1.10.0',
            'DatasetType': 'derivative',
            'DatasetLinks': {
                'raw': in_dir,
                'smriprep': smriprep_dir,
                'mese': mese_dir,
            },
            'GeneratedBy': [
                {
                    'Name': 'Custom code',
                    'Description': 'Custom Python code combining ANTsPy and tedana.',
                    'CodeURL': 'https://github.com/PennLINC/nibs',
                }
            ],
        }
        with open(dataset_description_file, 'w') as fobj:
            json.dump(dataset_description, fobj, sort_keys=True, indent=4)

    print('DONE!')


if __name__ == '__main__':
    _main()
