"""Process QSM data.

Steps:

1.  Warp QSM derivatives to T1w and MNI152NLin2009cAsym spaces.
2.  Generate scalar reports for QSM derivatives.

Notes:

- The R2* map is calculated using the monoexponential fit.
- This must be run after sMRIPrep and process_mese.py.
"""
import json
import os
from pprint import pprint

import ants
from bids.layout import BIDSLayout, Query
from nireports.assembler.report import Report

from utils import (
    get_filename,
    plot_coregistration,
    plot_scalar_map,
)


def collect_run_data(layout, bids_filters):
    queries = {
        # SEPIA Chimap
        'sepia_chimap': {
            'datatype': 'anat',
            'run': [Query.NONE, Query.ANY],
            'space': 'MEGRE',
            'res': Query.NONE,
            'desc': 'SEPIA',
            'suffix': 'Chimap',
            'extension': ['.nii', '.nii.gz'],
        },
        # Chisep iron map with R2'
        'chisep_iron_w_r2p': {
            'datatype': 'anat',
            'run': [Query.NONE, Query.ANY],
            'space': 'MEGRE',
            'res': Query.NONE,
            'desc': 'chisepR2p',
            'suffix': 'ironw',
            'extension': ['.nii', '.nii.gz'],
        },
        # Chisep myelin map with R2'
        'chisep_myelin_w_r2p': {
            'datatype': 'anat',
            'run': [Query.NONE, Query.ANY],
            'space': 'MEGRE',
            'res': Query.NONE,
            'desc': 'chisepR2p',
            'suffix': 'myelinw',
            'extension': ['.nii', '.nii.gz'],
        },
        # Chisep iron map with R2'
        'chisep_iron_wo_r2p': {
            'datatype': 'anat',
            'run': [Query.NONE, Query.ANY],
            'space': 'MEGRE',
            'res': Query.NONE,
            'desc': 'chisep',
            'suffix': 'ironw',
            'extension': ['.nii', '.nii.gz'],
        },
        # Chisep myelin map with R2'
        'chisep_myelin_wo_r2p': {
            'datatype': 'anat',
            'run': [Query.NONE, Query.ANY],
            'space': 'MEGRE',
            'res': Query.NONE,
            'desc': 'chisep',
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
        if len(files) != 1:
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
        Dictionary containing the paths to the MESE data.
    out_dir : str
        Path to the output directory.
    """
    # Coregister MEGRE data to preprocessed T1w
    coreg_transform = run_data['megre2t1w_xfm']

    # Warp T1w-space T2*map, R2*map, and S0map to MNI152NLin2009cAsym using normalization
    # transform from sMRIPrep and coregistration transform to sMRIPrep's T1w space.
    for file_ in [
        run_data['sepia_chimap'],
        run_data['chisep_iron_w_r2p'],
        run_data['chisep_myelin_w_r2p'],
        run_data['chisep_iron_wo_r2p'],
        run_data['chisep_myelin_wo_r2p'],
    ]:
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
        scalar_report = get_filename(
            name_source=mni_file,
            layout=layout,
            out_dir=out_dir,
            entities={'datatype': 'figures', 'desc': 'scalar', 'extension': '.svg'},
        )
        kwargs = {}
        plot_scalar_map(
            underlay=run_data['t1w_mni'],
            overlay=mni_file,
            mask=run_data['mni_mask'],
            dseg=run_data['dseg_mni'],
            out_file=scalar_report,
            **kwargs,
        )


if __name__ == '__main__':
    code_dir = '/cbica/projects/nibs/code'
    in_dir = '/cbica/projects/nibs/dset'
    smriprep_dir = '/cbica/projects/nibs/derivatives/smriprep'
    mese_dir = '/cbica/projects/nibs/derivatives/mese'
    out_dir = '/cbica/projects/nibs/derivatives/qsm'
    os.makedirs(out_dir, exist_ok=True)
    temp_dir = '/cbica/projects/nibs/work/qsm'
    os.makedirs(temp_dir, exist_ok=True)

    bootstrap_file = os.path.join(code_dir, 'processing', 'reports_spec_qsm.yml')
    assert os.path.isfile(bootstrap_file), f'Bootstrap file {bootstrap_file} not found'

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

            report_dir = os.path.join(out_dir, f'sub-{subject}', f'ses-{session}')
            robj = Report(
                report_dir,
                run_uuid=None,
                bootstrap_file=bootstrap_file,
                out_filename=f'sub-{subject}_ses-{session}.html',
                reportlets_dir=out_dir,
                plugins=None,
                plugin_meta=None,
                subject=subject,
                session=session,
            )
            robj.generate_report()

    print('DONE!')
