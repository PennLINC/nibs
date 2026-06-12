"""Process QSM data.

Steps:

1.  Rename copied PMACS QSM outputs to BIDS-compliant filenames.
2.  Warp QSM derivatives to T1w and MNI152NLin2009cAsym spaces.
3.  Generate scalar reports for QSM derivatives.

Notes:

- The R2* map is calculated using the monoexponential fit.
- This must be run after sMRIPrep, process_mese.py, and copying PMACS QSM outputs.
"""

from __future__ import annotations

import argparse
import json
import os
from glob import glob
from pprint import pformat

import ants
import nibabel as nb
import numpy as np
from bids.layout import BIDSLayout, Query
from nilearn import masking
from nireports.assembler.report import Report

from utils import (
    get_filename,
    load_config,
    plot_coregistration,
    plot_scalar_map,
)

CFG = load_config()
CODE_DIR = CFG['code_dir']


def _find_one_file(pattern: str, description: str) -> str | None:
    """Find a single file for an expected QSM output pattern."""
    matches = sorted(glob(pattern))
    if not matches:
        print(f'{description} not found with pattern: {pattern}')
        return None
    if len(matches) > 1:
        print(f'Multiple matches found for {description}; using {matches[0]}: {matches}')
    return matches[0]


def rename_qsm_outputs(subject_id: str, session: str) -> None:
    """Rename PMACS QSM outputs to BIDS-compliant filenames.

    Reads copied PMACS SEPIA/chi-separation NIfTIs and local r2p chi-separation
    NIfTIs, then writes compressed, BIDS-named files to the QSM derivatives
    directory.

    Parameters
    ----------
    subject_id : str
        BIDS subject label (without 'sub-' prefix).
    session : str
        BIDS session label (without 'ses-' prefix).
    """
    work_dir = CFG['work_dir']
    out_dir = CFG['derivatives']['qsm']
    ses_out_dir = os.path.join(out_dir, f'sub-{subject_id}', f'ses-{session}', 'anat')
    os.makedirs(ses_out_dir, exist_ok=True)

    qsm_dir = os.path.join(
        work_dir, 'qsm-pmacs', f'sub-{subject_id}', f'ses-{session}', 'anat', 'QSM'
    )

    sepia_outputs = [
        ('E12345+sepia', 'output'),
        ('E2345+sepia', 'outputsepia_2345'),
    ]
    chisep_outputs = [
        ('E12345+chisep+r2primenet', 'outputE12345', 'r2primenet'),
        ('E2345+chisep+r2primenet', 'outputE2345', 'r2primenet'),
        ('E12345+chisep+r2s', 'outputE12345', 'r2s'),
        ('E2345+chisep+r2s', 'outputE2345', 'r2s'),
    ]
    suffix_map = {
        'paramagnetic': 'para',
        'diamagnetic': 'dia',
        'total': 'Chimap',
    }

    if os.path.isdir(qsm_dir):
        for desc, sepia_dir in sepia_outputs:
            in_file = _find_one_file(
                os.path.join(qsm_dir, sepia_dir, '*Chimap.nii*'),
                f'SEPIA output for {desc}',
            )
            if in_file is None:
                continue
            out_file = os.path.join(
                ses_out_dir,
                f'sub-{subject_id}_ses-{session}_run-01_space-MEGRE_desc-{desc}_Chimap.nii.gz',
            )
            nb.load(in_file).to_filename(out_file)
    else:
        print(f'PMACS QSM output directory not found: {qsm_dir}')

    chisep_variant_dirs = [
        (
            desc,
            os.path.join(qsm_dir, output_dir),
            map_label,
        )
        for desc, output_dir, map_label in chisep_outputs
    ]
    chisep_variant_dirs.extend(
        [
            (
                'E12345+chisep+r2p',
                os.path.join(
                    work_dir,
                    'qsm-E12345+chisep+r2p',
                    f'sub-{subject_id}',
                    f'ses-{session}',
                    'anat',
                ),
                'r2p',
            ),
            (
                'E2345+chisep+r2p',
                os.path.join(
                    work_dir,
                    'qsm-E2345+chisep+r2p',
                    f'sub-{subject_id}',
                    f'ses-{session}',
                    'anat',
                ),
                'r2p',
            ),
        ]
    )

    for desc, variant_dir, map_label in chisep_variant_dirs:
        if not os.path.isdir(variant_dir):
            print(f'Chi-sep output directory not found: {variant_dir}')
            continue
        for contrast, bids_suffix in suffix_map.items():
            in_file = os.path.join(
                variant_dir,
                f'sub-{subject_id}_ses-{session}_{contrast}_{map_label}.nii',
            )
            out_file = os.path.join(
                ses_out_dir,
                f'sub-{subject_id}_ses-{session}_run-01_space-MEGRE_desc-{desc}_{bids_suffix}.nii.gz',
            )
            if not os.path.isfile(in_file):
                print(f'Chi-sep output not found: {in_file}')
                continue
            nb.load(in_file).to_filename(out_file)


def collect_run_data(layout: object, bids_filters: dict) -> dict[str, str]:
    """Collect SEPIA chi maps, GRE images, and masks for QSM post-processing.

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
            'suffix': 'para',
            'extension': ['.nii', '.nii.gz'],
        },
        'chisep_iron_r2p_e2345': {
            'datatype': 'anat',
            'run': [Query.NONE, Query.ANY],
            'space': 'MEGRE',
            'res': Query.NONE,
            'desc': 'E2345+chisep+r2p',
            'suffix': 'para',
            'extension': ['.nii', '.nii.gz'],
        },
        # Chisep myelin map with R2'
        'chisep_myelin_r2p_e12345': {
            'datatype': 'anat',
            'run': [Query.NONE, Query.ANY],
            'space': 'MEGRE',
            'res': Query.NONE,
            'desc': 'E12345+chisep+r2p',
            'suffix': 'dia',
            'extension': ['.nii', '.nii.gz'],
        },
        'chisep_myelin_r2p_e2345': {
            'datatype': 'anat',
            'run': [Query.NONE, Query.ANY],
            'space': 'MEGRE',
            'res': Query.NONE,
            'desc': 'E2345+chisep+r2p',
            'suffix': 'dia',
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
            'suffix': 'para',
            'extension': ['.nii', '.nii.gz'],
        },
        'chisep_iron_r2primenet_e2345': {
            'datatype': 'anat',
            'run': [Query.NONE, Query.ANY],
            'space': 'MEGRE',
            'res': Query.NONE,
            'desc': 'E2345+chisep+r2primenet',
            'suffix': 'para',
            'extension': ['.nii', '.nii.gz'],
        },
        # Chisep myelin map with R2'
        'chisep_myelin_r2primenet_e12345': {
            'datatype': 'anat',
            'run': [Query.NONE, Query.ANY],
            'space': 'MEGRE',
            'res': Query.NONE,
            'desc': 'E12345+chisep+r2primenet',
            'suffix': 'dia',
            'extension': ['.nii', '.nii.gz'],
        },
        'chisep_myelin_r2primenet_e2345': {
            'datatype': 'anat',
            'run': [Query.NONE, Query.ANY],
            'space': 'MEGRE',
            'res': Query.NONE,
            'desc': 'E2345+chisep+r2primenet',
            'suffix': 'dia',
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
            'suffix': 'para',
            'extension': ['.nii', '.nii.gz'],
        },
        'chisep_iron_r2s_e2345': {
            'datatype': 'anat',
            'run': [Query.NONE, Query.ANY],
            'space': 'MEGRE',
            'res': Query.NONE,
            'desc': 'E2345+chisep+r2s',
            'suffix': 'para',
            'extension': ['.nii', '.nii.gz'],
        },
        # Chisep myelin map with R2*
        'chisep_myelin_r2s_e12345': {
            'datatype': 'anat',
            'run': [Query.NONE, Query.ANY],
            'space': 'MEGRE',
            'res': Query.NONE,
            'desc': 'E12345+chisep+r2s',
            'suffix': 'dia',
            'extension': ['.nii', '.nii.gz'],
        },
        'chisep_myelin_r2s_e2345': {
            'datatype': 'anat',
            'run': [Query.NONE, Query.ANY],
            'space': 'MEGRE',
            'res': Query.NONE,
            'desc': 'E2345+chisep+r2s',
            'suffix': 'dia',
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

    print(f'Collected run data:\n{pformat(run_data, indent=4)}', flush=True)
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
            interpolator='nearestNeighbor',
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
            interpolator='nearestNeighbor',
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


def _get_parser() -> argparse.ArgumentParser:
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
    in_dir = CFG['bids_dir']
    smriprep_dir = CFG['derivatives']['smriprep']
    mese_dir = CFG['derivatives']['mese']
    out_dir = CFG['derivatives']['qsm']
    os.makedirs(out_dir, exist_ok=True)
    temp_dir = os.path.join(CFG['work_dir'], 'qsm')
    os.makedirs(temp_dir, exist_ok=True)

    bootstrap_file = os.path.join(CODE_DIR, 'configuration', 'reports_spec_qsm.yml')
    assert os.path.isfile(bootstrap_file), f'Bootstrap file {bootstrap_file} not found'

    session_search_patterns = [
        os.path.join(CFG['work_dir'], 'qsm-pmacs', f'sub-{subject_id}', 'ses-*'),
        os.path.join(CFG['work_dir'], 'qsm-E12345+chisep+r2p', f'sub-{subject_id}', 'ses-*'),
        os.path.join(CFG['work_dir'], 'qsm-E2345+chisep+r2p', f'sub-{subject_id}', 'ses-*'),
    ]
    print(f'searching: {session_search_patterns}', flush=True)

    sessions_to_rename = set()
    for search_pattern in session_search_patterns:
        sessions_to_rename.update(
            os.path.basename(d).removeprefix('ses-')
            for d in glob(search_pattern)
            if os.path.isdir(d)
        )
    print(f'sessions to rename: {sessions_to_rename}', flush=True)
    for session in sorted(sessions_to_rename):
        print(f'Renaming QSM outputs for session {session}')
        rename_qsm_outputs(subject_id, session)

    layout = BIDSLayout(
        in_dir,
        config=os.path.join(CODE_DIR, 'configuration', 'nibs_bids_config.json'),
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
