"""Calculate g ratio maps.

Steps:

1.  Calculate g ratio maps from various combinations.
2.  Warp g ratio maps to MNI152NLin2009cAsym using normalization transform from sMRIPrep.

Notes:

- sMRIPrep's preprocessed T1w image is used as the "native T1w space".
- This must be run after sMRIPrep and process_mp2rage.py.
"""

import argparse
import json
import os

import ants
import numpy as np
from bids.layout import BIDSLayout, Query
from nilearn import masking
from nireports.assembler.report import Report

from utils import get_filename, plot_scalar_map

CODE_DIR = '/cbica/projects/nibs/code'
# Scaling factors to be adjusted so that mean g-ratios in splenium are 0.7 across the sample.
MTsat_ISOVF_ICVF_scalar = 0.5890154242515564
ihMTR_ISOVF_ICVF_scalar = 0.5788185596466064


def collect_run_data(layout, bids_filters):
    queries = {
        # T1w-space T1w image from sMRIPrep
        't1w': {
            'datatype': 'anat',
            'session': [Query.NONE, Query.ANY],
            'run': [Query.NONE, Query.ANY],
            'reconstruction': [Query.NONE, Query.ANY],
            'space': Query.NONE,
            'res': Query.NONE,
            'desc': 'preproc',
            'suffix': 'T1w',
            'extension': ['.nii', '.nii.gz'],
        },
        # MNI-space T1w image from sMRIPrep
        't1w_mni': {
            'datatype': 'anat',
            'session': [Query.NONE, Query.ANY],
            'run': [Query.NONE, Query.ANY],
       	    'reconstruction': [Query.NONE, Query.ANY],
            'space': 'MNI152NLin2009cAsym',
            'desc': 'preproc',
            'suffix': 'T1w',
            'extension': ['.nii', '.nii.gz'],
        },
        # MNI-space MPRAGE T1w/T2w ratio map from process_t1wt2w_ratio.py
        'mprage_t1w_t2w_ratio_mni': {
            'datatype': 'anat',
            'run': [Query.NONE, Query.ANY],
       	    'reconstruction': [Query.NONE, Query.ANY],
            'space': 'MNI152NLin2009cAsym',
            'desc': 'MPRAGEscaled',
            'suffix': 'myelinw',
            'extension': ['.nii', '.nii.gz'],
        },
        # MNI-space SPACE T1w/T2w ratio map from process_t1wt2w_ratio.py
        'space_t1w_t2w_ratio_mni': {
            'datatype': 'anat',
            'run': [Query.NONE, Query.ANY],
       	    'reconstruction': [Query.NONE, Query.ANY],
            'space': 'MNI152NLin2009cAsym',
            'desc': 'SPACEscaled',
            'suffix': 'myelinw',
            'extension': ['.nii', '.nii.gz'],
        },
        # MNI-space ISOVF and ICVF maps from QSIRecon
        'isovf_mni': {
            'datatype': 'dwi',
            'run': [Query.NONE, Query.ANY],
       	    'reconstruction': [Query.NONE, Query.ANY],
            'space': 'MNI152NLin2009cAsym',
            'model': 'noddi',
            'param': 'isovf',
            'desc': Query.NONE,
            'suffix': 'dwimap',
            'extension': ['.nii', '.nii.gz'],
        },
        'icvf_mni': {
            'datatype': 'dwi',
            'run': [Query.NONE, Query.ANY],
       	    'reconstruction': [Query.NONE, Query.ANY],
            'space': 'MNI152NLin2009cAsym',
            'model': 'noddi',
            'param': 'icvf',
            'desc': Query.NONE,
            'suffix': 'dwimap',
            'extension': ['.nii', '.nii.gz'],
        },
        # MNI-space MTsat and ihMTR maps from process_ihmt.py
        'mtsat_mni': {
            'datatype': 'anat',
            'run': [Query.NONE, Query.ANY],
       	    'reconstruction': [Query.NONE, Query.ANY],
            'space': 'MNI152NLin2009cAsym',
            'suffix': 'ihMTsatB1sq',
            'extension': ['.nii', '.nii.gz'],
        },
        'ihmtr_mni': {
            'datatype': 'anat',
            'run': [Query.NONE, Query.ANY],
       	    'reconstruction': [Query.NONE, Query.ANY],
            'space': 'MNI152NLin2009cAsym',
            'suffix': 'ihMTR',
            'extension': ['.nii', '.nii.gz'],
        },
        # Coregistration transform for MPRAGE, from sMRIPrep
        'mprage2t1w_xfm': {
            'datatype': 'anat',
            'space': Query.NONE,
       	    'reconstruction': [Query.NONE, Query.ANY],
            'run': [Query.NONE, Query.ANY],
            'from': 'orig',
            'to': 'T1w',
            'mode': 'image',
            'suffix': 'xfm',
            'extension': '.txt',
        },
        # Normalization transform from sMRIPrep
        'mni2t1w_xfm': {
            'datatype': 'anat',
            'session': [Query.NONE, Query.ANY],
       	    'reconstruction': [Query.NONE, Query.ANY],
            'space': Query.NONE,
            'run': [Query.NONE, Query.ANY],
            'from': 'MNI152NLin2009cAsym',
            'to': 'T1w',
            'mode': 'image',
            'suffix': 'xfm',
            'extension': '.h5',
        },
        # MNI-space dseg from sMRIPrep
        'dseg_mni': {
            'datatype': 'anat',
            'session': [Query.NONE, Query.ANY],
       	    'reconstruction': [Query.NONE, Query.ANY],
            'run': [Query.NONE, Query.ANY],
            'space': 'MNI152NLin2009cAsym',
            'suffix': 'dseg',
            'extension': ['.nii', '.nii.gz'],
        },
        # sMRIPrep MNI-space brain mask
        'mni_mask': {
            'datatype': 'anat',
            'session': [Query.NONE, Query.ANY],
       	    'reconstruction': [Query.NONE, Query.ANY],
            'run': [Query.NONE, Query.ANY],
            'space': 'MNI152NLin2009cAsym',
            'desc': 'brain',
            'suffix': 'mask',
            'extension': ['.nii', '.nii.gz'],
        },
    }

    run_data = {}
    for key, query in queries.items():
        # I have no clue why, but BIDSLayout refuses to index 'param'
        param = None
        if 'param' in query:
            param = query.pop('param')

        query = {**bids_filters, **query}
        files = layout.get(**query)
        if param is not None:
            files = [f for f in files if f'_param-{param}_' in f.filename]
            if param == 'fa':
                # Both DIPYDKI and DSIStudio have 'fa' as a param. Use DIPYDKI.
                files = [f for f in files if 'qsirecon-DIPYDKI' in f.path]
            query['param'] = param

        if key == 'mprage2t1w_xfm' and len(files) == 0:
            print(f'No MPRAGE T1w coregistration transform found for {query}. Using identity transform.')
            run_data[key] = None
            continue
        elif len(files) != 1:
            raise ValueError(f'Expected 1 file for {key}, got {len(files)}: {query}')

        file = files[0]
        run_data[key] = file.path

    return run_data


def process_run(layout, run_data, out_dir, temp_dir):
    """Process a single MP2RAGE run.

    Parameters
    ----------
    layout : BIDSLayout
        BIDSLayout object.
    run_data : dict
        Dictionary of run data.
    out_dir : str
        Directory to write output files.
    temp_dir : str
        Directory to write temporary files.
    """
    # MVF measures: MPRAGE T1w/T2w ratio, SPACE T1w/T2w ratio, MTsat, ihMTR
    mprage_t1wt2w_mvf = ants.image_read(run_data['mprage_t1w_t2w_ratio_mni'])
    space_t1wt2w_mvf = ants.image_read(run_data['space_t1w_t2w_ratio_mni'])
    # Eq. 3 in Berg et al. (2022)
    mtsat_mvf = ants.image_read(run_data['mtsat_mni']) * MTsat_ISOVF_ICVF_scalar
    # Eq. 4 in Berg et al. (2022)
    ihmtr_mvf = ants.image_read(run_data['ihmtr_mni']) * ihMTR_ISOVF_ICVF_scalar

    # FVF/AVF measures: ISOVF, ICVF
    # They're in MNI space, but 1.7 mm isotropic, so we need to resample to the MVF images
    # (1 mm isotropic)
    isovf = ants.image_read(run_data['isovf_mni']).resample_image_to_target(
        mtsat_mvf,
        interp_type='lanczosWindowedSinc',
    )
    icvf = ants.image_read(run_data['icvf_mni']).resample_image_to_target(
        mtsat_mvf,
        interp_type='lanczosWindowedSinc',
    )

    # Eq 6 in Berg et al. (2022)
    mprage_t1wt2w_fvf = (1 - mprage_t1wt2w_mvf) * (1 - isovf) * icvf
    space_t1wt2w_fvf = (1 - space_t1wt2w_mvf) * (1 - isovf) * icvf
    mtsat_fvf = (1 - mtsat_mvf) * (1 - isovf) * icvf
    ihmtr_fvf = (1 - ihmtr_mvf) * (1 - isovf) * icvf

    # G = sqrt(1 - (MVF / (MVF + AVF))) [Eq. 1 in Newman et al. (2024)]
    # Same as sqrt(AVF / (AVF + MVF)) [Eq. 1 in Berg et al. (2022)]
    imgs = {}
    imgs['MPRAGET1wT2w+ISOVF+ICVF'] = (mprage_t1wt2w_fvf / (mprage_t1wt2w_fvf + mprage_t1wt2w_mvf)) ** 0.5
    imgs['SPACET1wT2w+ISOVF+ICVF'] = (space_t1wt2w_fvf / (space_t1wt2w_fvf + space_t1wt2w_mvf)) ** 0.5
    imgs['MTsat+ISOVF+ICVF'] = (mtsat_fvf / (mtsat_fvf + mtsat_mvf)) ** 0.5
    imgs['ihMTR+ISOVF+ICVF'] = (ihmtr_fvf / (ihmtr_fvf + ihmtr_mvf)) ** 0.5

    for desc, img in imgs.items():
        mni_file = get_filename(
            name_source=run_data['mprage_t1w_t2w_ratio_mni'],
            layout=layout,
            out_dir=out_dir,
            entities={'space': 'MNI152NLin2009cAsym', 'desc': desc, 'suffix': 'gratio'},
            dismiss_entities=['reconstruction', 'acquisition'],
        )
        ants.image_write(img, mni_file)

        scalar_desc = 'scalar'
        if desc:
            scalar_desc = f'{desc}{scalar_desc}'

        data = masking.apply_mask(mni_file, run_data['mni_mask'])
        vmin = np.percentile(data, 2)
        vmin = np.minimum(vmin, 0)
        vmax = np.percentile(data, 98)

        scalar_report = get_filename(
            name_source=mni_file,
            layout=layout,
            out_dir=out_dir,
            entities={'datatype': 'figures', 'desc': scalar_desc, 'extension': '.svg'},
        )
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
    """Run the process_mese workflow."""
    options = _get_parser().parse_args(argv)
    kwargs = vars(options)
    main(**kwargs)


def main(subject_id):
    in_dir = '/cbica/projects/nibs/dset'
    smriprep_dir = '/cbica/projects/nibs/derivatives/smriprep'
    dipydki_dir = '/cbica/projects/nibs/derivatives/qsirecon/derivatives/qsirecon-DIPYDKI'
    noddi_dir = '/cbica/projects/nibs/derivatives/qsirecon/derivatives/qsirecon-NODDI'
    ihmt_dir = '/cbica/projects/nibs/derivatives/ihmt'
    t1wt2w_dir = '/cbica/projects/nibs/derivatives/t1wt2w_ratio'
    out_dir = '/cbica/projects/nibs/derivatives/g_ratio'
    os.makedirs(out_dir, exist_ok=True)
    temp_dir = '/cbica/projects/nibs/work/g_ratio'
    os.makedirs(temp_dir, exist_ok=True)

    bootstrap_file = os.path.join(CODE_DIR, 'processing', 'reports_spec_g_ratio.yml')
    assert os.path.isfile(bootstrap_file), f'Bootstrap file {bootstrap_file} not found'

    layout = BIDSLayout(
        in_dir,
        config=os.path.join(CODE_DIR, 'nibs_bids_config.json'),
        validate=False,
        derivatives=[smriprep_dir, dipydki_dir, noddi_dir, ihmt_dir, t1wt2w_dir],
    )

    base_query = {
        'space': 'MNI152NLin2009cAsym',
        'suffix': 'ihMTR',
        'extension': ['.nii', '.nii.gz'],
    }

    print(f'Processing subject {subject_id}')
    sessions = layout.get_sessions(subject=subject_id, **base_query)
    for session in sessions:
        print(f'Processing session {session}')
        base_files = layout.get(
            subject=subject_id,
            session=session,
            **base_query,
        )
        if not base_files:
            print(f'No base files found for subject {subject_id} and session {session}')
            continue

        for base_file in base_files:
            entities = base_file.get_entities()
            try:
                run_data = collect_run_data(layout, entities)
            except ValueError as e:
                print(f'Failed {base_file}')
                print(e)
                continue

            fname = os.path.basename(base_file.path).split('.')[0]
            run_temp_dir = os.path.join(temp_dir, fname.replace('-', '').replace('_', ''))
            os.makedirs(run_temp_dir, exist_ok=True)
            process_run(layout, run_data, out_dir, run_temp_dir)

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
            'Name': 'NIBS G-Ratio Derivatives',
            'BIDSVersion': '1.10.0',
            'DatasetType': 'derivative',
            'DatasetLinks': {
                'raw': in_dir,
                'smriprep': smriprep_dir,
                'dipydki': dipydki_dir,
                'noddi': noddi_dir,
                'ihmt': ihmt_dir,
            },
            'GeneratedBy': [
                {
                    'Name': 'Custom code',
                    'Description': 'Custom Python code.',
                    'CodeURL': 'https://github.com/PennLINC/nibs',
                }
            ],
        }
        with open(dataset_description_file, 'w') as fobj:
            json.dump(dataset_description, fobj, sort_keys=True, indent=4)

    print('DONE!')


if __name__ == '__main__':
    _main()
