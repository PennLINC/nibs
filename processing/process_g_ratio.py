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
import nibabel as nb
import numpy as np
from bids.layout import BIDSLayout, Query
from nilearn import masking
from nireports.assembler.report import Report

from utils import get_filename, plot_scalar_map

CODE_DIR = '/cbica/projects/nibs/code'
# Scaling factors to be adjusted so that mean g-ratios in splenium are 0.7 across the sample.
ALPHA_MTR = 1
ALPHA_MTSAT = 1


def collect_run_data(layout, bids_filters):
    queries = {
        'space_t1w': {
            'part': Query.NONE,
            'acquisition': 'SPACE',
            'space': Query.NONE,
            'desc': Query.NONE,
            'suffix': 'T1w',
            'extension': ['.nii', '.nii.gz'],
        },
        'space_t2w': {
            'part': Query.NONE,
            'acquisition': 'SPACE',
            'space': Query.NONE,
            'desc': Query.NONE,
            'suffix': 'T2w',
            'extension': ['.nii', '.nii.gz'],
        },
        'mprage_t1w': {
            'part': Query.NONE,
            'acquisition': 'MPRAGE',
            'space': Query.NONE,
            'desc': Query.NONE,
            'suffix': 'T1w',
            'extension': ['.nii', '.nii.gz'],
        },
        # T1w-space T1w image from sMRIPrep
        't1w': {
            'datatype': 'anat',
            'session': [Query.NONE, Query.ANY],
            'run': [Query.NONE, Query.ANY],
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
            'space': 'MNI152NLin2009cAsym',
            'desc': 'preproc',
            'suffix': 'T1w',
            'extension': ['.nii', '.nii.gz'],
        },
        # MNI-space MPRAGE T1w/T2w ratio map from process_t1wt2w_ratio.py
        'mprage_t1w_t2w_ratio_mni': {
            'datatype': 'anat',
            'session': [Query.NONE, Query.ANY],
            'run': [Query.NONE, Query.ANY],
            'space': 'MNI152NLin2009cAsym',
            'desc': 'MPRAGEscaled',
            'suffix': 'myelinw',
            'extension': ['.nii', '.nii.gz'],
        },
        # MNI-space SPACE T1w/T2w ratio map from process_t1wt2w_ratio.py
        'space_t1w_t2w_ratio_mni': {
            'datatype': 'anat',
            'session': [Query.NONE, Query.ANY],
            'run': [Query.NONE, Query.ANY],
            'space': 'MNI152NLin2009cAsym',
            'desc': 'SPACEscaled',
            'suffix': 'myelinw',
            'extension': ['.nii', '.nii.gz'],
        },
        # MNI-space ISOVF and ICVF maps from QSIRecon
        'isovf_mni': {
            'datatype': 'dwi',
            'session': [Query.NONE, Query.ANY],
            'run': [Query.NONE, Query.ANY],
            'space': 'MNI152NLin2009cAsym',
            'model': 'noddi',
            'param': 'isovf',
            'suffix': 'dwimap',
            'extension': ['.nii', '.nii.gz'],
        },
        'icvf_mni': {
            'datatype': 'dwi',
            'session': [Query.NONE, Query.ANY],
            'run': [Query.NONE, Query.ANY],
            'space': 'MNI152NLin2009cAsym',
            'model': 'noddi',
            'param': 'icvf',
            'suffix': 'dwimap',
            'extension': ['.nii', '.nii.gz'],
        },
        # MNI-space MTsat and ihMTR maps from process_ihmt.py
        'mtsat_mni': {
            'datatype': 'anat',
            'session': [Query.NONE, Query.ANY],
            'run': [Query.NONE, Query.ANY],
            'space': 'MNI152NLin2009cAsym',
            'suffix': 'MTsat',
            'extension': ['.nii', '.nii.gz'],
        },
        'ihmtr_mni': {
            'datatype': 'anat',
            'session': [Query.NONE, Query.ANY],
            'run': [Query.NONE, Query.ANY],
            'space': 'MNI152NLin2009cAsym',
            'suffix': 'ihMTR',
            'extension': ['.nii', '.nii.gz'],
        },
        # Coregistration transform for MPRAGE, from sMRIPrep
        'mprage2t1w_xfm': {
            'datatype': 'anat',
            'session': [Query.NONE, Query.ANY],
            'run': [Query.NONE, Query.ANY],
            'from': 'orig',
            'to': 'T1w',
            'mode': 'image',
            'suffix': 'xfm',
            'extension': '.txt',
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
        # Normalization transform from sMRIPrep
        'mni2t1w_xfm': {
            'datatype': 'anat',
            'session': [Query.NONE, Query.ANY],
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
    # Get WM segmentation from sMRIPrep
    wm_seg_img = nb.load(run_data['dseg_mni'])
    wm_seg = wm_seg_img.get_fdata()
    wm_seg = (wm_seg == 2).astype(int)
    wm_seg_file = get_filename(
        name_source=run_data['dseg_mni'],
        layout=layout,
        out_dir=out_dir,
        entities={'space': 'MNI152NLin2009cAsym', 'desc': 'wm', 'suffix': 'mask'},
    )
    wm_seg_img = nb.Nifti1Image(wm_seg, wm_seg_img.affine, wm_seg_img.header)
    wm_seg_img.to_filename(wm_seg_file)

    # Warp WM segmentation to T1w space
    wm_seg_img = ants.image_read(wm_seg_file)
    wm_seg_t1w_img = ants.apply_transforms(
        fixed=ants.image_read(run_data['t1w']),
        moving=wm_seg_img,
        transformlist=[run_data['mni2t1w_xfm']],
    )
    wm_seg_t1w_file = get_filename(
        name_source=wm_seg_file,
        layout=layout,
        out_dir=out_dir,
        entities={'space': 'T1w', 'desc': 'wm', 'suffix': 'mask'},
    )
    ants.image_write(wm_seg_t1w_img, wm_seg_t1w_file)
    del wm_seg_img, wm_seg_t1w_img, wm_seg

    # MVF measures: MPRAGE T1w/T2w ratio, SPACE T1w/T2w ratio, MTsat, ihMTR
    mprage_t1wt2w_mvf = ants.image_read(run_data['mprage_t1w_t2w_ratio_mni'])
    space_t1wt2w_mvf = ants.image_read(run_data['space_t1w_t2w_ratio_mni'])
    mtsat_mvf = ants.image_read(run_data['mtsat_mni']) * ALPHA_MTSAT
    ihmtr_mvf = ants.image_read(run_data['ihmtr_mni']) * ALPHA_MTR

    # FVF/AVF measures: ISOVF, ICVF, AWF
    isovf = ants.image_read(run_data['isovf_mni'])
    icvf = ants.image_read(run_data['icvf_mni'])
    awf = ants.image_read(run_data['awf_mni'])

    mprage_t1wt2w_isovf_icvf_fvf = (1 - mprage_t1wt2w_mvf) * (1 - isovf) * icvf
    space_t1wt2w_isovf_icvf_fvf = (1 - space_t1wt2w_mvf) * (1 - isovf) * icvf
    mtsat_isovf_icvf_fvf = mtsat_mvf * (1 - isovf) * icvf
    ihmtr_isovf_icvf_fvf = ihmtr_mvf * (1 - isovf) * icvf

    # G = sqrt(FVF / (FVF + MVF))
    mprage_t1wt2w_isovf_icvf_g = np.sqrt(mprage_t1wt2w_isovf_icvf_fvf / (mprage_t1wt2w_isovf_icvf_fvf + mprage_t1wt2w_mvf))
    space_t1wt2w_isovf_icvf_g = np.sqrt(space_t1wt2w_isovf_icvf_fvf / (space_t1wt2w_isovf_icvf_fvf + space_t1wt2w_mvf))
    mtsat_isovf_icvf_g = np.sqrt(mtsat_isovf_icvf_fvf / (mtsat_isovf_icvf_fvf + mtsat_mvf))
    ihmtr_isovf_icvf_g = np.sqrt(ihmtr_isovf_icvf_fvf / (ihmtr_isovf_icvf_fvf + ihmtr_mvf))
    mprage_t1wt2w_awf_g = np.sqrt(awf / (awf + mprage_t1wt2w_mvf))
    space_t1wt2w_awf_g = np.sqrt(awf / (awf + space_t1wt2w_mvf))
    mtsat_awf_g = np.sqrt(awf / (awf + mtsat_mvf))
    ihmtr_awf_g = np.sqrt(awf / (awf + ihmtr_mvf))

    # Warp T1w-space SPACE T1w/SPACE T2w and MPRAGE T1w/SPACE T2w ratio maps to MNI152NLin2009cAsym
    # using normalization transform from sMRIPrep.
    files = [t1w_space_ratio_file, t1w_mprage_ratio_file]
    descs = ['SPACE', 'MPRAGE']
    for i_file, file_ in enumerate(files):
        desc = descs[i_file]
        mni_file = get_filename(
            name_source=file_,
            layout=layout,
            out_dir=out_dir,
            entities={'space': 'MNI152NLin2009cAsym'},
            dismiss_entities=['reconstruction', 'acquisition'],
        )
        mni_img = ants.apply_transforms(
            fixed=ants.image_read(run_data['t1w_mni']),
            moving=ants.image_read(file_),
            transformlist=[run_data['t1w2mni_xfm']],
            interpolator='lanczosWindowedSinc',
        )
        ants.image_write(mni_img, mni_file)

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
    out_dir = '/cbica/projects/nibs/derivatives/t1wt2w_ratio'
    os.makedirs(out_dir, exist_ok=True)
    temp_dir = '/cbica/projects/nibs/work/t1wt2w_ratio'
    os.makedirs(temp_dir, exist_ok=True)

    bootstrap_file = os.path.join(CODE_DIR, 'processing', 'reports_spec_t1wt2w_ratio.yml')
    assert os.path.isfile(bootstrap_file), f'Bootstrap file {bootstrap_file} not found'

    layout = BIDSLayout(
        in_dir,
        config=os.path.join(CODE_DIR, 'nibs_bids_config.json'),
        validate=False,
        derivatives=[smriprep_dir],
    )

    print(f'Processing subject {subject_id}')
    sessions = layout.get_sessions(subject=subject_id, acquisition='SPACE', suffix='T2w')
    for session in sessions:
        print(f'Processing session {session}')
        space_t2w_files = layout.get(
            subject=subject_id,
            session=session,
            acquisition='SPACE',
            suffix='T2w',
            extension=['.nii', '.nii.gz'],
        )
        if not space_t2w_files:
            print(f'No SPACE T2w files found for subject {subject_id} and session {session}')
            continue

        for space_t2w_file in space_t2w_files:
            entities = space_t2w_file.get_entities()
            entities.pop('acquisition')
            try:
                run_data = collect_run_data(layout, entities)
            except ValueError as e:
                print(f'Failed {space_t2w_file}')
                print(e)
                continue

            fname = os.path.basename(space_t2w_file.path).split('.')[0]
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
            'Name': 'NIBS T1w/T2w Ratio Derivatives',
            'BIDSVersion': '1.10.0',
            'DatasetType': 'derivative',
            'DatasetLinks': {
                'raw': in_dir,
                'smriprep': smriprep_dir,
            },
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
