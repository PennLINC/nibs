"""Calculate the scaling factors for the MTsat and ihMTR measures to achieve a mean g-ratio of
0.7 in the splenium.

The g-ratio formula is

g-ratio = sqrt(FVF / (FVF + (MVF * scaling_factor)))

where MVF is the MTsat or ihMTR value. FVF is held constant, so we need to solve for the scaling factor.

The equation is solved using the mean g-ratio in the splenium across subjects.
I need to solve for scaling_factor so that g = 0.7.
The first step is to calculate the splenium mask,
then calculate mean FVF and MVF in the splenium across subjects.

g = sqrt(FVF / (FVF + (MVF * scaling_factor)))

(g ** 2) = FVF / (FVF + (MVF * scaling_factor))

(g ** 2) * (FVF + (MVF * scaling_factor)) = FVF

((g ** 2) * FVF) + ((g ** 2) * MVF * scaling_factor) = FVF

((g ** 2) * MVF * scaling_factor) = FVF - (FVF * (g ** 2))

((g ** 2) * MVF * scaling_factor) = FVF * (1 - (g ** 2))

scaling_factor = (FVF * (1 - (g ** 2))) / ((g ** 2) * MVF)

# Now set g to 0.7
scaling_factor = (FVF * (1 - (0.7 ** 2))) / ((0.7 ** 2) * MVF)

scaling_factor = (FVF * (1 - 0.49)) / (0.49 * MVF)

scaling_factor = (FVF * 0.51) / (0.49 * MVF)

# We need to do this for each MTsat- and ihMTR-derived g-ratio combination, namely
# MTsat+ISOVF/ICVF, MTsat+AWF, ihMTR+ISOVF/ICVF, and ihMTR+AWF.
"""

import argparse
import json
import os

import ants
import nibabel as nb
import numpy as np
from bids.layout import BIDSLayout, Query
from nilearn import masking

from utils import get_filename

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
        'awf_mni': {
            'datatype': 'dwi',
            'session': [Query.NONE, Query.ANY],
            'run': [Query.NONE, Query.ANY],
            'space': 'MNI152NLin2009cAsym',
            'model': 'noddi',
            'param': 'awf',
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
        't1w2fs_xfm': {
            'datatype': 'anat',
            'session': [Query.NONE, Query.ANY],
            'run': [Query.NONE, Query.ANY],
            'from': 'T1w',
            'to': 'fsnative',
            'mode': 'image',
            'suffix': 'xfm',
            'extension': '.txt',
        },
        # T1w-space aseg dseg from sMRIPrep
        'aseg_t1w': {
            'datatype': 'anat',
            'session': [Query.NONE, Query.ANY],
            'run': [Query.NONE, Query.ANY],
            'space': 'T1w',
            'desc': 'aseg',
            'suffix': 'dseg',
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

    Returns
    -------
    splenium_g_ratios : np.ndarray of shape (4, n_voxels)
        Array of g-ratios in the splenium.
    """
    # MVF measures: MPRAGE T1w/T2w ratio, SPACE T1w/T2w ratio, MTsat, ihMTR
    mtsat_mvf = ants.image_read(run_data['mtsat_mni'])
    ihmtr_mvf = ants.image_read(run_data['ihmtr_mni'])

    # FVF/AVF measures: ISOVF, ICVF, AWF
    isovf = ants.image_read(run_data['isovf_mni'])
    icvf = ants.image_read(run_data['icvf_mni'])
    awf = ants.image_read(run_data['awf_mni'])

    mtsat_isovf_icvf_fvf = mtsat_mvf * (1 - isovf) * icvf
    ihmtr_isovf_icvf_fvf = ihmtr_mvf * (1 - isovf) * icvf

    # Warp each image to Freesurfer space
    mtsat_mvf_fs = ants.apply_transforms(
        fixed=ants.image_read(run_data['t1w']),
        moving=mtsat_mvf,
        transformlist=[run_data['t1w2fs_xfm'], run_data['mni2t1w_xfm']],
        interpolator='lanczosWindowedSinc',
    )

    # G = sqrt(FVF / (FVF + MVF))
    imgs = {}
    imgs['MTsat+ISOVF+ICVF'] = np.sqrt(mtsat_isovf_icvf_fvf / (mtsat_isovf_icvf_fvf + mtsat_mvf))
    imgs['ihMTR+ISOVF+ICVF'] = np.sqrt(ihmtr_isovf_icvf_fvf / (ihmtr_isovf_icvf_fvf + ihmtr_mvf))
    imgs['MTsat+AWF'] = np.sqrt(awf / (awf + mtsat_mvf))
    imgs['ihMTR+AWF'] = np.sqrt(awf / (awf + ihmtr_mvf))

    for i_img, (_, img) in enumerate(imgs.items()):
        data = masking.apply_mask(img, run_data['splenium_mask'])
        if i_img == 0:
            n_voxels = data.shape[0]
            splenium_fvfs = np.zeros((4, n_voxels))
            splenium_mvfs = np.zeros((4, n_voxels))

        splenium_fvfs[i_img, :] = data
        splenium_mvfs[i_img, :] = data

    return splenium_fvfs, splenium_mvfs


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
