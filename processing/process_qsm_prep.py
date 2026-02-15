"""Process QSM data.

Steps:

1.  Average the magnitude images.
2.  Calculate R2* map.
3.  Coregister the averaged magnitude to the preprocessed T1w image from sMRIPrep.
4.  Extract the average magnitude image brain by applying the sMRIPrep brain mask.
5.  Warp T1w mask from T1w space into the QSM space by applying the inverse of the coregistration
    transform.
6.  Apply the mask in QSM space to magnitude images.

Notes:

- The R2* map is calculated using the monoexponential fit.
- This must be run after sMRIPrep and process_mese.py.
"""

import argparse
import os
from pprint import pprint

import ants
import nibabel as nb
from bids.layout import BIDSLayout, Query
from nilearn import image

from utils import (
    coregister_to_t1,
    fit_monoexponential,
    get_filename,
    load_config,
    plot_coregistration,
)

CFG = load_config()


def collect_run_data(layout, bids_filters):
    queries = {
        # SWI images from raw BIDS dataset
        'megre_mag': {
            'datatype': 'anat',
            'acquisition': 'QSM',
            'reconstruction': [Query.NONE, Query.ANY],
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
            'reconstruction': [Query.NONE, Query.ANY],
            'part': 'phase',
            'echo': Query.ANY,
            'space': Query.NONE,
            'desc': Query.NONE,
            'suffix': 'MEGRE',
            'extension': ['.nii', '.nii.gz'],
        },
        # T1w-space R2 map from MESE pipeline
        'r2_map': {
            'datatype': 'anat',
            'reconstruction': [Query.NONE, Query.ANY],
            'space': 'T1w',
            'desc': 'MESE',
            'suffix': 'R2map',
            'extension': '.nii.gz',
        },
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
        # sMRIPrep T1w-space brain mask
        't1w_mask': {
            'datatype': 'anat',
            'session': [Query.NONE, Query.ANY],
            'run': [Query.NONE, Query.ANY],
            'reconstruction': [Query.NONE, Query.ANY],
            'space': Query.NONE,
            'res': Query.NONE,
            'desc': 'brain',
            'suffix': 'mask',
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
        # Normalization transform from sMRIPrep
        't1w2mni_xfm': {
            'datatype': 'anat',
            'session': [Query.NONE, Query.ANY],
            'run': [Query.NONE, Query.ANY],
            'reconstruction': [Query.NONE, Query.ANY],
            'from': 'T1w',
            'to': 'MNI152NLin2009cAsym',
            'mode': 'image',
            'suffix': 'xfm',
            'extension': '.h5',
        },
        'mni2t1w_xfm': {
            'datatype': 'anat',
            'session': [Query.NONE, Query.ANY],
            'run': [Query.NONE, Query.ANY],
            'reconstruction': [Query.NONE, Query.ANY],
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
            'reconstruction': [Query.NONE, Query.ANY],
            'space': 'MNI152NLin2009cAsym',
            'suffix': 'dseg',
            'extension': ['.nii', '.nii.gz'],
        },
        # sMRIPrep MNI-space brain mask
        'mni_mask': {
            'datatype': 'anat',
            'session': [Query.NONE, Query.ANY],
            'run': [Query.NONE, Query.ANY],
            'reconstruction': [Query.NONE, Query.ANY],
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
        if key.startswith('megre_'):
            if len(files) != 5:
                raise ValueError(f'Expected 5 files for {key}, got {len(files)}')
            else:
                run_data[key] = sorted([f.path for f in files])
                continue

        elif len(files) != 1:
            raise ValueError(f'Expected 1 file for {key}, got {len(files)} with query {query}')

        file = files[0]
        run_data[key] = file.path

    if len(run_data['megre_mag']) != len(run_data['megre_phase']):
        raise ValueError('Expected same number of magnitude and phase images')

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
    name_source = run_data['megre_mag'][0]

    megre_metadata = [layout.get_metadata(f) for f in run_data['megre_mag']]
    echo_times = [m['EchoTime'] for m in megre_metadata]  # TEs in seconds

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
        interpolator='nearestNeighbor',
    )
    wm_seg_t1w_file = get_filename(
        name_source=wm_seg_file,
        layout=layout,
        out_dir=out_dir,
        entities={'space': 'T1w', 'desc': 'wm', 'suffix': 'mask'},
    )
    ants.image_write(wm_seg_t1w_img, wm_seg_t1w_file)
    del wm_seg_img, wm_seg_t1w_img, wm_seg

    # Average the magnitude images, to use for coregistration
    mean_mag_img = image.mean_img(run_data['megre_mag'])
    mean_mag_filename = get_filename(
        name_source=name_source,
        layout=layout,
        out_dir=out_dir,
        entities={'space': 'MEGRE', 'desc': 'mean', 'suffix': 'MEGRE'},
        dismiss_entities=['echo'],
    )
    mean_mag_img.to_filename(mean_mag_filename)

    # Coregister MEGRE data to preprocessed T1w
    coreg_transform = coregister_to_t1(
        name_source=name_source,
        layout=layout,
        in_file=mean_mag_filename,
        t1_file=run_data['t1w'],
        source_space='MEGRE',
        target_space='T1w',
        out_dir=out_dir,
    )
    # coreg_transform = run_data['megre2t1w_xfm']
    t1_megre_ref_img = ants.apply_transforms(
        fixed=ants.image_read(run_data['t1w']),
        moving=ants.image_read(mean_mag_filename),
        transformlist=[coreg_transform],
        interpolator='lanczosWindowedSinc',
    )
    t1_megre_ref_filename = get_filename(
        name_source=mean_mag_filename,
        layout=layout,
        out_dir=out_dir,
        entities={'space': 'T1w', 'desc': 'mean', 'suffix': 'MEGRE'},
        dismiss_entities=['echo', 'part', 'reconstruction'],
    )
    ants.image_write(t1_megre_ref_img, t1_megre_ref_filename)
    plot_coregistration(
        name_source=t1_megre_ref_filename,
        layout=layout,
        in_file=t1_megre_ref_filename,
        t1_file=run_data['t1w_mni'],
        out_dir=out_dir,
        source_space='MEGRE',
        target_space='MNI152NLin2009cAsym',
        wm_seg=wm_seg_t1w_file,
    )

    mni_megre_ref_img = ants.apply_transforms(
        fixed=ants.image_read(run_data['t1w_mni']),
        moving=ants.image_read(mean_mag_filename),
        transformlist=[run_data['t1w2mni_xfm'], coreg_transform],
        interpolator='lanczosWindowedSinc',
    )
    mni_megre_ref_filename = get_filename(
        name_source=t1_megre_ref_filename,
        layout=layout,
        out_dir=out_dir,
        entities={'space': 'MNI152NLin2009cAsym', 'desc': 'mean', 'suffix': 'MEGRE'},
        dismiss_entities=['echo', 'part', 'reconstruction'],
    )
    ants.image_write(mni_megre_ref_img, mni_megre_ref_filename)
    plot_coregistration(
        name_source=mni_megre_ref_filename,
        layout=layout,
        in_file=mni_megre_ref_filename,
        t1_file=run_data['t1w_mni'],
        out_dir=out_dir,
        source_space='MEGRE',
        target_space='MNI152NLin2009cAsym',
        wm_seg=wm_seg_file,
    )

    # Warp R2 map from T1w space to MEGRE space
    r2_qsm_filename = get_filename(
        name_source=run_data['r2_map'],
        layout=layout,
        out_dir=out_dir,
        entities={'space': 'MEGRE'},
        dismiss_entities=['echo', 'part'],
    )
    r2_qsm_img = ants.apply_transforms(
        fixed=ants.image_read(mean_mag_filename),
        moving=ants.image_read(run_data['r2_map']),
        transformlist=[coreg_transform],
        whichtoinvert=[True],
        interpolator='lanczosWindowedSinc',
    )
    ants.image_write(r2_qsm_img, r2_qsm_filename)

    # Calculate R2* and R2' maps
    _, r2s_hz_img, _, _ = fit_monoexponential(run_data['megre_mag'], echo_times)
    r2s_hz_filename = get_filename(
        name_source=r2_qsm_filename,
        layout=layout,
        out_dir=out_dir,
        entities={'space': 'MEGRE', 'desc': 'MEGRE', 'suffix': 'R2starmap'},
    )
    r2s_hz_img.to_filename(r2s_hz_filename)

    r2prime_hz_filename = get_filename(
        name_source=r2_qsm_filename,
        layout=layout,
        out_dir=out_dir,
        entities={'space': 'MEGRE', 'suffix': 'R2primemap'},
    )
    r2s_hz_img = ants.image_read(r2s_hz_filename)
    r2prime_hz_img = r2s_hz_img - r2_qsm_img
    ants.image_write(r2prime_hz_img, r2prime_hz_filename)

    # Calculate R2* and R2' maps from echo 2 onwards
    _, r2s_hz_img, _, _ = fit_monoexponential(run_data['megre_mag'][1:], echo_times[1:])
    r2s_hz_filename = get_filename(
        name_source=r2_qsm_filename,
        layout=layout,
        out_dir=out_dir,
        entities={'space': 'MEGRE', 'desc': 'MEGRE+E2345', 'suffix': 'R2starmap'},
    )
    r2s_hz_img.to_filename(r2s_hz_filename)

    r2prime_hz_filename = get_filename(
        name_source=r2_qsm_filename,
        layout=layout,
        out_dir=out_dir,
        entities={'space': 'MEGRE', 'desc': 'MEGRE+E2345', 'suffix': 'R2primemap'},
    )
    r2s_hz_img = ants.image_read(r2s_hz_filename)
    r2prime_hz_img = r2s_hz_img - r2_qsm_img
    ants.image_write(r2prime_hz_img, r2prime_hz_filename)

    # Warp brain mask from T1w space to MEGRE space
    mask_qsm_filename = get_filename(
        name_source=name_source,
        layout=layout,
        out_dir=out_dir,
        entities={'space': 'MEGRE', 'desc': 'brain', 'suffix': 'mask'},
    )
    mask_qsm_img = ants.apply_transforms(
        fixed=ants.image_read(mean_mag_filename),
        moving=ants.image_read(run_data['t1w_mask']),
        transformlist=[coreg_transform],
        whichtoinvert=[True],
        interpolator='nearestNeighbor',
    )
    ants.image_write(mask_qsm_img, mask_qsm_filename)


def _get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--subject-id',
        type=lambda label: label.removeprefix('sub-'),
        required=True,
    )
    return parser


def _main(argv=None):
    """Run the process_qsm_prep workflow."""
    options = _get_parser().parse_args(argv)
    kwargs = vars(options)
    main(**kwargs)


def main(subject_id):
    code_dir = CFG['code_dir']
    in_dir = CFG['bids_dir']
    smriprep_dir = CFG['derivatives']['smriprep']
    mese_dir = CFG['derivatives']['mese']
    out_dir = CFG['derivatives']['qsm']
    os.makedirs(out_dir, exist_ok=True)

    layout = BIDSLayout(
        in_dir,
        config=os.path.join(code_dir, 'nibs_bids_config.json'),
        validate=False,
        derivatives=[smriprep_dir, mese_dir, out_dir],
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

    print('DONE!')


if __name__ == '__main__':
    _main()
