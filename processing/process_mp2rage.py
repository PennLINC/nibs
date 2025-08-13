"""Process MP2RAGE data.

Steps:

1.  Rescale B1 famp image into B1 field map.
2.  Register B1 anat image to inv1_magnitude with ANTs.
3.  Write out B1-to-MP2RAGE transform.
4.  Apply B1-to-MP2RAGE transform to B1 field map and anat image.
5.  Write out MP2RAGE-space B1 images.
6.  Calculate T1 map from MP2RAGE images.
7.  Correct T1 map for B1+ inhomogeneity.
8.  Write out T1 map and T1w image in T1-space.
9.  Coregister T1w image to sMRIPrep T1w image.
10. Write out coregistration transform.
11. Write out T1w-space T1map and T1w images.
12. Warp original and B1-corrected T1 maps to MNI152NLin2009cAsym (distortion map,
    coregistration transform, normalization transform from sMRIPrep).

Notes:

- The T1 map will be used for ihMTRAGE processing.
- sMRIPrep's preprocessed T1w image is used as the "native T1w space".
- This must be run after sMRIPrep.
"""

import argparse
import json
import os
import shutil

import ants
import nibabel as nb
import numpy as np
from bids.layout import BIDSLayout, Query
from nilearn import image, masking
from nireports.assembler.report import Report
from pymp2rage import MP2RAGE

from utils import coregister_to_t1, get_filename, plot_coregistration, plot_scalar_map


def collect_run_data(layout, bids_filters):
    queries = {
        # MP2RAGE images from raw BIDS dataset
        'inv1_magnitude': {
            'part': ['mag', Query.NONE],
            'inv': 1,
            'space': Query.NONE,
            'desc': Query.NONE,
            'suffix': 'MP2RAGE',
            'extension': ['.nii', '.nii.gz'],
        },
        'inv1_phase': {
            'part': 'phase',
            'inv': 1,
            'space': Query.NONE,
            'desc': Query.NONE,
            'suffix': 'MP2RAGE',
            'extension': ['.nii', '.nii.gz'],
        },
        'inv2_magnitude': {
            'part': ['mag', Query.NONE],
            'inv': 2,
            'space': Query.NONE,
            'desc': Query.NONE,
            'suffix': 'MP2RAGE',
            'extension': ['.nii', '.nii.gz'],
        },
        'inv2_phase': {
            'part': 'phase',
            'inv': 2,
            'space': Query.NONE,
            'desc': Query.NONE,
            'suffix': 'MP2RAGE',
            'extension': ['.nii', '.nii.gz'],
        },
        # B1 field map from raw BIDS dataset
        'b1_famp': {
            'datatype': 'fmap',
            'acquisition': 'famp',
            'reconstruction': Query.NONE,
            'space': Query.NONE,
            'desc': Query.NONE,
            'suffix': 'TB1TFL',
            'extension': ['.nii', '.nii.gz'],
        },
        'b1_anat': {
            'datatype': 'fmap',
            'acquisition': 'anat',
            'reconstruction': Query.NONE,
            'space': Query.NONE,
            'desc': Query.NONE,
            'suffix': 'TB1TFL',
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
        if 'phase' in key and len(files) == 0:
            print(f'No phase images found for {key}, skipping')
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
    name_source = run_data['inv1_magnitude']
    inv1_metadata = layout.get_metadata(run_data['inv1_magnitude'])
    inv2_metadata = layout.get_metadata(run_data['inv2_magnitude'])
    b1map_metadata = layout.get_metadata(run_data['b1_famp'])

    # Rescale b1_famp to percentage of flip angle
    scalar = b1map_metadata['FlipAngle'] * 10
    # scalar = 90 * 10  # original scalar from Manuel, but I think he had the wrong FA
    b1map_rescaled = image.math_img(f'img / {scalar}', img=run_data['b1_famp'])
    b1map_rescaled_file = os.path.join(temp_dir, os.path.basename(run_data['b1_famp']))
    b1map_rescaled.to_filename(b1map_rescaled_file)

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

    # Register b1_famp to inv1_magnitude using b1_anat with ANTs
    fixed_img = ants.image_read(run_data['inv1_magnitude'])
    reg_output = ants.registration(
        fixed=fixed_img,
        moving=ants.image_read(run_data['b1_anat']),
        type_of_transform='Rigid',
    )
    if len(reg_output['fwdtransforms']) != 1:
        print(
            f'Expected 1 transform, got {len(reg_output["fwdtransforms"])}: '
            f'{reg_output["fwdtransforms"]}'
        )
    b1_to_mp2rage_xfm = reg_output['fwdtransforms'][0]

    # Write the transform to a file
    fwd_transform_file = get_filename(
        name_source=name_source,
        layout=layout,
        out_dir=out_dir,
        entities={
            'datatype': 'fmap',
            'from': 'TB1map',
            'to': 'MP2RAGE',
            'mode': 'image',
            'suffix': 'xfm',
            'extension': '.txt',
        },
        dismiss_entities=['inv', 'part', 'reconstruction'],
    )
    shutil.copyfile(b1_to_mp2rage_xfm, fwd_transform_file)

    # Write the transform to a file
    inv_transform = reg_output['invtransforms'][0]
    inv_transform_file = get_filename(
        name_source=name_source,
        layout=layout,
        out_dir=out_dir,
        entities={
            'datatype': 'fmap',
            'from': 'MP2RAGE',
            'to': 'TB1map',
            'mode': 'image',
            'suffix': 'xfm',
            'extension': '.txt',
        },
        dismiss_entities=['inv', 'part', 'reconstruction'],
    )
    shutil.copyfile(inv_transform, inv_transform_file)

    # Apply the transform to b1_famp
    b1map_rescaled_img = ants.image_read(b1map_rescaled_file)
    b1map_rescaled_reg = ants.apply_transforms(
        fixed=fixed_img,
        moving=b1map_rescaled_img,
        transformlist=b1_to_mp2rage_xfm,
        interpolator='gaussian',
    )
    b1map_rescaled_reg_file = get_filename(
        name_source=name_source,
        layout=layout,
        out_dir=out_dir,
        entities={'datatype': 'fmap', 'space': 'MP2RAGE', 'suffix': 'TB1map'},
        dismiss_entities=['inv', 'part', 'reconstruction'],
    )
    ants.image_write(b1map_rescaled_reg, b1map_rescaled_reg_file)

    # Apply the transform to b1_anat
    b1_anat_img = ants.image_read(run_data['b1_anat'])
    b1_anat_reg = ants.apply_transforms(
        fixed=fixed_img,
        moving=b1_anat_img,
        transformlist=b1_to_mp2rage_xfm,
        interpolator='lanczosWindowedSinc',
    )
    b1_anat_reg_file = get_filename(
        name_source=name_source,
        layout=layout,
        out_dir=out_dir,
        entities={'datatype': 'fmap', 'space': 'MP2RAGE', 'suffix': 'B1anat'},
        dismiss_entities=['inv', 'part', 'reconstruction'],
    )
    ants.image_write(b1_anat_reg, b1_anat_reg_file)

    inversion_times = [
        inv1_metadata['InversionTime'],
        inv2_metadata['InversionTime'],
    ]
    flip_angles = [
        inv1_metadata['FlipAngle'],
        inv2_metadata['FlipAngle'],
    ]
    repetition_times = [
        inv1_metadata['RepetitionTimeExcitation'],
        inv2_metadata['RepetitionTimeExcitation'],
    ]
    n_slices = inv1_metadata['NumberShots']

    mp2rage = MP2RAGE(
        MPRAGE_tr=inv1_metadata['RepetitionTimePreparation'],
        invtimesAB=inversion_times,
        flipangleABdegree=flip_angles,
        B0=inv1_metadata['MagneticFieldStrength'],
        nZslices=n_slices,
        FLASH_tr=repetition_times,
        inv1=run_data['inv1_magnitude'],
        inv2=run_data['inv2_magnitude'],
        inv1ph=run_data.get('inv1_phase', None),
        inv2ph=run_data.get('inv2_phase', None),
    )
    t1map = mp2rage.t1map
    t1map_arr = t1map.get_fdata()
    t1map_arr = t1map_arr / 1000  # Convert from milliseconds to seconds
    t1map = nb.Nifti1Image(t1map_arr, t1map.affine, t1map.header)
    t1map_file = get_filename(
        name_source=name_source,
        layout=layout,
        out_dir=out_dir,
        entities={'suffix': 'T1map'},
        dismiss_entities=['inv', 'part'],
    )
    t1map.to_filename(t1map_file)

    t1w_uni_file = get_filename(
        name_source=name_source,
        layout=layout,
        out_dir=out_dir,
        entities={'suffix': 'T1w'},
        dismiss_entities=['inv', 'part'],
    )
    mp2rage.t1w_uni.to_filename(t1w_uni_file)

    # Correct for B1+ inhomogeneity
    mp2rage.correct_for_B1(b1map_rescaled_reg_file)

    t1map = mp2rage.t1map_b1_corrected
    t1map_arr = t1map.get_fdata()
    t1map_arr = t1map_arr / 1000  # Convert from milliseconds to seconds
    t1map = nb.Nifti1Image(t1map_arr, t1map.affine, t1map.header)
    t1map_b1_corrected_file = get_filename(
        name_source=name_source,
        layout=layout,
        out_dir=out_dir,
        entities={'suffix': 'T1map', 'desc': 'B1corrected'},
        dismiss_entities=['inv', 'part', 'reconstruction'],
    )
    t1map.to_filename(t1map_b1_corrected_file)
    t1w_uni_b1_corrected_file = get_filename(
        name_source=name_source,
        layout=layout,
        out_dir=out_dir,
        entities={'suffix': 'T1w', 'desc': 'B1corrected'},
        dismiss_entities=['inv', 'part', 'reconstruction'],
    )
    mp2rage.t1w_uni_b1_corrected.to_filename(t1w_uni_b1_corrected_file)

    # Coregister MP2RAGE-space T1w image to sMRIPrep T1w image
    mp2rage_to_smriprep_xfm = coregister_to_t1(
        name_source=name_source,
        layout=layout,
        in_file=t1w_uni_b1_corrected_file,
        t1_file=run_data['t1w'],
        source_space='MP2RAGE',
        target_space='T1w',
        out_dir=out_dir,
    )

    # We only want the coregistration figures for the T1w_uni_b1_corrected file
    t1w_t1w_uni_b1_corrected_file = get_filename(
        name_source=t1w_uni_b1_corrected_file,
        layout=layout,
        out_dir=out_dir,
        entities={'space': 'T1w'},
    )
    t1w_t1w_uni_b1_corrected_img = ants.apply_transforms(
        fixed=ants.image_read(run_data['t1w']),
        moving=ants.image_read(t1w_uni_b1_corrected_file),
        transformlist=[mp2rage_to_smriprep_xfm],
        interpolator='lanczosWindowedSinc',
    )
    ants.image_write(t1w_t1w_uni_b1_corrected_img, t1w_t1w_uni_b1_corrected_file)
    plot_coregistration(
        name_source=t1w_t1w_uni_b1_corrected_file,
        layout=layout,
        in_file=t1w_t1w_uni_b1_corrected_file,
        t1_file=run_data['t1w'],
        out_dir=out_dir,
        source_space='MP2RAGE',
        target_space='T1w',
        wm_seg=wm_seg_t1w_file,
    )
    del t1w_t1w_uni_b1_corrected_img, t1w_t1w_uni_b1_corrected_file

    mni_t1w_uni_b1_corrected_file = get_filename(
        name_source=t1w_uni_b1_corrected_file,
        layout=layout,
        out_dir=out_dir,
        entities={'space': 'MNI152NLin2009cAsym'},
    )
    mni_t1w_uni_b1_corrected_img = ants.apply_transforms(
        fixed=ants.image_read(run_data['t1w_mni']),
        moving=ants.image_read(t1w_uni_b1_corrected_file),
        transformlist=[run_data['t1w2mni_xfm'], mp2rage_to_smriprep_xfm],
        interpolator='lanczosWindowedSinc',
    )
    ants.image_write(mni_t1w_uni_b1_corrected_img, mni_t1w_uni_b1_corrected_file)
    plot_coregistration(
        name_source=mni_t1w_uni_b1_corrected_file,
        layout=layout,
        in_file=mni_t1w_uni_b1_corrected_file,
        t1_file=run_data['t1w_mni'],
        out_dir=out_dir,
        source_space='MP2RAGE',
        target_space='MNI152NLin2009cAsym',
        wm_seg=wm_seg_file,
    )
    del mni_t1w_uni_b1_corrected_img, mni_t1w_uni_b1_corrected_file

    # Warp T1w-space T1map and T1w image to MNI152NLin2009cAsym using normalization transform
    # from sMRIPrep and coregistration transform to sMRIPrep's T1w space.
    files = [t1map_file, t1map_b1_corrected_file]
    descs = [None, 'B1corrected']
    for i_file, file_ in enumerate(files):
        desc = descs[i_file]
        suffix = 'T1map'

        t1w_file = get_filename(
            name_source=name_source,
            layout=layout,
            out_dir=out_dir,
            entities={'space': 'T1w', 'suffix': suffix, 'desc': desc},
            dismiss_entities=['inv', 'part', 'reconstruction'],
        )
        t1w_img = ants.apply_transforms(
            fixed=ants.image_read(run_data['t1w']),
            moving=ants.image_read(file_),
            transformlist=[mp2rage_to_smriprep_xfm],
            interpolator='lanczosWindowedSinc',
        )
        ants.image_write(t1w_img, t1w_file)

        mni_file = get_filename(
            name_source=name_source,
            layout=layout,
            out_dir=out_dir,
            entities={'space': 'MNI152NLin2009cAsym', 'suffix': suffix, 'desc': desc},
            dismiss_entities=['inv', 'part', 'reconstruction'],
        )
        mni_img = ants.apply_transforms(
            fixed=ants.image_read(run_data['t1w_mni']),
            moving=ants.image_read(file_),
            transformlist=[run_data['t1w2mni_xfm'], mp2rage_to_smriprep_xfm],
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

    suffixes = ['B1anat', 'TB1map']
    for i_file, file_ in enumerate([run_data['b1_anat'], b1map_rescaled_file]):
        suffix = suffixes[i_file]

        t1w_file = get_filename(
            name_source=name_source,
            layout=layout,
            out_dir=out_dir,
            entities={'datatype': 'fmap', 'space': 'T1w', 'suffix': suffix},
            dismiss_entities=['inv', 'part', 'reconstruction'],
        )
        t1w_img = ants.apply_transforms(
            fixed=ants.image_read(run_data['t1w']),
            moving=ants.image_read(file_),
            transformlist=[mp2rage_to_smriprep_xfm, b1_to_mp2rage_xfm],
            interpolator='gaussian' if suffix == 'TB1map' else 'lanczosWindowedSinc',
        )
        ants.image_write(t1w_img, t1w_file)

        mni_file = get_filename(
            name_source=name_source,
            layout=layout,
            out_dir=out_dir,
            entities={'datatype': 'fmap', 'space': 'MNI152NLin2009cAsym', 'suffix': suffix},
            dismiss_entities=['inv', 'part', 'reconstruction'],
        )
        mni_img = ants.apply_transforms(
            fixed=ants.image_read(run_data['t1w_mni']),
            moving=ants.image_read(file_),
            transformlist=[run_data['t1w2mni_xfm'], mp2rage_to_smriprep_xfm, b1_to_mp2rage_xfm],
            interpolator='gaussian' if suffix == 'TB1map' else 'lanczosWindowedSinc',
        )
        ants.image_write(mni_img, mni_file)

        if suffix == 'B1anat':
            # We only want the coregistration figures for the B1anat file
            plot_coregistration(
                name_source=mni_file,
                layout=layout,
                in_file=mni_file,
                t1_file=run_data['t1w_mni'],
                out_dir=out_dir,
                source_space=suffix,
                target_space='MNI152NLin2009cAsym',
                wm_seg=wm_seg_file,
            )
            plot_coregistration(
                name_source=t1w_file,
                layout=layout,
                in_file=t1w_file,
                t1_file=run_data['t1w'],
                out_dir=out_dir,
                source_space=suffix,
                target_space='T1w',
                wm_seg=wm_seg_t1w_file,
            )
        else:
            scalar_report = get_filename(
                name_source=mni_file,
                layout=layout,
                out_dir=out_dir,
                entities={
                    'datatype': 'figures',
                    'space': 'MNI152NLin2009cAsym',
                    'desc': 'scalar',
                    'extension': '.svg',
                },
            )
            plot_scalar_map(
                underlay=run_data['t1w_mni'],
                overlay=mni_file,
                mask=run_data['mni_mask'],
                dseg=run_data['dseg_mni'],
                out_file=scalar_report,
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
    code_dir = '/cbica/projects/nibs/code'
    in_dir = '/cbica/projects/nibs/dset'
    smriprep_dir = '/cbica/projects/nibs/derivatives/smriprep'
    out_dir = '/cbica/projects/nibs/derivatives/pymp2rage'
    os.makedirs(out_dir, exist_ok=True)
    temp_dir = '/cbica/projects/nibs/work/pymp2rage'
    os.makedirs(temp_dir, exist_ok=True)

    bootstrap_file = os.path.join(code_dir, 'processing', 'reports_spec_mp2rage.yml')
    assert os.path.isfile(bootstrap_file), f'Bootstrap file {bootstrap_file} not found'

    layout = BIDSLayout(
        in_dir,
        config=os.path.join(code_dir, 'nibs_bids_config.json'),
        validate=False,
        derivatives=[smriprep_dir],
    )

    print(f'Processing subject {subject_id}')
    sessions = layout.get_sessions(subject=subject_id, suffix='MP2RAGE')
    for session in sessions:
        print(f'Processing session {session}')
        inv1_magnitude_files = layout.get(
            subject=subject_id,
            session=session,
            inv=1,
            part=['mag', Query.NONE],
            suffix='MP2RAGE',
            extension=['.nii', '.nii.gz'],
        )
        if not inv1_magnitude_files:
            print(f'No inv1 magnitude files found for subject {subject_id} and session {session}')
            continue

        for inv1_magnitude_file in inv1_magnitude_files:
            entities = inv1_magnitude_file.get_entities()
            entities.pop('inv')
            if 'part' in entities:
                entities.pop('part')

            try:
                run_data = collect_run_data(layout, entities)
            except ValueError as e:
                print(f'Failed {inv1_magnitude_file}')
                print(e)
                continue
            fname = os.path.basename(inv1_magnitude_file.path).split('.')[0]
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
            'Name': 'NIBS MP2RAGE Derivatives',
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
