"""Calculate T2/R2/S0 maps from MESE data.

This is still just a draft.
I need to calculate SDC from the first echo and apply that to the T2 map.
Plus we need proper output names.

Steps:

1.  Calculate T2 map from AP MESE data.
2.  Calculate distortion map from AP and PA echo-1 data with SDCFlows.
    -   topup vs. 3dQwarp vs. something else?
    -   Currently disabled.
3.  Apply SDC transform to AP echo-1 image.
    - Currently disabled.  This is not needed for the T2 map.
4.  Coregister SDCed AP echo-1 image to preprocessed T1w from sMRIPrep.
    -   Currently using non-SDCed MESE data.
5.  Write out coregistration transform to preprocessed T1w.
6.  Warp T2 map to MNI152NLin2009cAsym (distortion map, coregistration transform,
    normalization transform from sMRIPrep).
7.  Warp S0 map to MNI152NLin2009cAsym.

Notes:

- The T2 map will be used for QSM processing.
- sMRIPrep's preprocessed T1w image is used as the "native T1w space".
- This must be run after sMRIPrep.
"""

import argparse
import json
import os
import shutil
from pprint import pprint

import ants
import nibabel as nb
import numpy as np
from bids.layout import BIDSLayout, Query
from nilearn import masking
from nireports.assembler.report import Report

from utils import (
    fit_monoexponential,
    get_filename,
    plot_coregistration,
    plot_scalar_map,
    run_command,
)

os.environ['SUBJECTS_DIR'] = '/cbica/projects/nibs/derivatives/smriprep/sourcedata/freesurfer'
os.environ['FS_LICENSE'] = '/cbica/projects/nibs/tokens/freesurfer_license.txt'
os.environ['FSLOUTPUTTYPE'] = 'NIFTI_GZ'
CODE_DIR = '/cbica/projects/nibs/code'


def collect_run_data(layout, bids_filters):
    queries = {
        # MESE images from raw BIDS dataset
        'mese_mag_ap': {
            'part': ['mag', Query.NONE],
            'echo': Query.ANY,
            'direction': 'AP',
            'suffix': 'MESE',
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
        # sMRIPrep T1w-space brain mask
        't1w_mask': {
            'datatype': 'anat',
            'session': [Query.NONE, Query.ANY],
            'run': [Query.NONE, Query.ANY],
            'space': Query.NONE,
            'res': Query.NONE,
            'desc': 'brain',
            'suffix': 'mask',
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
        'fsnative2t1w_xfm': {
            'datatype': 'anat',
            'session': [Query.NONE, Query.ANY],
            'run': [Query.NONE, Query.ANY],
            'from': 'fsnative',
            'to': 'T1w',
            'mode': 'image',
            'suffix': 'xfm',
            'extension': '.txt',
        },
    }

    run_data = {}
    for key, query in queries.items():
        query = {**bids_filters, **query}
        files = layout.get(**query)
        if key == 'mese_mag_ap' and len(files) != 4:
            raise ValueError(f'Expected 4 files for {key}, got {len(files)}')
        elif key == 'mese_mag_ap':
            files = [f.path for f in files]
        elif len(files) != 1:
            raise ValueError(f'Expected 1 file for {key}, got {len(files)}: {query}')
        else:
            files = files[0].path

        run_data[key] = files

    pprint(run_data)

    return run_data


def process_run(layout, run_data, out_dir, temp_dir):
    """Process a single run of MESE data.

    TODO: Use SDCFlows to calculate and possibly apply distortion map.

    Parameters
    ----------
    layout : BIDSLayout
        BIDSLayout object for the dataset.
    run_data : dict
        Dictionary containing the paths to the MESE data.
    out_dir : str
        Path to the output directory.
    temp_dir : str
        Path to the temporary directory.
        Not currently used.
    """
    name_source = run_data['mese_mag_ap'][0]
    mese_ap_metadata = [layout.get_metadata(f) for f in run_data['mese_mag_ap']]
    echo_times = [m['EchoTime'] for m in mese_ap_metadata]  # TEs in seconds

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

    # Coregister echoes 2-4 of AP MESE data to echo 1
    hmced_files, brain_mask = iterative_motion_correction(
        name_sources=run_data['mese_mag_ap'],
        layout=layout,
        in_files=run_data['mese_mag_ap'],
        out_dir=out_dir,
        temp_dir=temp_dir,
    )

    mese_mag_ap_echo1 = run_data['mese_mag_ap'][0]

    mese_mag_ap_echo1_t1_file = get_filename(
        name_source=name_source,
        layout=layout,
        out_dir=out_dir,
        entities={'space': 'T1w', 'suffix': 'MESE'},
    )

    t1w_img = ants.image_read(run_data['t1w'])

    # Coregister AP echo-1 data to preprocessed T1w
    mese_to_smriprep_warp_xfm = get_filename(
        name_source=name_source,
        layout=layout,
        out_dir=out_dir,
        entities={
            'from': 'MESEref',
            'to': 'T1w',
            'mode': 'image',
            'suffix': 'xfm',
            'extension': 'nii.gz',
        },
        dismiss_entities=[
            'acquisition',
            'inv',
            'reconstruction',
            'mt',
            'echo',
            'part',
            'direction',
        ],
    )
    mese_to_smriprep_affine_xfm = get_filename(
        name_source=name_source,
        layout=layout,
        out_dir=out_dir,
        entities={
            'from': 'MESEref',
            'to': 'T1w',
            'mode': 'image',
            'suffix': 'xfm',
            'extension': 'mat',
        },
        dismiss_entities=[
            'acquisition',
            'inv',
            'reconstruction',
            'mt',
            'echo',
            'part',
            'direction',
        ],
    )

    xfm_prefix = os.path.join(temp_dir, 'mese_to_t1_syn_')

    args = [
        '--verbose',
        '1',
        '--dimensionality',
        '3',
        '--float',
        '0',
        '--output',
        xfm_prefix,
        '--interpolation',
        'LanczosWindowedSinc',
        '--winsorize-image-intensities',
        '[0.005,0.995]',
        '--initial-moving-transform',
        f'[{run_data["t1w_mask"]},{brain_mask},1]',
        '--masks',
        f'[{run_data["t1w_mask"]},NONE]',
        '--transform',
        'Rigid[0.1]',
        '--metric',
        f'Mattes[{run_data["t1w"]},{mese_mag_ap_echo1},1,32]',
        '--convergence',
        '[50x50x50,1e-8,10]',
        '--shrink-factors',
        '3x2x1',
        '--smoothing-sigmas',
        '1x1x0mm',
        '--transform',
        'SyN[0.2,3,0.5]',
        '--metric',
        f'CC[{run_data["t1w"]},{mese_mag_ap_echo1},1,2]',
        '--convergence',
        '[20x10,1e-6,10]',
        '--shrink-factors',
        '2x1',
        '--smoothing-sigmas',
        '1x0mm',
    ]
    ants.registration(args, None)
    in_mese_to_smriprep_affine_xfm = f'{xfm_prefix}0GenericAffine.mat'
    in_mese_to_smriprep_warp_xfm = f'{xfm_prefix}1Warp.nii.gz'
    assert os.path.isfile(in_mese_to_smriprep_warp_xfm), f'{in_mese_to_smriprep_warp_xfm} not found'
    assert os.path.isfile(in_mese_to_smriprep_affine_xfm), f'{in_mese_to_smriprep_affine_xfm} not found'
    shutil.copyfile(in_mese_to_smriprep_warp_xfm, mese_to_smriprep_warp_xfm)
    shutil.copyfile(in_mese_to_smriprep_affine_xfm, mese_to_smriprep_affine_xfm)

    mese_mag_ap_echo1_t1_img = ants.apply_transforms(
        fixed=ants.image_read(run_data['t1w']),
        moving=ants.image_read(mese_mag_ap_echo1),
        transformlist=[mese_to_smriprep_warp_xfm, mese_to_smriprep_affine_xfm],
        interpolator='lanczosWindowedSinc',
    )
    ants.image_write(mese_mag_ap_echo1_t1_img, mese_mag_ap_echo1_t1_file)

    mese_to_smriprep = [mese_to_smriprep_warp_xfm, mese_to_smriprep_affine_xfm]

    plot_coregistration(
        name_source=mese_mag_ap_echo1_t1_file,
        layout=layout,
        in_file=mese_mag_ap_echo1_t1_file,
        t1_file=run_data['t1w'],
        out_dir=out_dir,
        source_space='MESEref',
        target_space='T1w',
        wm_seg=wm_seg_t1w_file,
    )

    mese_mag_ap_echo1_mni_file = get_filename(
        name_source=name_source,
        layout=layout,
        out_dir=out_dir,
        entities={'space': 'MNI152NLin2009cAsym', 'suffix': 'MESE'},
    )
    mese_mag_ap_echo1_mni_img = ants.apply_transforms(
        fixed=ants.image_read(run_data['t1w_mni']),
        moving=ants.image_read(mese_mag_ap_echo1),
        transformlist=[run_data['t1w2mni_xfm']] + mese_to_smriprep,
        interpolator='lanczosWindowedSinc',
    )
    ants.image_write(mese_mag_ap_echo1_mni_img, mese_mag_ap_echo1_mni_file)
    plot_coregistration(
        name_source=mese_mag_ap_echo1_mni_file,
        layout=layout,
        in_file=mese_mag_ap_echo1_mni_file,
        t1_file=run_data['t1w_mni'],
        out_dir=out_dir,
        source_space='MESEref',
        target_space='MNI152NLin2009cAsym',
        wm_seg=wm_seg_file,
    )

    # Calculate T2 map from AP MESE data
    t2_img, r2_img, s0_img, r_squared_img = fit_monoexponential(
        in_files=hmced_files,
        echo_times=echo_times,
    )

    # Warp T1w-space T1map and T1w image to MNI152NLin2009cAsym using normalization transform
    # from sMRIPrep and coregistration transform to sMRIPrep's T1w space.
    image_types = ['T2map', 'R2map', 'S0map', 'Rsquaredmap']
    images = [t2_img, r2_img, s0_img, r_squared_img]
    for i_file, img in enumerate(images):
        suffix = image_types[i_file]
        file_ = get_filename(
            name_source=name_source,
            layout=layout,
            out_dir=out_dir,
            entities={
                'datatype': 'anat',
                'space': 'MESEref',
                'suffix': suffix,
                'extension': '.nii.gz',
            },
            dismiss_entities=['echo', 'direction'],
        )
        img.to_filename(file_)

        # Warp to T1w space
        t1w_file = get_filename(
            name_source=file_,
            layout=layout,
            out_dir=out_dir,
            entities={'space': 'T1w'},
        )
        t1w_img = ants.apply_transforms(
            fixed=ants.image_read(run_data['t1w']),
            moving=ants.image_read(file_),
            transformlist=mese_to_smriprep,
        )
        ants.image_write(t1w_img, t1w_file)

        # Warp to MNI152NLin2009cAsym space
        mni_file = get_filename(
            name_source=file_,
            layout=layout,
            out_dir=out_dir,
            entities={'space': 'MNI152NLin2009cAsym'},
        )
        mni_img = ants.apply_transforms(
            fixed=ants.image_read(run_data['t1w_mni']),
            moving=ants.image_read(file_),
            transformlist=[run_data['t1w2mni_xfm']] + mese_to_smriprep,
        )
        ants.image_write(mni_img, mni_file)

        # Plot scalar map
        scalar_report = get_filename(
            name_source=mni_file,
            layout=layout,
            out_dir=out_dir,
            entities={'datatype': 'figures', 'desc': 'scalar', 'extension': '.svg'},
        )
        if image_types[i_file] == 'R2map':
            kwargs = {'vmin': 0, 'vmax': 20}
        elif image_types[i_file] == 'Rsquaredmap':
            kwargs = {'vmin': 0, 'vmax': 1}
        else:
            data = masking.apply_mask(mni_file, run_data['mni_mask'])
            vmin = np.percentile(data, 2)
            vmin = np.minimum(vmin, 0)
            vmax = np.percentile(data, 98)
            kwargs = {'vmin': vmin, 'vmax': vmax}

        plot_scalar_map(
            underlay=run_data['t1w_mni'],
            overlay=mni_file,
            mask=run_data['mni_mask'],
            dseg=run_data['dseg_mni'],
            out_file=scalar_report,
            **kwargs,
        )


def iterative_motion_correction(name_sources, layout, in_files, out_dir, temp_dir):
    """Apply iterative motion correction to a list of images.

    Parameters
    ----------
    name_sources : list of str
        List of names of the source files to use for output file names.
    layout : BIDSLayout
        BIDSLayout object.
    in_files : list of str
        List of input image files.
    out_dir : str
        Directory to write output files.
    temp_dir : str
        Directory to write temporary files.

    Returns
    -------
    hmced_files : list of str
        List of paths to the motion-corrected images.
    brain_mask : str
        Path to the brain mask.
    """
    # Step 1: Create a brain mask from the first image with SynthStrip.
    brain_mask = get_filename(
        name_source=in_files[0],
        layout=layout,
        out_dir=out_dir,
        entities={'space': 'MESEref', 'desc': 'brain', 'suffix': 'mask'},
        dismiss_entities=['echo', 'direction'],
    )
    skullstripped_file = os.path.join(temp_dir, f'skullstripped_{os.path.basename(in_files[0])}')
    cmd = (
        'singularity run /cbica/projects/nibs/apptainer/synthstrip-1.7.sif '
        f'-i {in_files[0]} -o {skullstripped_file} -m {brain_mask}'
    )
    run_command(cmd)

    # Step 2: Skull-strip each image.
    skullstripped_files = [skullstripped_file]
    for i_file, in_file in enumerate(in_files):
        if i_file == 0:
            continue

        skullstripped_file = os.path.join(temp_dir, f'skullstripped_{os.path.basename(in_file)}')
        cmd = (
            'singularity run /cbica/projects/nibs/apptainer/synthstrip-1.7.sif '
            f'-i {in_file} -o {skullstripped_file}'
        )
        run_command(cmd)
        skullstripped_files.append(skullstripped_file)

    # Step 3: Register each image to the first image.
    transforms = []
    for i_file, skullstripped_file in enumerate(skullstripped_files):
        in_file = in_files[i_file]
        if i_file == 0:
            ref_img = ants.image_read(skullstripped_file)
            transform = os.path.join(CODE_DIR, 'processing', 'itkIdentityTransform.txt')
        else:
            reg = ants.registration(
                fixed=ref_img,
                moving=ants.image_read(skullstripped_file),
                type_of_transform='Rigid',
            )
            transform = reg['fwdtransforms'][0]

        # Step 3a: Write out individual transforms to MESEref space.
        transform_file = get_filename(
            name_source=name_sources[i_file],
            layout=layout,
            out_dir=out_dir,
            entities={
                'from': 'MESE',
                'to': 'MESEref',
                'mode': 'image',
                'suffix': 'xfm',
                'extension': 'txt' if transform.endswith('.txt') else 'mat',
            },
        )
        shutil.copyfile(transform, transform_file)
        transforms.append(transform_file)

    # Step 4: Write out first image as MESEref.nii.gz
    ref_file = get_filename(
        name_source=name_sources[0],
        layout=layout,
        out_dir=out_dir,
        entities={'suffix': 'MESEref'},
        dismiss_entities=['echo', 'direction'],
    )
    ants.image_write(ref_img, ref_file)

    # Step 5: Apply transforms to original images.
    hmced_files = [in_files[0]]
    for i_file, in_file in enumerate(in_files):
        if i_file == 0:
            plot_coregistration(
                name_source=in_file,
                layout=layout,
                in_file=in_file,
                t1_file=ref_file,
                out_dir=out_dir,
                source_space='MESE',
                target_space='MESEref',
                wm_seg=brain_mask,
            )
            continue

        transform_file = transforms[i_file]
        out_file = get_filename(
            name_source=name_sources[i_file],
            layout=layout,
            out_dir=out_dir,
            entities={'space': 'MESEref'},
        )
        out_img = ants.apply_transforms(
            fixed=ants.image_read(ref_file),
            moving=ants.image_read(in_file),
            transformlist=[transform_file],
            interpolator='lanczosWindowedSinc',
        )
        ants.image_write(out_img, out_file)

        plot_coregistration(
            name_source=out_file,
            layout=layout,
            in_file=out_file,
            t1_file=ref_file,
            out_dir=out_dir,
            source_space='MESE',
            target_space='MESEref',
            wm_seg=brain_mask,
        )
        hmced_files.append(out_file)

    return hmced_files, brain_mask


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
    out_dir = '/cbica/projects/nibs/derivatives/mese'
    os.makedirs(out_dir, exist_ok=True)
    temp_dir = '/cbica/projects/nibs/work/mese'
    os.makedirs(temp_dir, exist_ok=True)

    bootstrap_file = os.path.join(CODE_DIR, 'processing', 'reports_spec_mese.yml')
    assert os.path.isfile(bootstrap_file), f'Bootstrap file {bootstrap_file} not found'

    layout = BIDSLayout(
        in_dir,
        config=os.path.join(CODE_DIR, 'nibs_bids_config.json'),
        validate=False,
        derivatives=[smriprep_dir],
    )

    print(f'Processing subject {subject_id}')
    sessions = layout.get_sessions(subject=subject_id, suffix='MESE')
    for session in sessions:
        print(f'Processing session {session}')
        mese_files = layout.get(
            subject=subject_id,
            session=session,
            echo=1,
            part=['mag', Query.NONE],
            direction='AP',
            suffix='MESE',
            extension=['.nii', '.nii.gz'],
        )
        for mese_file in mese_files:
            print(f'Processing MESE file {mese_file.path}')
            entities = mese_file.get_entities()
            entities.pop('echo')
            if 'part' in entities:
                entities.pop('part')

            entities.pop('direction')
            try:
                run_data = collect_run_data(layout, entities)
            except ValueError as e:
                print(f'Failed {mese_file}')
                print(e)
                continue
            run_temp_dir = os.path.join(temp_dir, os.path.basename(mese_file.path).split('.')[0])
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
            'Name': 'NIBS MESE Derivatives',
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
        with open(dataset_description_file, 'w') as fobj:
            json.dump(dataset_description, fobj, sort_keys=True, indent=4)

    print('DONE!')


if __name__ == "__main__":
    _main()
