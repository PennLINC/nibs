"""Calculate basic T1w/T2w ratio maps.

Steps:

1.  Coregister SPACE T1w, SPACE T2w, and MPRAGE to sMRIPrep's preprocessed T1w image.
2.  Calculate SPACE T1w/SPACE T2w ratio map.
3.  Calculate MPRAGE T1w/SPACE T2w ratio map.
3.  Warp T1w/T2w ratio maps to MNI152NLin2009cAsym using normalization transform from sMRIPrep.

Notes:

- sMRIPrep's preprocessed T1w image is used as the "native T1w space".
- This must be run after sMRIPrep and process_mp2rage.py.
"""

import json
import os
import shutil

import ants
import nibabel as nb
from bids.layout import BIDSLayout, Query
from nireports.assembler.report import Report

from utils import get_filename, plot_coregistration, plot_scalar_map


def collect_run_data(layout, bids_filters):
    queries = {
        'space_t1w': {
            'part': Query.NONE,
            'acquisition': 'SPACE',
            'suffix': 'T1w',
            'extension': ['.nii', '.nii.gz'],
        },
        'space_t2w': {
            'part': Query.NONE,
            'acquisition': 'SPACE',
            'suffix': 'T2w',
            'extension': ['.nii', '.nii.gz'],
        },
        'mprage_t1w': {
            'part': Query.NONE,
            'acquisition': 'MPRAGE',
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
        # Coregistration transform from MPRAGE T1w to sMRIPrep T1w
        'mprage2t1w_xfm': {
            'datatype': 'anat',
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
        if len(files) != 1:
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

    # Register SPACE T1w to sMRIPrep T1w with ANTs
    fixed_img = ants.image_read(run_data['t1w'])
    moving_img = ants.image_read(run_data['space_t1w'])
    reg_output = ants.registration(
        fixed=fixed_img,
        moving=moving_img,
        type_of_transform='Rigid',
    )
    if len(reg_output['fwdtransforms']) != 1:
        print(
            f'Expected 1 transform, got {len(reg_output["fwdtransforms"])}: '
            f'{reg_output["fwdtransforms"]}'
        )
    fwd_transform = reg_output['fwdtransforms'][0]
    inv_transform = reg_output['invtransforms'][0]
    del moving_img, reg_output

    # Write the transform to a file
    fwd_transform_file = get_filename(
        name_source=run_data['space_t1w'],
        layout=layout,
        out_dir=out_dir,
        entities={
            'from': 'SPACET1w',
            'to': 'T1w',
            'mode': 'image',
            'suffix': 'xfm',
            'extension': '.txt',
        },
        dismiss_entities=['reconstruction', 'acquisition'],
    )
    shutil.copyfile(fwd_transform, fwd_transform_file)

    # Write the inverse transform to a file
    inv_transform_file = get_filename(
        name_source=run_data['space_t1w'],
        layout=layout,
        out_dir=out_dir,
        entities={
            'from': 'T1w',
            'to': 'SPACET1w',
            'mode': 'image',
            'suffix': 'xfm',
            'extension': '.txt',
        },
        dismiss_entities=['reconstruction', 'acquisition'],
    )
    shutil.copyfile(inv_transform, inv_transform_file)
    del fwd_transform_file, inv_transform_file

    # Apply the transform to SPACE T1w
    space_t1w_img = ants.image_read(run_data['space_t1w'])
    t1w_space_t1w_img = ants.apply_transforms(
        fixed=fixed_img,
        moving=space_t1w_img,
        transformlist=fwd_transform,
        interpolator='gaussian',
    )
    t1w_space_t1w_file = get_filename(
        name_source=run_data['space_t1w'],
        layout=layout,
        out_dir=out_dir,
        entities={'space': 'T1w', 'suffix': 'T1w'},
        dismiss_entities=['reconstruction'],
    )
    ants.image_write(t1w_space_t1w_img, t1w_space_t1w_file)
    del space_t1w_img

    # Register SPACE T2w to sMRIPrep T1w with ANTs
    moving_img = ants.image_read(run_data['space_t2w'])
    reg_output = ants.registration(
        fixed=fixed_img,
        moving=moving_img,
        type_of_transform='Rigid',
    )
    if len(reg_output['fwdtransforms']) != 1:
        print(
            f'Expected 1 transform, got {len(reg_output["fwdtransforms"])}: '
            f'{reg_output["fwdtransforms"]}'
        )
    fwd_transform = reg_output['fwdtransforms'][0]
    inv_transform = reg_output['invtransforms'][0]
    del moving_img, reg_output

    # Write the transform to a file
    fwd_transform_file = get_filename(
        name_source=run_data['space_t2w'],
        layout=layout,
        out_dir=out_dir,
        entities={
            'from': 'T1w',
            'to': 'SPACET2w',
            'mode': 'image',
            'suffix': 'xfm',
            'extension': '.txt',
        },
        dismiss_entities=['reconstruction', 'acquisition'],
    )
    shutil.copyfile(fwd_transform, fwd_transform_file)

    inv_transform_file = get_filename(
        name_source=run_data['space_t2w'],
        layout=layout,
        out_dir=out_dir,
        entities={
            'from': 'T1w',
            'to': 'SPACET2w',
            'mode': 'image',
            'suffix': 'xfm',
            'extension': '.txt',
        },
        dismiss_entities=['reconstruction', 'acquisition'],
    )
    shutil.copyfile(inv_transform, inv_transform_file)
    del fwd_transform_file, inv_transform_file

    # Apply the transform to SPACE T2w
    space_t2w_img = ants.image_read(run_data['space_t2w'])
    t1w_space_t2w_img = ants.apply_transforms(
        fixed=fixed_img,
        moving=space_t2w_img,
        transformlist=fwd_transform,
        interpolator='gaussian',
    )
    t1w_space_t2w_file = get_filename(
        name_source=run_data['space_t2w'],
        layout=layout,
        out_dir=out_dir,
        entities={'space': 'T1w', 'suffix': 'T2w'},
        dismiss_entities=['reconstruction'],
    )
    ants.image_write(t1w_space_t2w_img, t1w_space_t2w_file)
    del space_t2w_img

    # Apply the sMRIPrep coregistration transform to MPRAGE T1w
    mprage_t1w_img = ants.image_read(run_data['mprage_t1w'])
    fwd_transform = run_data['mprage2t1w_xfm']
    t1w_mprage_t1w_img = ants.apply_transforms(
        fixed=fixed_img,
        moving=mprage_t1w_img,
        transformlist=fwd_transform,
        interpolator='gaussian',
    )
    t1w_mprage_t1w_file = get_filename(
        name_source=run_data['mprage_t1w'],
        layout=layout,
        out_dir=out_dir,
        entities={'space': 'T1w', 'suffix': 'T1w'},
    )
    ants.image_write(t1w_mprage_t1w_img, t1w_mprage_t1w_file)
    del mprage_t1w_img

    # Plot coregistration of SPACE and MPRAGE files to sMRIPrep T1w
    descs = ['SPACE', 'MPRAGE', 'SPACE']
    for i_file, file_ in enumerate([t1w_space_t1w_file, t1w_mprage_t1w_file, t1w_space_t2w_file]):
        plot_coregistration(
            name_source=file_,
            layout=layout,
            in_file=file_,
            t1_file=run_data['t1w'],
            out_dir=out_dir,
            source_space=descs[i_file],
            target_space='T1w',
            wm_seg=wm_seg_t1w_file,
        )

        mni_file = get_filename(
            name_source=file_,
            layout=layout,
            out_dir=out_dir,
            entities={'space': 'MNI152NLin2009cAsym'},
            dismiss_entities=['reconstruction'],
        )
        mni_img = ants.apply_transforms(
            fixed=ants.image_read(run_data['t1w_mni']),
            moving=ants.image_read(file_),
            transformlist=[run_data['t1w2mni_xfm']],
            interpolator='lanczosWindowedSinc',
        )
        ants.image_write(mni_img, mni_file)

        plot_coregistration(
            name_source=mni_file,
            layout=layout,
            in_file=mni_file,
            t1_file=run_data['t1w_mni'],
            out_dir=out_dir,
            source_space=descs[i_file],
            target_space='MNI152NLin2009cAsym',
            wm_seg=wm_seg_file,
        )
        del mni_img, mni_file

    del t1w_space_t1w_file, t1w_mprage_t1w_file, t1w_space_t2w_file

    # Calculate SPACE T1w/SPACE T2w ratio map
    t1w_space_ratio_file = get_filename(
        name_source=run_data['space_t1w'],
        layout=layout,
        out_dir=out_dir,
        entities={'space': 'T1w', 'desc': 'SPACE', 'suffix': 'myelinw'},
        dismiss_entities=['reconstruction', 'acquisition'],
    )
    t1w_space_ratio_img = t1w_space_t1w_img / t1w_space_t2w_img
    ants.image_write(t1w_space_ratio_img, t1w_space_ratio_file)

    # Calculate MPRAGE T1w/SPACE T2w ratio map
    t1w_mprage_ratio_file = get_filename(
        name_source=run_data['mprage_t1w'],
        layout=layout,
        out_dir=out_dir,
        entities={'space': 'T1w', 'desc': 'MPRAGE', 'suffix': 'myelinw'},
        dismiss_entities=['reconstruction', 'acquisition'],
    )
    t1w_mprage_ratio_img = t1w_mprage_t1w_img / t1w_space_t2w_img
    ants.image_write(t1w_mprage_ratio_img, t1w_mprage_ratio_file)

    # Warp T1w-space SPACE T1w/SPACE T2w and MPRAGE T1w/SPACE T2w ratio maps to MNI152NLin2009cAsym
    # using normalization transform from sMRIPrep.
    files = [t1w_space_ratio_file, t1w_mprage_ratio_file]
    descs = ['SPACE', 'MPRAGE']
    vmaxes = [3, 2]
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
            vmin=0,
            vmax=vmaxes[i_file],
        )


if __name__ == '__main__':
    code_dir = '/cbica/projects/nibs/code'
    in_dir = '/cbica/projects/nibs/dset'
    smriprep_dir = '/cbica/projects/nibs/derivatives/smriprep'
    out_dir = '/cbica/projects/nibs/derivatives/t1wt2w_ratio'
    os.makedirs(out_dir, exist_ok=True)
    temp_dir = '/cbica/projects/nibs/work/t1wt2w_ratio'
    os.makedirs(temp_dir, exist_ok=True)

    bootstrap_file = os.path.join(code_dir, 'processing', 'reports_spec_t1wt2w_ratio.yml')
    assert os.path.isfile(bootstrap_file), f'Bootstrap file {bootstrap_file} not found'

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
    with open(os.path.join(out_dir, 'dataset_description.json'), 'w') as f:
        json.dump(dataset_description, f, sort_keys=True, indent=4)

    layout = BIDSLayout(
        in_dir,
        config=os.path.join(code_dir, 'nibs_bids_config.json'),
        validate=False,
        derivatives=[smriprep_dir],
    )
    subjects = layout.get_subjects(acquisition='SPACE', suffix='T2w')
    for subject in subjects:
        print(f'Processing subject {subject}')
        sessions = layout.get_sessions(subject=subject, acquisition='SPACE', suffix='T2w')
        for session in sessions:
            print(f'Processing session {session}')
            space_t2w_files = layout.get(
                subject=subject,
                session=session,
                acquisition='SPACE',
                suffix='T2w',
                extension=['.nii', '.nii.gz'],
            )
            for space_t2w_file in space_t2w_files:
                entities = space_t2w_file.get_entities()
                entities.pop('acquisition')
                try:
                    run_data = collect_run_data(layout, entities)
                except ValueError as e:
                    print(f'Failed {space_t2w_file}')
                    print(e)
                    continue

                run_temp_dir = os.path.join(
                    temp_dir,
                    os.path.basename(space_t2w_file).replace('.', '_'),
                )
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
