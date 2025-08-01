"""Calculate derivatives for ihMTRAGE files.

Steps:

1.  Concatenate ihMTRAGE files into one 4D file (has to be right order).
2.  Motion correct ihMTRAGE files using iterative motion correction algorithm.
3.  Coregister ihMTRAGE reference image to T1w image from sMRIPrep.
4.  Apply motion correction and coregistration transforms to ihMTRAGE files.
5.  Concatenate T1w-space ihMTRAGE files into one 4D file (has to be right order)
6.  Calculate T1w-space ihMT derivatives with ihmt_proc.
7.  Warp T1w-space ihMT derivatives to MNI152NLin2009cAsym using normalization transform from
    sMRIPrep.

Notes:

- sMRIPrep's preprocessed T1w image is used as the "native T1w space".
- This must be run after sMRIPrep and process_mp2rage.py.
"""

import json
import os
import shutil

import ants
import antspynet
import nibabel as nb
from bids.layout import BIDSLayout, Query
from ihmt_proc import cli
from nilearn import image
from nireports.assembler.report import Report

from utils import coregister_to_t1, get_filename, plot_coregistration, plot_scalar_map, run_command

# CODE_DIR = '/Users/taylor/Documents/linc/nibs'
CODE_DIR = '/cbica/projects/nibs/code'


def collect_run_data(layout, bids_filters):
    queries = {
        # ihMTRAGE files from raw BIDS dataset
        'm0': {
            'datatype': 'anat',
            'acquisition': 'nosat',
            'mt': 'off',
            'suffix': 'ihMTRAGE',
            'extension': ['.nii', '.nii.gz'],
        },
        'mtplus': {
            'datatype': 'anat',
            'acquisition': 'singlepos',
            'mt': 'on',
            'suffix': 'ihMTRAGE',
            'extension': ['.nii', '.nii.gz'],
        },
        'mtminus': {
            'datatype': 'anat',
            'acquisition': 'singleneg',
            'mt': 'on',
            'suffix': 'ihMTRAGE',
            'extension': ['.nii', '.nii.gz'],
        },
        'mtdual1': {
            'datatype': 'anat',
            'acquisition': 'dual1',
            'mt': 'on',
            'suffix': 'ihMTRAGE',
            'extension': ['.nii', '.nii.gz'],
        },
        'mtdual2': {
            'datatype': 'anat',
            'acquisition': 'dual2',
            'mt': 'on',
            'suffix': 'ihMTRAGE',
            'extension': ['.nii', '.nii.gz'],
        },
        # sMRIPrep T1w-space B1 map from MP2RAGE derivatives
        'b1map': {
            'datatype': 'fmap',
            'space': 'T1w',
            'suffix': 'TB1map',
            'extension': ['.nii', '.nii.gz'],
        },
        # sMRIPrep T1w-space T1 map from MP2RAGE derivatives
        't1map': {
            'datatype': 'anat',
            'space': 'T1w',
            # 'desc': 'B1corrected',
            'desc': Query.NONE,
            'suffix': 'T1map',
            'extension': ['.nii', '.nii.gz'],
        },
        # sMRIPrep T1w image
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
        if len(files) != 1:
            raise ValueError(f'Expected 1 file for {key}, got {len(files)}: {query}')

        file = files[0]
        run_data[key] = file.path

    return run_data


def process_run(layout, run_data, out_dir, temp_dir):
    name_base = os.path.basename(run_data['m0'])

    # ihMTRAGE parameters
    interpulse_delay = 100  # ms
    n_pulses_per_burst = 1
    n_bursts = 10
    burst_tr = 100  # ms
    burst_tr_final = 100  # ms, MT is unsure
    ihmt_params = f'{interpulse_delay},{n_pulses_per_burst},{n_bursts},{burst_tr},{burst_tr_final}'

    # TB1TFL parameters
    echo_spacing = 4.4  # ms
    flip_angle = 80  # deg
    n_repetitions = 1
    t_r = 9500  # ms
    tfl_params = f'{echo_spacing},{flip_angle},{n_repetitions},{t_r}'

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

    # Concatenate ihMTRAGE files into one 4D file (has to be right order)
    # nosat_mt-off, singlepos_mt-on, dual1_mt-on, singleneg_mt-on, dual2_mt-on
    filetypes = ['m0', 'mtplus', 'mtdual1', 'mtminus', 'mtdual2']
    in_files = [run_data[filetype] for filetype in filetypes]
    concat_ihmt_img = image.concat_imgs([in_files])
    concat_ihmt_file = os.path.join(temp_dir, f'concat_{name_base}')
    concat_ihmt_img.to_filename(concat_ihmt_file)

    denoised_ihmt_file = os.path.join(temp_dir, f'dwidenoise_{name_base}')
    cmd = f'dwidenoise {concat_ihmt_file} {denoised_ihmt_file} -force --extent 3,3,3'
    run_command(cmd)

    denoised_ihmt_imgs = list(image.iter_img(denoised_ihmt_file))
    denoised_ihmt_files = []
    for i_img, img in enumerate(denoised_ihmt_imgs):
        denoised_ihmt_file = os.path.join(temp_dir, f'dwidenoise_{i_img}_{name_base}')
        img.to_filename(denoised_ihmt_file)
        denoised_ihmt_files.append(denoised_ihmt_file)

    # Motion correct ihMTRAGE files
    ihmt_template, hmc_transforms = iterative_motion_correction(
        name_sources=in_files,
        layout=layout,
        in_files=denoised_ihmt_files,
        filetypes=filetypes,
        out_dir=out_dir,
        temp_dir=temp_dir,
    )

    # Coregister ihMTRAGE reference image to sMRIPrep T1w image
    ihmtrage_to_smriprep_xfm = coregister_to_t1(
        name_source=run_data['m0'],
        layout=layout,
        in_file=ihmt_template,
        t1_file=run_data['t1w'],
        out_dir=out_dir,
        source_space='ihMTRAGEref',
        target_space='T1w',
    )

    # Apply motion correction and coregistration to ihMTRAGE files
    ihmt_files_t1space = []
    for i_file, ihmt_file in enumerate(denoised_ihmt_files):
        in_file = in_files[i_file]
        ihmt_file = denoised_ihmt_files[i_file]
        hmc_transform = hmc_transforms[i_file]
        ihmt_img = ants.image_read(ihmt_file)
        ihmt_img_t1space = ants.apply_transforms(
            fixed=ants.image_read(run_data['t1w']),
            moving=ihmt_img,
            transformlist=[ihmtrage_to_smriprep_xfm, hmc_transform],
            interpolator='lanczosWindowedSinc',
        )
        ihmt_file_t1space = get_filename(
            name_source=in_file,
            layout=layout,
            out_dir=out_dir,
            entities={'space': 'T1w'},
        )
        ants.image_write(ihmt_img_t1space, ihmt_file_t1space)
        ihmt_files_t1space.append(ihmt_file_t1space)

    # Calculate ihMTw
    ihmtw_img = image.math_img(
        'mtplus + mtminus - (mtdual1 + mtdual2)',
        mtplus=ihmt_files_t1space[1],
        mtminus=ihmt_files_t1space[3],
        mtdual1=ihmt_files_t1space[2],
        mtdual2=ihmt_files_t1space[4],
    )
    ihmtw_file = get_filename(
        name_source=run_data['m0'],
        layout=layout,
        out_dir=out_dir,
        entities={'space': 'T1w', 'suffix': 'ihMTw'},
        dismiss_entities=['acquisition', 'mt'],
    )
    ihmtw_img.to_filename(ihmtw_file)

    # Calculate ihMTR
    ihmtr_img = image.math_img('ihmt / m0', ihmt=ihmtw_img, m0=ihmt_files_t1space[0])
    ihmtr_file = get_filename(
        name_source=run_data['m0'],
        layout=layout,
        out_dir=out_dir,
        entities={'space': 'T1w', 'suffix': 'ihMTR'},
        dismiss_entities=['acquisition', 'mt'],
    )
    ihmtr_img.to_filename(ihmtr_file)

    # Calculate MTR
    mtr_img = image.math_img(
        '1 - mtplus / m0',
        mtplus=ihmt_files_t1space[1],
        m0=ihmt_files_t1space[0],
    )
    mtr_file = get_filename(
        name_source=run_data['m0'],
        layout=layout,
        out_dir=out_dir,
        entities={'space': 'T1w', 'suffix': 'MTRmap'},
        dismiss_entities=['acquisition', 'mt'],
    )
    mtr_img.to_filename(mtr_file)

    # Concatenate ihMTRAGE files into one 4D file (has to be right order)
    # nosat_mt-off, singlepos_mt-on, dual1_mt-on, singleneg_mt-on, dual2_mt-on
    concat_ihmt_t1space = os.path.join(temp_dir, f'concat_t1space_{name_base}')
    concat_ihmt_img = image.concat_imgs(ihmt_files_t1space)
    concat_ihmt_img.to_filename(concat_ihmt_t1space)

    # Run ihmt_proc to calculate T1w-space ihMT derivatives
    ihmtsat_file = get_filename(
        name_source=run_data['m0'],
        layout=layout,
        out_dir=out_dir,
        entities={'space': 'T1w', 'suffix': 'ihMTsat'},
        dismiss_entities=['acquisition', 'mt'],
    )
    mtdsat_file = get_filename(
        name_source=run_data['m0'],
        layout=layout,
        out_dir=out_dir,
        entities={'space': 'T1w', 'suffix': 'MTdsat'},
        dismiss_entities=['acquisition', 'mt'],
    )
    mtssat_file = get_filename(
        name_source=run_data['m0'],
        layout=layout,
        out_dir=out_dir,
        entities={'space': 'T1w', 'suffix': 'MTssat'},
        dismiss_entities=['acquisition', 'mt'],
    )
    ihmtsatb1sq_file = get_filename(
        name_source=run_data['m0'],
        layout=layout,
        out_dir=out_dir,
        entities={'space': 'T1w', 'suffix': 'ihMTsatB1sq'},
        dismiss_entities=['acquisition', 'mt'],
    )
    mtdsatb1sq_file = get_filename(
        name_source=run_data['m0'],
        layout=layout,
        out_dir=out_dir,
        entities={'space': 'T1w', 'suffix': 'MTdsatB1sq'},
        dismiss_entities=['acquisition', 'mt'],
    )
    mtssatb1sq_file = get_filename(
        name_source=run_data['m0'],
        layout=layout,
        out_dir=out_dir,
        entities={'space': 'T1w', 'suffix': 'MTssatB1sq'},
        dismiss_entities=['acquisition', 'mt'],
    )

    cli.main(
        ihmt=concat_ihmt_t1space,
        t1=run_data['t1map'],
        ihmtparx=ihmt_params,
        tflparx=tfl_params,
        b1=run_data['b1map'],
        ihmtsat_file=ihmtsat_file,
        mtdsat_file=mtdsat_file,
        mtssat_file=mtssat_file,
        ihmtsatb1sq_file=ihmtsatb1sq_file,
        mtdsatb1sq_file=mtdsatb1sq_file,
        mtssatb1sq_file=mtssatb1sq_file,
    )

    # Warp T1w-space ihMT derivatives to MNI152NLin2009cAsym using normalization transform from
    # sMRIPrep
    for file_ in [
        ihmtw_file,
        ihmtr_file,
        mtr_file,
        ihmtsat_file,
        mtdsat_file,
        mtssat_file,
        ihmtsatb1sq_file,
        mtdsatb1sq_file,
        mtssatb1sq_file,
    ]:
        suffix = os.path.basename(file_).split('_')[-1].split('.')[0]
        out_file = get_filename(
            name_source=run_data['m0'],
            layout=layout,
            out_dir=out_dir,
            entities={'space': 'MNI152NLin2009cAsym', 'suffix': suffix},
        )
        reg_img = ants.apply_transforms(
            fixed=ants.image_read(run_data['t1w_mni']),
            moving=ants.image_read(file_),
            transformlist=[run_data['t1w2mni_xfm']],
            interpolator='lanczosWindowedSinc',
        )
        ants.image_write(reg_img, out_file)

        scalar_report = get_filename(
            name_source=out_file,
            layout=layout,
            out_dir=out_dir,
            entities={'datatype': 'figures', 'desc': 'scalar', 'extension': '.svg'},
        )
        plot_scalar_map(
            underlay=run_data['t1w_mni'],
            overlay=out_file,
            mask=run_data['mni_mask'],
            dseg=run_data['dseg_mni'],
            out_file=scalar_report,
        )

        if suffix == 'ihMTw':
            plot_coregistration(
                name_source=file_,
                layout=layout,
                in_file=file_,
                t1_file=run_data['t1w'],
                out_dir=out_dir,
                source_space='ihMTRAGEref',
                target_space='T1w',
                wm_seg=wm_seg_t1w_file,
            )
            plot_coregistration(
                name_source=out_file,
                layout=layout,
                in_file=out_file,
                t1_file=run_data['t1w'],
                out_dir=out_dir,
                source_space='ihMTRAGEref',
                target_space='MNI152NLin2009cAsym',
                wm_seg=wm_seg_file,
            )


def iterative_motion_correction(name_sources, layout, in_files, filetypes, out_dir, temp_dir):
    """Apply iterative motion correction to a list of images.

    This method is based on the method described in https://doi.org/10.1101/2020.09.11.292649.

    Parameters
    ----------
    name_sources : list of str
        List of names of the source files to use for output file names.
    layout : BIDSLayout
        BIDSLayout object.
    in_files : list of str
        List of input image files.
    filetypes : list of str
        List of filetypes of the input images.
    out_dir : str
        Directory to write output files.
    temp_dir : str
        Directory to write temporary files.

    Returns
    -------
    template_file : str
        Path to the template image in ihMTRAGEref space.
    transforms : list of str
        List of transform files.
    """
    # Step 1: Apply N4 bias field correction and skull-stripping to each image.
    brain_n4_files = []
    for i_file, in_file in enumerate(in_files):
        in_img = ants.image_read(in_file)
        # Bias field correction
        n4_img = ants.n4_bias_field_correction(in_img)

        # Skull-stripping
        dseg_img = antspynet.utilities.brain_extraction(n4_img, modality='t1threetissue')
        dseg_img = dseg_img['segmentation_image']
        mask_img = ants.threshold_image(
            dseg_img,
            low_thresh=1,
            high_thresh=1,
            inval=1,
            outval=0,
            binary=True,
        )
        n4_img_masked = n4_img * mask_img
        name_base = os.path.basename(name_sources[i_file])
        brain_n4_file = os.path.join(temp_dir, f'brain_n4_{i_file}_{name_base}')
        ants.image_write(n4_img_masked, brain_n4_file)
        brain_n4_files.append(brain_n4_file)

    # Step 2: Define template image, then register each image to the template.
    # Update the template image with the registered images.
    transforms = []
    for i_file, brain_n4_file in enumerate(brain_n4_files):
        in_file = in_files[i_file]
        filetype = filetypes[i_file]
        if i_file == 0:
            template_img = ants.image_read(brain_n4_file)
            transform = os.path.join(CODE_DIR, 'processing', 'itkIdentityTransform.txt')
        else:
            reg = ants.registration(
                fixed=template_img,
                moving=ants.image_read(brain_n4_file),
                type_of_transform='Rigid',
            )
            transform = reg['fwdtransforms'][0]
            warped_img = reg['warpedmovout']

            # The updated template is the mean of the template and the registered image.
            template_img = (template_img + warped_img) / 2

        # Step 2a: Write out individual transforms to ihMTRAGEref space.
        transform_file = get_filename(
            name_source=name_sources[i_file],
            layout=layout,
            out_dir=out_dir,
            entities={
                'from': filetype,
                'to': 'ihMTRAGEref',
                'mode': 'image',
                'suffix': 'xfm',
                'extension': 'txt' if transform.endswith('.txt') else 'mat',
            },
            dismiss_entities=['acquisition', 'mt'],
        )
        shutil.copyfile(transform, transform_file)
        transforms.append(transform_file)

    # Step 3: Write out template image as ihMTRAGEref.nii.gz
    template_file = get_filename(
        name_source=name_sources[0],
        layout=layout,
        out_dir=out_dir,
        entities={'suffix': 'ihMTRAGEref'},
        dismiss_entities=['acquisition', 'mt'],
    )
    ants.image_write(template_img, template_file)

    # Step 5: Apply transforms to original images.
    for i_file, in_file in enumerate(in_files):
        transform_file = transforms[i_file]
        out_file = get_filename(
            name_source=name_sources[i_file],
            layout=layout,
            out_dir=out_dir,
            entities={'space': 'ihMTRAGEref'},
        )
        out_img = ants.apply_transforms(
            fixed=ants.image_read(template_file),
            moving=ants.image_read(in_file),
            transformlist=[transform_file],
            interpolator='lanczosWindowedSinc',
        )
        ants.image_write(out_img, out_file)

        plot_coregistration(
            name_source=out_file,
            layout=layout,
            in_file=out_file,
            t1_file=template_file,
            out_dir=out_dir,
            source_space=filetypes[i_file],
            target_space='ihMTRAGEref',
        )

    return template_file, transforms


if __name__ == '__main__':
    code_dir = '/cbica/projects/nibs/code'
    in_dir = '/cbica/projects/nibs/dset'
    mp2rage_dir = '/cbica/projects/nibs/derivatives/pymp2rage'
    smriprep_dir = '/cbica/projects/nibs/derivatives/smriprep'
    out_dir = '/cbica/projects/nibs/derivatives/ihmt'
    os.makedirs(out_dir, exist_ok=True)
    temp_dir = '/cbica/projects/nibs/work/ihmt'
    os.makedirs(temp_dir, exist_ok=True)

    bootstrap_file = os.path.join(code_dir, 'processing', 'reports_spec_ihmt.yml')
    assert os.path.isfile(bootstrap_file), f'Bootstrap file {bootstrap_file} not found'

    dataset_description = {
        'Name': 'NIBS',
        'BIDSVersion': '1.10.0',
        'DatasetType': 'derivative',
        'DatasetLinks': {
            'raw': in_dir,
            'mp2rage': mp2rage_dir,
            'smriprep': smriprep_dir,
        },
        'GeneratedBy': [
            {
                'Name': 'Custom code',
                'Description': 'Custom Python code to calculate ihMTw and MTR.',
                'CodeURL': 'https://github.com/PennLINC/nibs',
            }
        ],
    }
    with open(os.path.join(out_dir, 'dataset_description.json'), 'w') as f:
        json.dump(dataset_description, f, sort_keys=True, indent=4)

    layout = BIDSLayout(
        in_dir,
        config=os.path.join(CODE_DIR, 'nibs_bids_config.json'),
        validate=False,
        derivatives=[mp2rage_dir, smriprep_dir],
    )
    subjects = layout.get_subjects(suffix='ihMTRAGE')
    for subject in subjects:
        print(f'Processing subject {subject}')
        sessions = layout.get_sessions(subject=subject, suffix='ihMTRAGE')
        for session in sessions:
            print(f'Processing session {session}')
            m0_files = layout.get(
                subject=subject,
                session=session,
                acquisition='nosat',
                mt='off',
                suffix='ihMTRAGE',
                extension=['.nii', '.nii.gz'],
            )
            for m0_file in m0_files:
                entities = m0_file.get_entities()
                entities.pop('acquisition')
                entities.pop('mt')
                try:
                    run_data = collect_run_data(layout, entities)
                except ValueError as e:
                    print(f'Failed {m0_file}')
                    print(e)
                    continue
                process_run(layout, run_data, out_dir, temp_dir)

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
