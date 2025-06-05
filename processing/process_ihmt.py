"""Calculate derivatives for ihMTRAGE files."""

import json
import os
import shutil

import ants
from bids.layout import BIDSLayout, Query
from ihmt_proc import cli
from nilearn import image

from utils import run_command, get_filename


def collect_run_data(layout, bids_filters):
    queries = {
        'm0': {
            'acquisition': 'nosat',
            'mt': 'off',
            'suffix': 'ihMTRAGE',
            'extension': ['.nii', '.nii.gz'],
        },
        'mt+': {
            'acquisition': 'singlepos',
            'mt': 'on',
            'suffix': 'ihMTRAGE',
            'extension': ['.nii', '.nii.gz'],
        },
        'mt-': {
            'acquisition': 'singleneg',
            'mt': 'on',
            'suffix': 'ihMTRAGE',
            'extension': ['.nii', '.nii.gz'],
        },
        'mtdual1': {
            'acquisition': 'dual1',
            'mt': 'on',
            'suffix': 'ihMTRAGE',
            'extension': ['.nii', '.nii.gz'],
        },
        'mtdual2': {
            'acquisition': 'dual2',
            'mt': 'on',
            'suffix': 'ihMTRAGE',
            'extension': ['.nii', '.nii.gz'],
        },
        'b1map': {
            'space': 'T1map',
            'suffix': 'B1map',
            'extension': ['.nii', '.nii.gz'],
        },
        't1map': {
            'desc': 'B1corrected',
            'suffix': 'T1map',
            'extension': ['.nii', '.nii.gz'],
        },
    }

    run_data = {}
    for key, query in queries.items():
        query = {**bids_filters, **query}
        files = layout.get(**query)
        if len(files) > 1:
            raise ValueError(f'Expected 1 file for {key}, got {len(files)}: {query}')
        elif len(files) == 0:
            print(f'Expected 1 file for {key}, got {len(files)}: {query}')
            run_data[key] = None
            continue

        file = files[0]
        run_data[key] = file.path

    return run_data


def process_run(name_source, layout, run_data, out_dir, temp_dir):
    name_base = os.path.basename(name_source)

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
    tb1tfl_params = f'{echo_spacing},{flip_angle},{n_repetitions},{t_r}'

    # TODO: Concatenate ihMTRAGE files into one 4D file (has to be right order)
    # nosat_mt-off, singlepos_mt-on, dual1_mt-on, singleneg_mt-on, dual2_mt-on
    in_files = [
        run_data['m0'],
        run_data['mt+'],
        run_data['mtdual1'],
        run_data['mt-'],
        run_data['mtdual2'],
    ]
    concat_ihmt_img = image.concat_imgs([in_files])
    concat_ihmt_file = os.path.join(temp_dir, f'concat_{name_base}')
    concat_ihmt_img.to_filename(concat_ihmt_file)

    dwidenoise_extent = '3,3,3'
    denoised_ihmt_file = os.path.join(temp_dir, f'dwidenoise_{name_base}')
    cmd = f'dwidenoise {concat_ihmt_file} {denoised_ihmt_file} -force --extent {dwidenoise_extent}'
    run_command(cmd)

    denoised_ihmt_imgs = list(image.iter_img(denoised_ihmt_file))
    denoised_ihmt_files = []
    for i_img, img in enumerate(denoised_ihmt_imgs):
        denoised_ihmt_file = os.path.join(temp_dir, f'dwidenoise_{i_img}_{name_base}')
        img.to_filename(denoised_ihmt_file)
        denoised_ihmt_files.append(denoised_ihmt_file)

    # Motion correct ihMTRAGE files
    ihmt_template, hmc_transforms = iterative_motion_correction(
        name_source=name_source,
        layout=layout,
        in_files=denoised_ihmt_files,
        out_dir=out_dir,
        temp_dir=temp_dir,
    )
    coreg_transform = coregister_to_t1(
        name_source=name_source,
        layout=layout,
        in_file=ihmt_template,
        t1_file=run_data['t1map'],
        out_dir=out_dir,
    )

    # Apply motion correction and coregistration to ihMTRAGE files
    ihmt_files_t1space = []
    for i_file, ihmt_file in enumerate(denoised_ihmt_files):
        in_file = in_files[i_file]
        ihmt_file = denoised_ihmt_files[i_file]
        hmc_transform = hmc_transforms[i_file]
        ihmt_img = ants.image_read(ihmt_file)
        ihmt_img_t1space = ants.apply_transforms(
            fixed=ants.image_read(run_data['t1map']),
            moving=ihmt_img,
            transformlist=[coreg_transform, hmc_transform],
            interpolator='gaussian',
        )
        ihmt_file_t1space = get_filename(
            name_source=in_file,
            layout=layout,
            out_dir=out_dir,
            entities={'space': 'T1map'},
            dismiss_entities=['acquisition', 'mt'],
        )
        ants.image_write(ihmt_img_t1space, ihmt_file_t1space)
        ihmt_files_t1space.append(ihmt_file_t1space)

    # TODO: Concatenate ihMTRAGE files into one 4D file (has to be right order)
    # nosat_mt-off, singlepos_mt-on, dual1_mt-on, singleneg_mt-on, dual2_mt-on
    concat_ihmt_t1space = os.path.join(temp_dir, f'concat_t1space_{name_base}')
    concat_ihmt_img = image.concat_imgs(ihmt_files_t1space)
    concat_ihmt_img.to_filename(concat_ihmt_t1space)

    ihmtsat_file = get_filename(
        name_source=name_source,
        layout=layout,
        out_dir=out_dir,
        entities={'space': 'T1map', 'suffix': 'ihMTsat'},
        dismiss_entities=['acquisition', 'mt'],
    )
    mtdsat_file = get_filename(
        name_source=name_source,
        layout=layout,
        out_dir=out_dir,
        entities={'space': 'T1map', 'suffix': 'MTdsat'},
        dismiss_entities=['acquisition', 'mt'],
    )
    mtssat_file = get_filename(
        name_source=name_source,
        layout=layout,
        out_dir=out_dir,
        entities={'space': 'T1map', 'suffix': 'MTssat'},
        dismiss_entities=['acquisition', 'mt'],
    )
    ihmtsatb1sq_file = get_filename(
        name_source=name_source,
        layout=layout,
        out_dir=out_dir,
        entities={'space': 'T1map', 'suffix': 'ihMTsatB1sq'},
        dismiss_entities=['acquisition', 'mt'],
    )
    mtdsatb1sq_file = get_filename(
        name_source=name_source,
        layout=layout,
        out_dir=out_dir,
        entities={'space': 'T1map', 'suffix': 'MTdsatB1sq'},
        dismiss_entities=['acquisition', 'mt'],
    )
    mtssatb1sq_file = get_filename(
        name_source=name_source,
        layout=layout,
        out_dir=out_dir,
        entities={'suffix': 'MTssatB1sq'},
        dismiss_entities=['acquisition', 'mt'],
    )

    cli.main(
        ihmt=concat_ihmt_t1space,
        t1=run_data['t1map'],
        ihmtparx=ihmt_params,
        tb1tflparx=tb1tfl_params,
        b1=run_data['b1map'],
        ihmtsat_file=ihmtsat_file,
        mtdsat_file=mtdsat_file,
        mtssat_file=mtssat_file,
        ihmtsatb1sq_file=ihmtsatb1sq_file,
        mtdsatb1sq_file=mtdsatb1sq_file,
        mtssatb1sq_file=mtssatb1sq_file,
    )


def iterative_motion_correction(name_source, layout, in_files, out_dir, temp_dir):
    """Apply iterative motion correction to a list of images.

    This method is based on the method described in https://doi.org/10.1101/2020.09.11.292649.

    Parameters
    ----------
    name_source : str
        Name of the source file to use for output file names.
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
    template_file : str
        Path to the template image in ihMTRAGEref space.
    transforms : list of str
        List of transform files.
    """
    name_base = os.path.basename(name_source)

    # Step 1: Apply N4 bias field correction and skull-stripping to each image.
    brain_n4_files = []
    for i_file, in_file in enumerate(in_files):
        in_img = ants.image_read(in_file)
        # Bias field correction
        n4_img = ants.n4_bias_field_correction(in_img)

        # Skull-stripping
        # XXX: Need an actual function to skullstrip
        brain_n4_file = os.path.join(temp_dir, f'brain_n4_{i_file}_{name_base}')
        brain_n4_img = ants.skullstrip(n4_img)
        ants.image_write(brain_n4_img, brain_n4_file)
        brain_n4_files.append(brain_n4_file)

    # Step 2: Define template image, then register each image to the template.
    # Update the template image with the registered images.
    transforms = []
    for i_file, brain_n4_file in enumerate(brain_n4_files):
        in_file = in_files[i_file]
        suffix = layout.parse_file_entities(in_file)['suffix']
        if i_file == 0:
            template_img = ants.image_read(brain_n4_file)
            # XXX: Locate identity ITK transform file
            transform = 'identity'  # identity transform
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
            name_source=name_source,
            layout=layout,
            out_dir=out_dir,
            entities={
                'from': suffix,
                'to': 'ihMTRAGEref',
                'mode': 'image',
                'suffix': 'xfm',
                'extension': 'txt',
            },
            dismiss_entities=['acquisition', 'mt'],
        )
        shutil.copyfile(transform, transform_file)
        transforms.append(transform_file)

    # Step 3: Write out template image as ihMTRAGEref.nii.gz
    template_file = get_filename(
        name_source=name_source,
        layout=layout,
        out_dir=out_dir,
        entities={'suffix': 'ihMTRAGEref'},
        dismiss_entities=['acquisition', 'mt'],
    )
    ants.image_write(template_img, template_file)

    # Step 5: Apply transforms to original images.
    for i_file, in_file in enumerate(in_files):
        suffix = layout.parse_file_entities(in_file)['suffix']
        transform_file = transforms[i_file]
        out_file = get_filename(
            name_source=name_source,
            layout=layout,
            out_dir=out_dir,
            entities={'space': 'ihMTRAGEref', 'suffix': suffix},
        )
        out_img = ants.apply_transforms(
            fixed=ants.image_read(template_file),
            moving=ants.image_read(in_file),
            transformlist=[transform_file],
            interpolator='gaussian',
        )
        ants.image_write(out_img, out_file)

    return template_file, transforms


def coregister_to_t1(name_source, layout, in_file, t1_file, out_dir):
    """Coregister an image to a T1w image.

    Parameters
    ----------
    name_source : str
        Name of the source file to use for output file names.
    layout : BIDSLayout
        BIDSLayout object.
    in_file : str
        Path to the input image.
    t1_file : str
        Path to the T1w image.
    out_dir : str
        Directory to write output files.

    Returns
    -------
    transform_file : str
        Path to the transform file.
    """
    # Step 1: Apply N4 bias field correction and skull-stripping to T1.
    t1_img = ants.image_read(t1_file)
    n4_img = ants.n4_bias_field_correction(t1_img)
    brain_n4_img = ants.skullstrip(n4_img)
    ants.image_write(brain_n4_img, os.path.join(out_dir, 'space-T1map_brain.nii.gz'))

    # Step 2: Coregister the brain-extracted image to the T1w image.
    registered_img = ants.registration(
        fixed=ants.image_read(os.path.join(out_dir, 'space-T1map_brain.nii.gz')),
        moving=ants.image_read(in_file),
        type_of_transform='Rigid',
    )
    transform = registered_img['fwdtransforms'][0]
    transform_file = get_filename(
        name_source=name_source,
        layout=layout,
        out_dir=out_dir,
        entities={
            'from': 'ihMTRAGEref',
            'to': 'T1map',
            'mode': 'image',
            'suffix': 'xfm',
            'extension': 'txt',
        },
        dismiss_entities=['acquisition', 'mt'],
    )
    shutil.copyfile(transform, transform_file)
    return transform_file


if __name__ == '__main__':
    code_dir = '/Users/taylor/Documents/linc/nibs'
    # in_dir = "/cbica/projects/nibs/dset"
    in_dir = '/Users/taylor/Documents/datasets/nibs/dset'
    mp2rage_dir = '/Users/taylor/Documents/datasets/nibs/derivatives/pymp2rage'
    # out_dir = "/cbica/projects/nibs/derivatives/ihmt"
    out_dir = '/Users/taylor/Documents/datasets/nibs/derivatives/ihmt'
    os.makedirs(out_dir, exist_ok=True)
    temp_dir = '/Users/taylor/Documents/datasets/nibs/work/ihmt'
    os.makedirs(temp_dir, exist_ok=True)

    dataset_description = {
        'Name': 'NIBS',
        'BIDSVersion': '1.10.0',
        'DatasetType': 'derivative',
        'DatasetLinks': {
            'raw': in_dir,
            'mp2rage': mp2rage_dir,
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
        config=os.path.join(code_dir, 'nibs_bids_config.json'),
        validate=False,
        derivatives=[mp2rage_dir],
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
                run_data = collect_run_data(layout, entities)
                process_run(m0_file, layout, run_data, out_dir, temp_dir)
