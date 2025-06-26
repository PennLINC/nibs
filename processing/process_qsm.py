"""Process QSM data.

Steps:
1.  Average the magnitude images.
2.  Calculate R2* map.
3.  Coregister the averaged magnitude to the preprocessed T1w image from sMRIPrep.
4.  Extract the average magnitude image brain by applying the sMRIPrep brain mask.
5.  Warp T1w mask from T1w space into the QSM space by applying the inverse of the coregistration
    transform.
6.  Apply the mask in QSM space to magnitude images.
7.  Run SEPIA QSM estimation by calling the MATLAB script.
8.  Run chi-separation QSM estimation by calling the MATLAB script.
9.  Warp QSM derivatives to MNI152NLin2009cAsym space.

Notes:

- Remember to name the QSM files with the suffix "Chimap".
- Chimap outputs should be in parts per million (ppm).
- The R2* map is calculated using the monoexponential fit.
- This must be run after sMRIPrep and process_mese.py.
"""
import json
import os
import subprocess
from pprint import pprint

import ants
import nibabel as nb
import numpy as np
from bids.layout import BIDSLayout, Query
from nilearn import image
from nireports.assembler.report import Report

from utils import (
    coregister_to_t1,
    fit_monoexponential,
    get_filename,
    plot_coregistration,
    plot_scalar_map,
)


def collect_run_data(layout, bids_filters):
    queries = {
        # SWI images from raw BIDS dataset
        'megre_mag': {
            'datatype': 'anat',
            'acquisition': 'QSM',
            'part': 'mag',
            'echo': Query.ANY,
            'suffix': 'MEGRE',
            'extension': ['.nii', '.nii.gz'],
        },
        'megre_phase': {
            'datatype': 'anat',
            'acquisition': 'QSM',
            'part': 'phase',
            'echo': Query.ANY,
            'suffix': 'MEGRE',
            'extension': ['.nii', '.nii.gz'],
        },
        # T1w-space R2 map from MESE pipeline
        'r2_map': {
            'datatype': 'anat',
            'space': 'T1w',
            'suffix': 'R2map',
            'extension': '.nii.gz',
        },
        # T1w-space T1w image from sMRIPrep
        't1w': {
            'datatype': 'anat',
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
            'run': [Query.NONE, Query.ANY],
            'space': Query.NONE,
            'res': Query.NONE,
            'desc': 'brain',
            'suffix': 'mask',
            'extension': ['.nii', '.nii.gz'],
        },
        # MNI-space T1w image from sMRIPrep
        't1w_mni': {
            'datatype': 'anat',
            'run': [Query.NONE, Query.ANY],
            'space': 'MNI152NLin2009cAsym',
            'desc': 'preproc',
            'suffix': 'T1w',
            'extension': ['.nii', '.nii.gz'],
        },
        # Normalization transform from sMRIPrep
        't1w2mni_xfm': {
            'datatype': 'anat',
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
            'run': [Query.NONE, Query.ANY],
            'space': 'MNI152NLin2009cAsym',
            'suffix': 'dseg',
            'extension': ['.nii', '.nii.gz'],
        },
        # sMRIPrep MNI-space brain mask
        'mni_mask': {
            'datatype': 'anat',
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


def process_run(layout, run_data, out_dir, temp_dir):
    """Process a single run of QSM data.

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
    """
    name_source = run_data['megre_mag'][0]

    # Calculate T2*, R2*, and S0 maps
    # NOTE: layout.get_metadata only works on full paths
    megre_metadata = [layout.get_metadata(f) for f in run_data['megre_mag']]
    echo_times = [m['EchoTime'] for m in megre_metadata]  # TEs in seconds
    t2s_img, r2s_img, s0_img, rsquared_img = fit_monoexponential(
        in_files=run_data['megre_mag'],
        echo_times=echo_times,
    )
    t2s_filename = get_filename(
        name_source=name_source,
        layout=layout,
        out_dir=out_dir,
        entities={
            'datatype': 'anat',
            'space': 'MEGRE',
            'suffix': 'T2starmap',
            'extension': '.nii.gz',
        },
        dismiss_entities=['echo', 'part'],
    )
    t2s_img.to_filename(t2s_filename)

    r2s_filename = get_filename(
        name_source=name_source,
        layout=layout,
        out_dir=out_dir,
        entities={
            'datatype': 'anat',
            'space': 'MEGRE',
            'suffix': 'R2starmap',
            'extension': '.nii.gz',
        },
        dismiss_entities=['echo', 'part'],
    )
    r2s_img.to_filename(r2s_filename)
    r2s_img = ants.image_read(r2s_filename)

    s0_filename = get_filename(
        name_source=name_source,
        layout=layout,
        out_dir=out_dir,
        entities={
            'datatype': 'anat',
            'space': 'MEGRE',
            'suffix': 'S0map',
            'extension': '.nii.gz',
        },
        dismiss_entities=['echo', 'part'],
    )
    s0_img.to_filename(s0_filename)

    rsquared_filename = get_filename(
        name_source=name_source,
        layout=layout,
        out_dir=out_dir,
        entities={'space': 'MEGRE', 'suffix': 'Rsquaredmap'},
        dismiss_entities=['echo', 'part'],
    )
    rsquared_img.to_filename(rsquared_filename)

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

    # Warp R2 map from T1w space to MEGRE space
    r2_qsm_filename = get_filename(
        name_source=name_source,
        layout=layout,
        out_dir=out_dir,
        entities={'space': 'MEGRE', 'suffix': 'R2map'},
        dismiss_entities=['echo', 'part'],
    )
    r2_qsm_img = ants.apply_transforms(
        fixed=ants.image_read(mean_mag_filename),
        moving=ants.image_read(run_data['r2_map']),
        transformlist=[coreg_transform],
        whichtoinvert=[True],
    )
    ants.image_write(r2_qsm_img, r2_qsm_filename)

    # Calculate R2' (R2 - R2*)
    # R2' is used in chi-separation QSM estimation.
    r2_prime_filename = get_filename(
        name_source=name_source,
        layout=layout,
        out_dir=out_dir,
        entities={'space': 'MEGRE', 'suffix': 'R2primemap'},
        dismiss_entities=['echo', 'part'],
    )
    r2_prime_img = r2s_img - r2_qsm_img
    ants.image_write(r2_prime_img, r2_prime_filename)

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

    # Warp T1w-space T2*map, R2*map, and S0map to MNI152NLin2009cAsym using normalization
    # transform from sMRIPrep and coregistration transform to sMRIPrep's T1w space.
    for file_ in [t2s_filename, r2s_filename, s0_filename, rsquared_filename, r2_prime_filename]:
        suffix = os.path.basename(file_).split('_')[-1].split('.')[0]
        mni_file = get_filename(
            name_source=name_source,
            layout=layout,
            out_dir=out_dir,
            entities={'space': 'MNI152NLin2009cAsym', 'suffix': suffix},
            dismiss_entities=['echo', 'part'],
        )
        reg_img = ants.apply_transforms(
            fixed=ants.image_read(run_data['t1w_mni']),
            moving=ants.image_read(file_),
            transformlist=[run_data['t1w2mni_xfm'], coreg_transform],
        )
        ants.image_write(reg_img, mni_file)

        plot_coregistration(
            name_source=mni_file,
            layout=layout,
            in_file=mni_file,
            t1_file=run_data['t1w_mni'],
            out_dir=out_dir,
            source_space=suffix,
            target_space='MNI152NLin2009cAsym',
        )

        scalar_report = get_filename(
            name_source=mni_file,
            layout=layout,
            out_dir=out_dir,
            entities={'datatype': 'figures', 'desc': 'scalar', 'extension': '.svg'},
        )
        if suffix == 'T2starmap':
            kwargs = {'vmin': 0, 'vmax': 0.08}
        elif suffix == 'R2starmap':
            kwargs = {'vmin': 0, 'vmax': 50}
        elif suffix == 'S0map':
            kwargs = {}
        elif suffix == 'Rsquaredmap':
            kwargs = {'vmin': 0, 'vmax': 1}
        elif suffix == 'R2primemap':
            kwargs = {'vmin': 0, 'vmax': 30}

        plot_scalar_map(
            underlay=run_data['t1w_mni'],
            overlay=mni_file,
            mask=run_data['mni_mask'],
            dseg=run_data['dseg_mni'],
            out_file=scalar_report,
            **kwargs,
        )

    # Create MATLAB-compatible NIfTIs for QSM
    # We will explicitly set the slope and intercept to 1 and 0 to avoid issues with matlab nifti
    # tools and write out uncompressed nifti files.
    mask_qsm_img = nb.load(mask_qsm_filename)
    mask_qsm_img.header.set_slope_inter(1, 0)
    mask_qsm_img.set_data_dtype(np.uint8)
    matlab_mask_filename = os.path.join(temp_dir, 'python_mask.nii')
    mask_qsm_img.to_filename(matlab_mask_filename)

    # Concatenate MEGRE images across echoes
    mag_img = image.concat_imgs(run_data['megre_mag'])
    phase_img = image.concat_imgs(run_data['megre_phase'])
    # Explicitly set slope and intercept to 1 and 0 to avoid issues with matlab nifti tools.
    mag_img.header.set_slope_inter(1, 0)
    mag_img.set_data_dtype(np.float32)
    phase_img.header.set_slope_inter(1, 0)
    phase_img.set_data_dtype(np.int16)
    matlab_mag_filename = os.path.join(temp_dir, 'python_mag.nii')
    matlab_phase_filename = os.path.join(temp_dir, 'python_phase.nii')
    mag_img.to_filename(matlab_mag_filename)
    phase_img.to_filename(matlab_phase_filename)

    # Run SEPIA QSM estimation
    sepia_dir = os.path.join(temp_dir, 'sepia')
    os.makedirs(sepia_dir, exist_ok=True)
    sepia_script = os.path.join(code_dir, 'processing', 'process_qsm_sepia.m')
    with open(sepia_script, 'r') as fobj:
        base_sepia_script = fobj.read()

    modified_sepia_script = (
        base_sepia_script.replace("{{ phase_file }}", matlab_phase_filename)
        .replace("{{ mag_file }}", matlab_mag_filename)
        .replace("{{ mask_file }}", matlab_mask_filename)
        .replace("{{ output_dir }}", sepia_dir)
    )

    out_sepia_script = os.path.join(temp_dir, 'process_qsm_sepia.m')
    with open(out_sepia_script, "w") as fobj:
        fobj.write(modified_sepia_script)

    subprocess.run(
        [
            "matlab",
            "-nodisplay",
            "-nosplash",
            "-nodesktop",
            "-r",
            f"run('{out_sepia_script}');",
            "exit;",
        ],
    )

    return

    # Prepare for chi-separation QSM estimation
    # The chi-separation QSM estimation can use R2* and R2' maps.
    r2s_img = nb.load(r2s_filename)
    r2s_img.header.set_slope_inter(1, 0)
    matlab_r2s_filename = os.path.join(temp_dir, 'python_r2s.nii')
    r2s_img.to_filename(r2s_filename)

    r2_prime_img = nb.load(r2_prime_filename)
    r2_prime_img.header.set_slope_inter(1, 0)
    matlab_r2p_filename = os.path.join(temp_dir, 'python_r2p.nii')
    r2_prime_img.to_filename(matlab_r2p_filename)

    # Now run the chi-separation QSM estimation with R2' map
    chisep_r2p_dir = os.path.join(temp_dir, 'chisep_r2p', 'chisep_output')
    os.makedirs(chisep_r2p_dir, exist_ok=True)
    chisep_script = os.path.join(code_dir, 'processing', 'process_qsm_chisep.m')
    with open(chisep_script, 'r') as fobj:
        base_chisep_script = fobj.read()

    modified_chisep_script = (
        base_chisep_script.replace("{{ mag_file }}", matlab_mag_filename)
        .replace("{{ phase_file }}", matlab_phase_filename)
        .replace("{{ mask_file }}", matlab_mask_filename)
        .replace("{{ r2s_file }}", matlab_r2s_filename)
        .replace("{{ r2p_file }}", matlab_r2p_filename)
        .replace("{{ output_dir }}", os.path.join(temp_dir, 'chisep_r2p'))
    )

    out_chisep_script = os.path.join(temp_dir, 'process_qsm_chisep_r2p.m')
    with open(out_chisep_script, "w") as fobj:
        fobj.write(modified_chisep_script)

    subprocess.run(
        [
            "matlab",
            "-nodisplay",
            "-nosplash",
            "-nodesktop",
            "-r",
            f"run('{out_chisep_script}');",
            "exit;",
        ],
    )

    # Run X-separation QSM estimation with R2' map
    chisep_no_r_dir = os.path.join(temp_dir, 'chisep_no_r2p', 'chisep_output')
    os.makedirs(chisep_no_r_dir, exist_ok=True)
    chisep_script = os.path.join(code_dir, 'processing', 'process_qsm_chisep.m')
    with open(chisep_script, 'r') as fobj:
        base_chisep_script = fobj.read()

    modified_chisep_script = (
        base_chisep_script.replace("{{ mag_file }}", matlab_mag_filename)
        .replace("{{ phase_file }}", matlab_phase_filename)
        .replace("{{ mask_file }}", matlab_mask_filename)
        .replace("{{ r2s_file }}", matlab_r2s_filename)
        .replace("{{ r2p_file }}", 'None')
        .replace("{{ output_dir }}", os.path.join(temp_dir, 'chisep_no_r2p'))
    )

    out_chisep_script = os.path.join(temp_dir, 'process_qsm_chisep_no_r2p.m')
    with open(out_chisep_script, "w") as fobj:
        fobj.write(modified_chisep_script)

    subprocess.run(
        [
            "matlab",
            "-nodisplay",
            "-nosplash",
            "-nodesktop",
            "-r",
            f"run('{out_chisep_script}');",
            "exit;",
        ],
    )


if __name__ == '__main__':
    code_dir = '/cbica/projects/nibs/code'
    in_dir = '/cbica/projects/nibs/dset'
    smriprep_dir = '/cbica/projects/nibs/derivatives/smriprep'
    mese_dir = '/cbica/projects/nibs/derivatives/mese'
    out_dir = '/cbica/projects/nibs/derivatives/qsm'
    os.makedirs(out_dir, exist_ok=True)
    temp_dir = '/cbica/projects/nibs/work/qsm'
    os.makedirs(temp_dir, exist_ok=True)

    bootstrap_file = os.path.join(code_dir, 'processing', 'reports_spec_qsm.yml')
    assert os.path.isfile(bootstrap_file), f'Bootstrap file {bootstrap_file} not found'

    dataset_description = {
        'Name': 'NIBS QSM Derivatives',
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
    with open(os.path.join(out_dir, 'dataset_description.json'), 'w') as f:
        json.dump(dataset_description, f, sort_keys=True, indent=4)

    layout = BIDSLayout(
        in_dir,
        config=os.path.join(code_dir, 'nibs_bids_config.json'),
        validate=False,
        derivatives=[smriprep_dir, mese_dir],
    )
    subjects = layout.get_subjects(suffix='MEGRE')
    # PILOT02 has MEGRE but not MESE, so we skip it.
    subjects = ['PILOT03', 'PILOT04']
    for subject in subjects:
        print(f'Processing subject {subject}')
        sessions = layout.get_sessions(subject=subject, suffix='MEGRE')
        for session in sessions:
            print(f'Processing session {session}')
            megre_files = layout.get(
                subject=subject,
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
                run_data = collect_run_data(layout, entities)
                run_temp_dir = os.path.join(temp_dir, os.path.basename(megre_file.path).split('.')[0])
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
