"""Calculate T2/R2/S0 maps from MESE data.

This is still just a draft.
I need to calculate SDC from the first echo and apply that to the T2 map.
Plus we need proper output names.

Steps:

1.  Calculate T2 map from AP MESE data.
2.  Calculate distortion map from AP and PA echo-1 data with SDCFlows.
    - topup vs. 3dQwarp vs. something else?
3.  Apply SDC transform to AP echo-1 image.
4.  Coregister SDCed AP echo-1 image to preprocessed T1w from sMRIPrep.
5.  Write out coregistration transform to preprocessed T1w.
6.  Warp T2 map to MNI152NLin2009cAsym (distortion map, coregistration transform,
    normalization transform from sMRIPrep).
7.  Warp S0 map to MNI152NLin2009cAsym.

Notes:

- The T2 map will be used for QSM processing.
- sMRIPrep's preprocessed T1w image is used as the "native T1w space".
- This must be run after sMRIPrep.
"""

import json
import os
import shutil

import ants
from bids.layout import BIDSLayout, Query
# from sdcflows.workflows.fit.pepolar import init_topup_wf

from utils import coregister_to_t1, fit_monoexponential, get_filename


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
        'mese_mag_pa': {
            'part': ['mag', Query.NONE],
            'echo': 1,
            'direction': 'PA',
            'suffix': 'MESE',
            'extension': ['.nii', '.nii.gz'],
        },
        # T1w-space T1w image from sMRIPrep
        't1w': {
            'datatype': 'anat',
            'run': Query.NONE,
            'space': Query.NONE,
            'res': Query.NONE,
            'desc': 'preproc',
            'suffix': 'T1w',
            'extension': ['.nii', '.nii.gz'],
        },
        # sMRIPrep T1w-space brain mask
        't1w_mask': {
            'datatype': 'anat',
            'run': Query.NONE,
            'space': Query.NONE,
            'res': Query.NONE,
            'desc': 'brain',
            'suffix': 'mask',
            'extension': ['.nii', '.nii.gz'],
        },
        # MNI-space T1w image from sMRIPrep
        't1w_mni': {
            'datatype': 'anat',
            'run': Query.NONE,
            'space': 'MNI152NLin2009cAsym',
            'desc': 'preproc',
            'suffix': 'T1w',
            'extension': ['.nii', '.nii.gz'],
        },
        # Normalization transform from sMRIPrep
        't1w2mni_xfm': {
            'datatype': 'anat',
            'run': Query.NONE,
            'from': 'T1w',
            'to': 'MNI152NLin2009cAsym',
            'mode': 'image',
            'suffix': 'xfm',
            'extension': '.h5',
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
        elif len(files) > 1:
            raise ValueError(f'Expected 1 file for {key}, got {len(files)}')
        elif len(files) == 0:
            raise ValueError(f'Expected 1 file for {key}, got {len(files)} with query {query}')
        else:
            files = files[0].path

        run_data[key] = files

    return run_data


def process_run(layout, run_data, out_dir, temp_dir):
    """Process a single run of MESE data.

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
    mese_pa_metadata = layout.get_metadata(run_data['mese_mag_pa'])
    echo_times = [m['EchoTime'] * 1000 for m in mese_ap_metadata]
    t2_img, r2_img, s0_img = fit_monoexponential(
        in_files=run_data['mese_mag_ap'],
        echo_times=echo_times,
    )
    t2_filename = get_filename(
        name_source=name_source,
        layout=layout,
        out_dir=out_dir,
        entities={
            'datatype': 'anat',
            'space': 'MESE',
            'suffix': 'T2map',
            'extension': '.nii.gz',
        },
        dismiss_entities=['echo', 'part'],
    )
    t2_img.to_filename(t2_filename)

    r2_filename = get_filename(
        name_source=name_source,
        layout=layout,
        out_dir=out_dir,
        entities={
            'datatype': 'anat',
            'space': 'MESE',
            'suffix': 'R2map',
            'extension': '.nii.gz',
        },
        dismiss_entities=['echo', 'part'],
    )
    r2_img.to_filename(r2_filename)

    s0_filename = get_filename(
        name_source=name_source,
        layout=layout,
        out_dir=out_dir,
        entities={
            'datatype': 'anat',
            'space': 'MESE',
            'suffix': 'S0map',
            'extension': '.nii.gz',
        },
        dismiss_entities=['echo', 'part'],
    )
    s0_img.to_filename(s0_filename)

    # Calculate distortion map from AP and PA echo-1 data
    mese_mag_ap_echo1 = run_data['mese_mag_ap'][0]
    # mese_mag_pa_echo1 = run_data['mese_mag_pa'][0]
    # topup_wf = init_topup_wf(
    #     grid_reference=0,
    #     use_metadata_estimates=False,
    #     fallback_total_readout_time=None,
    #     omp_nthreads=1,
    #     sloppy=False,
    #     debug=False,
    #     name='topup_estimate_wf',
    # )
    # topup_wf.inputs.in_files = [mese_mag_ap_echo1, mese_mag_pa_echo1]
    # topup_wf.inputs.metadata = [mese_ap_metadata[0], mese_pa_metadata]
    # wf_results = topup_wf.run()
    # output fields are fmap, fmap_ref, fmap_coeff, fmap_mask, jacobians, xfms, out_warps, method
    # fmap_coeff = wf_results.outputs.fmap_coeff
    # fmap_ref = wf_results.outputs.fmap_ref
    # fmap = wf_results.outputs.fmap
    # fmap_mask = wf_results.outputs.fmap_mask
    # jacobians = wf_results.outputs.jacobians
    # xfms = wf_results.outputs.xfms
    # out_warps = wf_results.outputs.out_warps

    """fmap_filename = get_filename(
        name_source=name_source,
        layout=layout,
        out_dir=out_dir,
        entities={
            'datatype': 'fmap',
            'fmapid': 'MESE',
            'desc': 'preproc',
            'suffix': 'fieldmap',
            'extension': '.nii.gz',
        },
        dismiss_entities=['echo', 'part'],
    )
    shutil.copyfile(fmap, fmap_filename)

    fmap_ref_filename = get_filename(
        name_source=name_source,
        layout=layout,
        out_dir=out_dir,
        entities={
            'datatype': 'fmap',
            'fmapid': 'MESE',
            'desc': 'ref',
            'suffix': 'fieldmap',
            'extension': '.nii.gz',
        },
        dismiss_entities=['echo', 'part'],
    )
    shutil.copyfile(fmap_ref, fmap_ref_filename)

    fmap_coeff_filename = get_filename(
        name_source=name_source,
        layout=layout,
        out_dir=out_dir,
        entities={
            'datatype': 'fmap',
            'fmapid': 'MESE',
            'desc': 'coeff',
            'suffix': 'fieldmap',
            'extension': '.nii.gz',
        },
        dismiss_entities=['echo', 'part'],
    )
    shutil.copyfile(fmap_coeff, fmap_coeff_filename)

    fmap_mask_filename = get_filename(
        name_source=name_source,
        layout=layout,
        out_dir=out_dir,
        entities={
            'datatype': 'fmap',
            'fmapid': 'MESE',
            'desc': 'fieldmap',
            'suffix': 'mask',
            'extension': '.nii.gz',
        },
        dismiss_entities=['echo', 'part'],
    )
    shutil.copyfile(fmap_mask, fmap_mask_filename)"""

    # Coregister AP echo-1 data to preprocessed T1w
    # XXX: This is currently using non-SDCed MESE data.
    coreg_transform = coregister_to_t1(
        name_source=name_source,
        layout=layout,
        in_file=mese_mag_ap_echo1,
        t1_file=run_data['t1w'],
        source_space='MESE',
        target_space='T1w',
    )

    # Warp T1w-space T1map and T1w image to MNI152NLin2009cAsym using normalization transform
    # from sMRIPrep and coregistration transform to sMRIPrep's T1w space.
    # XXX: This ignores the SDC transform.
    for file_ in [t2_filename, r2_filename, s0_filename]:
        suffix = os.path.basename(file_).split('_')[1].split('.')[0]
        out_file = get_filename(
            name_source=name_source,
            layout=layout,
            out_dir=out_dir,
            entities={'space': 'MNI152NLin2009cAsym', 'suffix': suffix},
        )
        reg_img = ants.apply_transforms(
            fixed=ants.image_read(run_data['t1w_mni']),
            moving=ants.image_read(file_),
            transformlist=[run_data['t1w2mni_xfm'], coreg_transform],
        )
        ants.image_write(reg_img, out_file)


if __name__ == '__main__':
    # code_dir = '/Users/taylor/Documents/linc/nibs'
    code_dir = '/cbica/projects/nibs/code'
    # in_dir = '/Users/taylor/Documents/datasets/nibs/dset'
    in_dir = '/cbica/projects/nibs/dset'
    # smriprep_dir = '/Users/taylor/Documents/datasets/nibs/derivatives/smriprep'
    smriprep_dir = '/cbica/projects/nibs/derivatives/smriprep'
    # out_dir = '/Users/taylor/Documents/datasets/nibs/derivatives/mese'
    out_dir = '/cbica/projects/nibs/derivatives/mese'
    os.makedirs(out_dir, exist_ok=True)
    # temp_dir = '/Users/taylor/Documents/datasets/nibs/work/mese'
    temp_dir = '/cbica/projects/nibs/work/mese'
    os.makedirs(temp_dir, exist_ok=True)

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
    with open(os.path.join(out_dir, 'dataset_description.json'), 'w') as f:
        json.dump(dataset_description, f, sort_keys=True, indent=4)

    layout = BIDSLayout(
        in_dir,
        config=os.path.join(code_dir, 'nibs_bids_config.json'),
        validate=False,
        derivatives=[smriprep_dir],
    )
    subjects = layout.get_subjects(suffix='MESE')
    for subject in subjects:
        print(f'Processing subject {subject}')
        sessions = layout.get_sessions(subject=subject, suffix='MESE')
        for session in sessions:
            print(f'Processing session {session}')
            mese_files = layout.get(
                subject=subject,
                session=session,
                echo=1,
                part=['mag', Query.NONE],
                direction='AP',
                suffix='MESE',
                extension=['.nii', '.nii.gz'],
            )
            if not mese_files:
                files = layout.get(subject=subject, session=session, suffix='MESE', extension=['.nii', '.nii.gz'])
                print(files[0].get_entities())
                raise ValueError(f'No MESE files found for subject {subject} and session {session}')

            for mese_file in mese_files:
                print(f'Processing MESE file {mese_file.path}')
                entities = mese_file.get_entities()
                entities.pop('echo')
                if 'part' in entities:
                    entities.pop('part')

                entities.pop('direction')
                run_data = collect_run_data(layout, entities)
                process_run(layout, run_data, out_dir, temp_dir)

    print('DONE!')
