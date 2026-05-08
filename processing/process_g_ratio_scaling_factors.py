"""Calculate the scaling factors for the MTsat and ihMTR measures to achieve a mean g-ratio of
0.7 in the splenium.

The g-ratio formula is

g-ratio = sqrt(FVF / (FVF + (MVF * scaling_factor)))

where MVF is the MTsat or ihMTR value. FVF is held constant, so we need to solve for the scaling factor.

The equation is solved using the mean g-ratio in the splenium across subjects.
I need to solve for scaling_factor so that g = 0.7.
The first step is to calculate the splenium mask,
then calculate mean FVF and MVF in the splenium across subjects.
"""

from __future__ import annotations

import os
from pprint import pformat

import ants
import numpy as np
import pandas as pd
from bids.layout import BIDSLayout, Query
from nilearn import masking, plotting

from utils import coregister_to_t1, get_filename, load_config

CFG = load_config()
CODE_DIR = CFG['code_dir']


def collect_run_data(layout: object, bids_filters: dict, smriprep_dir: str) -> dict[str, str]:
    """Collect ISOVF, ICVF, and myelin-sensitive maps for scaling factor calibration.

    Parameters
    ----------
    layout : bids.BIDSLayout
        BIDSLayout indexing the dataset and derivatives.
    bids_filters : dict
        BIDS entity filters (e.g., subject, session, run) to narrow the query.
    smriprep_dir : str
        Path to the sMRIPrep derivatives directory.

    Returns
    -------
    run_data : dict
        Mapping of descriptive keys to resolved file paths.
    """
    queries = {
        # T1w-space MTsat and ihMTR maps from process_ihmt.py
        'mtsat_t1w': {
            'datatype': 'anat',
            'run': [Query.NONE, Query.ANY],
            'reconstruction': [Query.NONE, Query.ANY],
            'space': 'T1w',
            'suffix': 'ihMTsatB1sq',
            'extension': ['.nii', '.nii.gz'],
        },
        'ihmtr_t1w': {
            'datatype': 'anat',
            'run': [Query.NONE, Query.ANY],
            'reconstruction': [Query.NONE, Query.ANY],
            'space': 'T1w',
            'suffix': 'ihMTR',
            'extension': ['.nii', '.nii.gz'],
        },
        # sMRIPrep preprocessed T1w in native space (ACPC registration target)
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
        # ACPC-space ISOVF and ICVF maps from QSIRecon
        'isovf_acpc': {
            'datatype': 'dwi',
            'run': [Query.NONE, Query.ANY],
            'reconstruction': [Query.NONE, Query.ANY],
            'space': 'ACPC',
            'model': 'noddi',
            'param': 'isovf',
            'desc': Query.NONE,
            'suffix': 'dwimap',
            'extension': ['.nii', '.nii.gz'],
        },
        'icvf_acpc': {
            'datatype': 'dwi',
            'run': [Query.NONE, Query.ANY],
            'reconstruction': [Query.NONE, Query.ANY],
            'space': 'ACPC',
            'model': 'noddi',
            'param': 'icvf',
            'desc': Query.NONE,
            'suffix': 'dwimap',
            'extension': ['.nii', '.nii.gz'],
        },
        # ACPC-space T1w from QSIPrep (used to compute the ACPC-to-T1w rigid transform)
        't1w_acpc': {
            'datatype': 'anat',
            'session': [Query.NONE, Query.ANY],
            'run': [Query.NONE, Query.ANY],
            'reconstruction': [Query.NONE, Query.ANY],
            'space': 'ACPC',
            'desc': 'preproc',
            'suffix': 'T1w',
            'extension': ['.nii', '.nii.gz'],
        },
        # Transform from Freesurfer to sMRIPrep T1w space
        'fs2t1w_xfm': {
            'datatype': 'anat',
            'session': [Query.NONE, Query.ANY],
            'space': Query.NONE,
            'reconstruction': [Query.NONE, Query.ANY],
            'run': [Query.NONE, Query.ANY],
            'to': 'T1w',
            'from': 'fsnative',
            'mode': 'image',
            'suffix': 'xfm',
            'extension': '.txt',
        },
    }

    run_data = {}
    for key, query in queries.items():
        # I have no clue why, but BIDSLayout refuses to index 'param'
        param = None
        if 'param' in query:
            param = query.pop('param')

        query = {**bids_filters, **query}
        files = layout.get(**query)
        if param is not None:
            files = [f for f in files if f'_param-{param}_' in f.filename]
            if param == 'fa':
                # Both DIPYDKI and DSIStudio have 'fa' as a param. Use DIPYDKI.
                files = [f for f in files if 'qsirecon-DIPYDKI' in f.path]
            query['param'] = param

        if len(files) != 1:
            raise ValueError(f'Expected 1 file for {key}, got {len(files)}: {query}')

        file = files[0]
        run_data[key] = file.path

    run_data['aseg_fsnative'] = os.path.join(
        smriprep_dir,
        'sourcedata',
        'freesurfer',
        f'sub-{bids_filters["subject"]}',
        'mri',
        'aseg.mgz',
    )
    assert os.path.isfile(run_data['aseg_fsnative']), (
        f'Aseg file {run_data["aseg_fsnative"]} not found'
    )

    run_data['brain_fsnative'] = os.path.join(
        smriprep_dir,
        'sourcedata',
        'freesurfer',
        f'sub-{bids_filters["subject"]}',
        'mri',
        'brain.mgz',
    )
    assert os.path.isfile(run_data['brain_fsnative']), (
        f'Brain file {run_data["brain_fsnative"]} not found'
    )

    print(f"Collected run data:\n{pformat(run_data, indent=4)}", flush=True)
    return run_data


def process_run(layout, run_data, out_dir, temp_dir, bids_filters):
    """Calculate mean splenium values for g-ratio scaling factor estimation.

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
    bids_filters : dict
        BIDS entities for the current run.

    Returns
    -------
    splenium_values : pandas.Series
        Mean values of ISOVF, ICVF, MTsat, and ihMTR in the splenium.
    """
    # Register ACPC-space T1w (QSIPrep) to native T1w (sMRIPrep) and use that transform
    # to warp ISOVF and ICVF into T1w space, writing the results for downstream use.
    acpc2t1w_xfm = coregister_to_t1(
        name_source=run_data['t1w_acpc'],
        layout=layout,
        in_file=run_data['t1w_acpc'],
        t1_file=run_data['t1w'],
        out_dir=out_dir,
        source_space='ACPC',
        target_space='T1w',
    )

    t1w_img = ants.image_read(run_data['t1w'])

    isovf_t1w_img = ants.apply_transforms(
        fixed=t1w_img,
        moving=ants.image_read(run_data['isovf_acpc']),
        transformlist=[acpc2t1w_xfm],
        interpolator='nearestNeighbor',
    )
    isovf_t1w_file = get_filename(
        name_source=run_data['isovf_acpc'],
        layout=layout,
        out_dir=out_dir,
        entities={'space': 'T1w'},
    )
    ants.image_write(isovf_t1w_img, isovf_t1w_file)

    icvf_t1w_img = ants.apply_transforms(
        fixed=t1w_img,
        moving=ants.image_read(run_data['icvf_acpc']),
        transformlist=[acpc2t1w_xfm],
        interpolator='nearestNeighbor',
    )
    icvf_t1w_file = get_filename(
        name_source=run_data['icvf_acpc'],
        layout=layout,
        out_dir=out_dir,
        entities={'space': 'T1w'},
    )
    ants.image_write(icvf_t1w_img, icvf_t1w_file)

    # Select only the splenium voxels and warp to T1w space at DWI resolution
    splenium_mask = ants.image_read(run_data['aseg_fsnative']) == 251
    splenium_mask_t1w = ants.apply_transforms(
        fixed=isovf_t1w_img,
        moving=splenium_mask,
        transformlist=[run_data['fs2t1w_xfm']],
        interpolator='nearestNeighbor',
    )
    splenium_mask_file = os.path.join(temp_dir, 'splenium_mask_t1w_dwires.nii.gz')
    ants.image_write(splenium_mask_t1w, splenium_mask_file)

    # Resample T1w-space MTsat and ihMTR to DWI resolution
    mtsat_t1w = ants.image_read(run_data['mtsat_t1w']).resample_image_to_target(
        isovf_t1w_img, interp_type='nearestNeighbor'
    )
    mtsat_t1w_file = os.path.join(temp_dir, 'mtsat_t1w_dwires.nii.gz')
    ants.image_write(mtsat_t1w, mtsat_t1w_file)
    ihmtr_t1w = ants.image_read(run_data['ihmtr_t1w']).resample_image_to_target(
        isovf_t1w_img, interp_type='nearestNeighbor'
    )
    ihmtr_t1w_file = os.path.join(temp_dir, 'ihmtr_t1w_dwires.nii.gz')
    ants.image_write(ihmtr_t1w, ihmtr_t1w_file)

    # Get the data in the splenium
    isovf_splenium = masking.apply_mask(isovf_t1w_file, splenium_mask_file)
    icvf_splenium = masking.apply_mask(icvf_t1w_file, splenium_mask_file)
    mtsat_splenium = masking.apply_mask(mtsat_t1w_file, splenium_mask_file)
    ihmtr_splenium = masking.apply_mask(ihmtr_t1w_file, splenium_mask_file)

    # Plot the splenium mask on top of the T1w brain
    brain_img = ants.image_read(run_data['brain_fsnative'])
    brain_img_t1w = ants.apply_transforms(
        fixed=isovf_t1w_img,
        moving=brain_img,
        transformlist=[run_data['fs2t1w_xfm']],
        interpolator='nearestNeighbor',
    )
    ants.image_write(brain_img_t1w, os.path.join(temp_dir, 'brain_t1w_dwires.nii.gz'))

    splenium_plot = get_filename(
        name_source=isovf_t1w_file,
        layout=layout,
        out_dir=out_dir,
        entities={'datatype': 'figures', 'desc': 'splenium', 'suffix': 'mask', 'extension': '.svg'},
        dismiss_entities=['model', 'param'],
    )
    plotting.plot_roi(
        splenium_mask_file,
        bg_img=os.path.join(temp_dir, 'brain_t1w_dwires.nii.gz'),
        output_file=splenium_plot,
        display_mode='mosaic',
    )

    # Calculate the mean values in the splenium
    splenium_values = pd.Series(
        data={
            'subject_id': bids_filters['subject'],
            'session_id': bids_filters['session'],
            'run': bids_filters['run'],
            'ISOVF': np.nanmean(isovf_splenium),
            'ICVF': np.nanmean(icvf_splenium),
            'MTsat': np.nanmean(mtsat_splenium),
            'ihMTR': np.nanmean(ihmtr_splenium),
        },
    )

    return splenium_values


def compute_scaling_factor(ICVF, MVF, ISOVF, g=0.7):
    """
    Compute the scaling_factor such that:
        g = sqrt(FVF / (FVF + MVFs))
    given:
        FVF = (1 - MVFs) * (1 - ISOVF) * ICVF
        MVFs = MVF * scaling_factor

    Parameters
    ----------
    ICVF : float or array-like
        Intra-cellular volume fraction.
    MVF : float or array-like
        Myelin volume fraction.
    ISOVF : float or array-like
        Isotropic volume fraction.
    g : float, optional
        Target g-ratio. Default is 0.7.

    Returns
    -------
    scaling_factor : float or ndarray
        Value that satisfies the given equation.
    """
    g2 = np.square(g)
    numerator = (1 - ISOVF) * ICVF * (1 - g2)
    denominator = MVF * (g2 + (1 - ISOVF) * ICVF * (1 - g2))
    return numerator / denominator


def main():
    in_dir = CFG['bids_dir']
    smriprep_dir = CFG['derivatives']['smriprep']
    qsiprep_dir = CFG['derivatives']['qsiprep']
    noddi_dir = CFG['derivatives']['qsirecon_noddi']
    ihmt_dir = CFG['derivatives']['ihmt']
    out_dir = CFG['derivatives']['g_ratio']
    os.makedirs(out_dir, exist_ok=True)
    temp_dir = os.path.join(CFG['work_dir'], 'g_ratio')
    os.makedirs(temp_dir, exist_ok=True)

    layout = BIDSLayout(
        in_dir,
        config=os.path.join(CODE_DIR, 'configuration', 'nibs_bids_config.json'),
        validate=False,
        derivatives=[smriprep_dir, qsiprep_dir, noddi_dir, ihmt_dir],
    )

    base_query = {
        'space': 'T1w',
        'suffix': 'ihMTR',
        'extension': ['.nii', '.nii.gz'],
    }
    subject_ids = layout.get_subjects(**base_query)
    print(f'Found {len(subject_ids)} subjects', flush=True)
    splenium_dfs = []
    for subject_id in subject_ids:
        print(f'Processing subject {subject_id}', flush=True)
        sessions = layout.get_sessions(subject=subject_id, **base_query)
        for session in sessions:
            print(f'Processing session {session}', flush=True)
            base_files = layout.get(
                subject=subject_id,
                session=session,
                **base_query,
            )
            if not base_files:
                print(
                    f'No ihMTR files found for subject {subject_id} and session {session}',
                    flush=True,
                )
                continue

            for base_file in base_files:
                entities = base_file.get_entities()
                try:
                    run_data = collect_run_data(layout, entities, smriprep_dir=smriprep_dir)
                except ValueError as e:
                    print(f'Failed {base_file}', flush=True)
                    print(e, flush=True)
                    continue

                fname = os.path.basename(base_file.path).split('.')[0]
                run_temp_dir = os.path.join(temp_dir, fname.replace('-', '').replace('_', ''))
                os.makedirs(run_temp_dir, exist_ok=True)
                splenium_values = process_run(layout, run_data, out_dir, run_temp_dir, entities)
                splenium_dfs.append(splenium_values)

    splenium_df = pd.DataFrame(splenium_dfs)
    splenium_df.to_csv(os.path.join(CODE_DIR, 'data/splenium_values.tsv'), sep='\t', index=False)

    # Calculate the scaling factors
    MTsat_ISOVF_ICVF_scalar = compute_scaling_factor(
        ICVF=splenium_df['ICVF'].mean(),
        MVF=splenium_df['MTsat'].mean(),
        ISOVF=splenium_df['ISOVF'].mean(),
        g=0.7,
    )
    ihMTR_ISOVF_ICVF_scalar = compute_scaling_factor(
        ICVF=splenium_df['ICVF'].mean(),
        MVF=splenium_df['ihMTR'].mean(),
        ISOVF=splenium_df['ISOVF'].mean(),
        g=0.7,
    )

    print(f'MTsat_ISOVF_ICVF_scalar: {MTsat_ISOVF_ICVF_scalar}', flush=True)
    print(f'ihMTR_ISOVF_ICVF_scalar: {ihMTR_ISOVF_ICVF_scalar}', flush=True)

    print('DONE!', flush=True)


if __name__ == '__main__':
    main()
