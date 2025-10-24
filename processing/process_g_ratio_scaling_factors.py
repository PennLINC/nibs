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

import os

import ants
import numpy as np
import pandas as pd
from bids.layout import BIDSLayout, Query
from nilearn import masking, plotting

from utils import get_filename

CODE_DIR = '/cbica/projects/nibs/code'


def collect_run_data(layout, bids_filters, smriprep_dir):
    queries = {
        # MNI-space ISOVF and ICVF maps from QSIRecon
        'isovf_mni': {
            'datatype': 'dwi',
            'run': [Query.NONE, Query.ANY],
       	    'reconstruction': [Query.NONE, Query.ANY],
            'space': 'MNI152NLin2009cAsym',
            'model': 'noddi',
            'param': 'isovf',
            'desc': Query.NONE,
            'suffix': 'dwimap',
            'extension': ['.nii', '.nii.gz'],
        },
        'icvf_mni': {
            'datatype': 'dwi',
            'run': [Query.NONE, Query.ANY],
       	    'reconstruction': [Query.NONE, Query.ANY],
            'space': 'MNI152NLin2009cAsym',
            'model': 'noddi',
            'param': 'icvf',
            'desc': Query.NONE,
            'suffix': 'dwimap',
            'extension': ['.nii', '.nii.gz'],
        },
        # ihMT-space MTsat and ihMTR maps from process_ihmt.py
        'mtsat_ihmtrageref': {
            'datatype': 'anat',
            'run': [Query.NONE, Query.ANY],
       	    'reconstruction': [Query.NONE, Query.ANY],
            'space': 'ihMTref',
            'suffix': 'ihMTsatB1sq',
            'extension': ['.nii', '.nii.gz'],
        },
        'ihmtr_ihmtrageref': {
            'datatype': 'anat',
            'run': [Query.NONE, Query.ANY],
       	    'reconstruction': [Query.NONE, Query.ANY],
            'space': 'MNI152NLin2009cAsym',
            'suffix': 'ihMTR',
            'extension': ['.nii', '.nii.gz'],
        },
        # Transform from ihMT-space to T1w space
        'ihmtrageref2t1w_xfm': {
            'datatype': 'anat',
            'reconstruction': [Query.NONE, Query.ANY],
            'space': Query.NONE,
            'run': [Query.NONE, Query.ANY],
            'to': 'T1w',
            'from': 'ihMTRAGEref',
            'mode': 'image',
            'suffix': 'xfm',
            'extension': '.mat',
        },
        # Transform from sMRIPrep T1w to MNI space
        't1w2mni_xfm': {
            'datatype': 'anat',
            'session': [Query.NONE, Query.ANY],
            'reconstruction': [Query.NONE, Query.ANY],
            'space': Query.NONE,
            'run': [Query.NONE, Query.ANY],
            'to': 'MNI152NLin2009cAsym',
            'from': 'T1w',
            'mode': 'image',
            'suffix': 'xfm',
            'extension': '.h5',
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
    assert os.path.isfile(run_data['aseg_fsnative']), f'Aseg file {run_data["aseg_fsnative"]} not found'

    run_data['brain_fsnative'] = os.path.join(
        smriprep_dir,
        'sourcedata',
        'freesurfer',
        f'sub-{bids_filters["subject"]}',
        'mri',
        'brain.mgz',
    )
    assert os.path.isfile(run_data['brain_fsnative']), f'Brain file {run_data["brain_fsnative"]} not found'

    return run_data


def process_run(layout, run_data, out_dir, temp_dir, bids_filters):
    """Process a single MP2RAGE run.

    Parameters
    ----------
    run_data : dict
        Dictionary of run data.
    temp_dir : str
        Directory to write temporary files.

    Returns
    -------
    splenium_g_ratios : np.ndarray of shape (4, n_voxels)
        Array of g-ratios in the splenium.
    """
    # Load images for target resolutions
    isovf = ants.image_read(run_data['isovf_mni'])  # DWI resolution (1.7 mm isotropic)

    # Select only the splenium voxels
    splenium_mask = ants.image_read(run_data['aseg_fsnative']) == 251
    splenium_mask_dwires = ants.apply_transforms(
        fixed=isovf,
        moving=splenium_mask,
        transformlist=[run_data['t1w2mni_xfm'], run_data['fs2t1w_xfm']],
        interpolator='nearestNeighbor',
    )
    splenium_mask_file_dwires = os.path.join(temp_dir, 'splenium_mask_mni_dwires.nii.gz')
    ants.image_write(splenium_mask_dwires, splenium_mask_file_dwires)

    # Warp ihMTRAGEref-space MTsat and ihMTR to MNI space
    mtsat_mni = ants.apply_transforms(
        fixed=isovf,
        moving=ants.image_read(run_data['mtsat_ihmtrageref']),
        transformlist=[run_data['t1w2mni_xfm'], run_data['ihmtrageref2t1w_xfm']],
        interpolator='nearestNeighbor',
    )
    mtsat_mni_file = os.path.join(temp_dir, 'mtsat_mni.nii.gz')
    ants.image_write(mtsat_mni, mtsat_mni_file)
    ihmtr_mni = ants.apply_transforms(
        fixed=isovf,
        moving=ants.image_read(run_data['ihmtr_ihmtrageref']),
        transformlist=[run_data['t1w2mni_xfm'], run_data['ihmtrageref2t1w_xfm']],
        interpolator='nearestNeighbor',
    )
    ihmtr_mni_file = os.path.join(temp_dir, 'ihmtr_mni.nii.gz')
    ants.image_write(ihmtr_mni, ihmtr_mni_file)

    # Get the data in the splenium
    isovf_splenium = masking.apply_mask(run_data['isovf_mni'], splenium_mask_file_dwires)
    icvf_splenium = masking.apply_mask(run_data['icvf_mni'], splenium_mask_file_dwires)
    mtsat_splenium = masking.apply_mask(mtsat_mni_file, splenium_mask_file_dwires)
    ihmtr_splenium = masking.apply_mask(ihmtr_mni_file, splenium_mask_file_dwires)

    # Plot the splenium mask on top of the brain
    brain_img = ants.image_read(run_data['brain_fsnative'])
    brain_img_dwires = ants.apply_transforms(
        fixed=isovf,
        moving=brain_img,
        transformlist=[run_data['t1w2mni_xfm'],run_data['fs2t1w_xfm']],
        interpolator='lanczosWindowedSinc',
    )
    ants.image_write(brain_img_dwires, os.path.join(temp_dir, 'brain_mni_dwires.nii.gz'))

    splenium_plot = get_filename(
        name_source=run_data['isovf_mni'],
        layout=layout,
        out_dir=out_dir,
        entities={'datatype': 'figures', 'desc': 'splenium', 'suffix': 'mask', 'extension': '.svg'},
        dismiss_entities=['model', 'param'],
    )
    plotting.plot_roi(
        splenium_mask_file_dwires,
        bg_img=os.path.join(temp_dir, 'brain_mni_dwires.nii.gz'),
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
    in_dir = '/cbica/projects/nibs/dset'
    smriprep_dir = '/cbica/projects/nibs/derivatives/smriprep'
    dipydki_dir = '/cbica/projects/nibs/derivatives/qsirecon/derivatives/qsirecon-DIPYDKI'
    noddi_dir = '/cbica/projects/nibs/derivatives/qsirecon/derivatives/qsirecon-NODDI'
    ihmt_dir = '/cbica/projects/nibs/derivatives/ihmt'
    out_dir = '/cbica/projects/nibs/derivatives/g_ratio'
    os.makedirs(out_dir, exist_ok=True)
    temp_dir = '/cbica/projects/nibs/work/g_ratio'
    os.makedirs(temp_dir, exist_ok=True)

    layout = BIDSLayout(
        in_dir,
        config=os.path.join(CODE_DIR, 'nibs_bids_config.json'),
        validate=False,
        derivatives=[smriprep_dir, dipydki_dir, noddi_dir, ihmt_dir],
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
                print(f'No ihMTR files found for subject {subject_id} and session {session}', flush=True)
                continue

            for base_file in base_files:
                entities = base_file.get_entities()
                try:
                    run_data = collect_run_data(layout, entities, smriprep_dir=smriprep_dir)
                except ValueError as e:
                    print(f'Failed {base_file}, flush=True')
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
