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

import nibabel as nb
import numpy as np
from bids.layout import BIDSLayout, Query
from nilearn import image
from nireports.assembler.report import Report

from utils import fit_monoexponential, get_filename


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
        # MEGRE-space R2 map from MESE pipeline + prep step
        'r2_map': {
            'datatype': 'anat',
            'space': 'MEGRE',
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
        # Coregistration transform from MEGRE to T1w
        'megre2t1w_xfm': {
            'datatype': 'anat',
            'run': [Query.NONE, Query.ANY],
            'from': 'MEGRE',
            'to': 'T1w',
            'mode': 'image',
            'suffix': 'xfm',
            'extension': '.mat',
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

    # Collect MATLAB-compatible NIfTIs for QSM
    matlab_mask_filename = os.path.join(temp_dir, 'python_mask.nii')
    matlab_mag_filename = os.path.join(temp_dir, 'python_mag.nii')
    matlab_phase_filename = os.path.join(temp_dir, 'python_phase.nii')

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
        .replace("{{ output_dir }}", os.path.join(sepia_dir, 'sepia'))
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
            f"run('{out_sepia_script}'); exit;",
        ],
    )
    sepia_chimap_file = os.path.join(sepia_dir, 'sepia_Chimap.nii.gz')
    if not os.path.isfile(sepia_chimap_file):
        raise FileNotFoundError(f'SEPIA QSM output file {sepia_chimap_file} not found')

    sepia_chimap_img = nb.load(sepia_chimap_file)
    sepia_chimap_filename = get_filename(
        name_source=name_source,
        layout=layout,
        out_dir=out_dir,
        entities={'space': 'MEGRE', 'desc': 'SEPIA', 'suffix': 'Chimap'},
        dismiss_entities=['echo', 'part', 'inv', 'reconstruction'],
    )
    sepia_chimap_img.to_filename(sepia_chimap_filename)


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
        derivatives=[smriprep_dir, mese_dir, out_dir],
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
