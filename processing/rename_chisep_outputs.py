"""Rename Chi-separation outputs to follow BIDS specification."""

import os
from glob import glob

import nibabel as nb

from utils import load_config


if __name__ == '__main__':
    cfg = load_config()
    in_dir = os.path.join(cfg['work_dir'], 'qsm')
    out_dir = cfg['derivatives']['qsm']
    subject_dirs = sorted(glob(os.path.join(in_dir, 'sub-*')))
    for subject_dir in subject_dirs:
        subject_id = os.path.basename(subject_dir)
        session_dirs = sorted(glob(os.path.join(subject_dir, 'ses-*')))
        for session_dir in session_dirs:
            session_id = os.path.basename(session_dir)

            ses_out_dir = os.path.join(out_dir, subject_id, session_id, 'anat')
            os.makedirs(ses_out_dir, exist_ok=True)

            # We have two output folders: output-12345 (all5) and outputE2345 (bw4)
            all5_dir = os.path.join(session_dir, 'anat', 'output-12345')
            bw4_dir = os.path.join(session_dir, 'anat', 'outputE2345')
            pipelines = {
                'output-12345': 'E12345+',
                'outputE2345': 'E2345+',
            }
            for pipeline, base_desc in pipelines.items():
                pipeline_dir = os.path.join(session_dir, 'anat', pipeline)
                if not os.path.exists(pipeline_dir):
                    print(f'{pipeline_dir} does not exist')
                    continue

                base = f'{subject_id}_{session_id}_run-01_space-MEGRE_desc-{base_desc}'
                renamer = {
                    f'{subject_id}_{session_id}_diamagnetic_r2p.nii': f'{base}chisep+r2p_myelinw.nii.gz',
                    f'{subject_id}_{session_id}_paramagnetic_r2p.nii': f'{base}chisep+r2p_ironw.nii.gz',
                    f'{subject_id}_{session_id}_total_r2p.nii': f'{base}chisep+r2p_Chimap.nii.gz',
                    f'{subject_id}_{session_id}_diamagnetic_r2primenet.nii': f'{base}chisep+r2primenet_myelinw.nii.gz',
                    f'{subject_id}_{session_id}_paramagnetic_r2primenet.nii': f'{base}chisep+r2primenet_ironw.nii.gz',
                    f'{subject_id}_{session_id}_total_r2primenet.nii': f'{base}chisep+r2primenet_Chimap.nii.gz',
                    f'{subject_id}_{session_id}_diamagnetic_r2s.nii': f'{base}chisep+r2s_myelinw.nii.gz',
                    f'{subject_id}_{session_id}_paramagnetic_r2s.nii': f'{base}chisep+r2s_ironw.nii.gz',
                    f'{subject_id}_{session_id}_total_r2s.nii': f'{base}chisep+r2s_Chimap.nii.gz',
                }
                for in_basename, out_basename in renamer.items():
                    in_file = os.path.join(pipeline_dir, in_basename)
                    out_file = os.path.join(ses_out_dir, out_basename)
                    if not os.path.exists(in_file):
                        print(f'{in_file} does not exist')
                        continue

                    nb.load(in_file).to_filename(out_file)
