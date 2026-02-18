"""Compute mean Fisher-z correlation matrices across subjects for each tissue mask."""

import json
import os
import sys
from glob import glob

import numpy as np
import pandas as pd


if __name__ == '__main__':
    _script_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, os.path.join(_script_dir, '..'))
    from config import load_config

    _cfg = load_config()

    bids_dir = _cfg['bids_dir']
    work_dir = os.path.join(_cfg['work_dir'], 'correlation_matrices')
    out_dir = os.path.join(_script_dir, '..', 'data')

    masks = [
        'cortical_gm',
        'deep_gm',
        'gm',
        'wm',
        'wb',
    ]

    with open(os.path.join(_script_dir, 'patterns.json'), 'r') as f:
        patterns = json.load(f)

    flat_patterns = {k: v for subdict in patterns.values() for k, v in subdict.items()}
    subject_dirs = sorted(glob(os.path.join(bids_dir, 'sub-*')))
    subjects = [os.path.basename(d) for d in subject_dirs]
    subjects = [s for s in subjects if not s.startswith('sub-PILOT')]
    for mask in masks:
        mask_arrs = []
        for subject in subjects:
            session_dirs = sorted(glob(os.path.join(bids_dir, subject, 'ses-*')))
            sessions = [os.path.basename(d) for d in session_dirs]
            subject_arrs = []
            for session in sessions:
                in_file = os.path.join(work_dir, f'{subject}_{session}_{mask}.npy')
                if not os.path.exists(in_file):
                    print(f'{in_file} does not exist')
                    continue

                arr = np.load(in_file)
                corr_mat = np.arctanh(np.corrcoef(arr))
                np.fill_diagonal(corr_mat, 0)
                subject_arrs.append(corr_mat)

            subject_corr_mat = np.nanmean(np.stack(subject_arrs), axis=0)
            mask_arrs.append(subject_corr_mat)

        full_corr_mat = np.nanmean(np.stack(mask_arrs), axis=0)
        corr_df = pd.DataFrame(
            full_corr_mat,
            index=flat_patterns.keys(),
            columns=flat_patterns.keys(),
        )
        corr_df.to_csv(
            os.path.join(out_dir, f'mean_{mask}_corr_mat.tsv'),
            sep='\t',
            index=True,
            index_label='Image',
        )
