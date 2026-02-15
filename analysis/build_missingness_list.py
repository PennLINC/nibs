import os
from glob import glob

import pandas as pd

from utils import convert_to_multindex, matrix

# Repository root (one level up from analysis/)
CODE_DIR = os.path.normpath(os.path.join(os.path.dirname(__file__), '..'))


if __name__ == '__main__':
    PATTERNS = {
        'MPRAGE T1w': ['anat/*acq-MPRAGE*T1w.nii.gz'],
        'SPACE T1w': ['anat/*acq-SPACE*T1w.nii.gz'],
        'SPACE T2w': ['anat/*acq-SPACE*T2w.nii.gz'],
        'MP2RAGE': ['anat/*part-mag*MP2RAGE.nii.gz'],
        'MP2RAGE-P': ['anat/*part-phase*MP2RAGE.nii.gz'],
        'dMRI': ['dwi/*dir-AP*dwi.nii.gz', 'dwi/*dir-PA*dwi.nii.gz'],
        'MEGRE': ['anat/*MEGRE.nii.gz'],
        'ihMTRAGE': ['anat/*ihMTRAGE.nii.gz'],
        'MESE': ['anat/*dir-AP*MESE.nii.gz'],
        'B1+': ['fmap/*TB1TFL.nii.gz'],
    }
    SESSIONS = {
        'Session 01': 'ses-01',
        'Session 02': 'ses-02',
    }

    in_dir = '/cbica/projects/nibs/dset'
    participants_file = os.path.join(in_dir, 'participants.tsv')
    participants = pd.read_table(participants_file)
    subject_ids = participants['participant_id'].tolist()
    # Move PILOT subjects to the beginning of the list
    pilots = [sid for sid in subject_ids if 'PILOT' in sid]
    subject_ids = [sid for sid in subject_ids if 'PILOT' not in sid]
    subject_ids = pilots + subject_ids

    columns = []
    for ses_name in SESSIONS.keys():
        for modality_name in PATTERNS.keys():
            columns.append(f'{ses_name}--{modality_name}')

    df = pd.DataFrame(
        index=subject_ids,
        columns=columns,
    )
    for subject_id in subject_ids:
        for ses_name, ses_id in SESSIONS.items():
            for modality_name, modality_patterns in PATTERNS.items():
                found = True
                for modality_pattern in modality_patterns:
                    files = sorted(glob(os.path.join(in_dir, subject_id, ses_id, modality_pattern)))
                    if not files:
                        found = None
                        break

                df.loc[subject_id, f'{ses_name}--{modality_name}'] = found

    df.to_csv(
        os.path.join(CODE_DIR, 'data', 'missingness_list.tsv'),
        sep='\t',
        index=True,
        index_label='participant_id',
    )

    df = convert_to_multindex(df)
    ax = matrix(df)
    ax.figure.savefig(
        os.path.join(CODE_DIR, 'figures', 'missingness.png'),
        bbox_inches='tight',
    )
