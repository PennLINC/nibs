"""Fix MP2RAGE phase image conversions."""

import json
import os
from glob import glob


if __name__ == '__main__':
    dset_dir = '/cbica/projects/nibs/dset'
    subject_dirs = sorted(glob(os.path.join(dset_dir, 'sub-*')))
    for subject_dir in subject_dirs:
        session_dirs = sorted(glob(os.path.join(subject_dir, 'ses-*')))
        for session_dir in session_dirs:
            mp2rage_dir = os.path.join(session_dir, 'anat')
            mp2rage_files = sorted(glob(os.path.join(mp2rage_dir, '*_part-phase_MP2RAGE*')))
            for mp2rage_file in mp2rage_files:
                if 'MP2RAGE_ph' not in mp2rage_file:
                    print(f'Misnamed magnitude image: {mp2rage_file}')
                    os.remove(mp2rage_file)
                    continue

                if 'inv-1' in mp2rage_file:
                    inv = 1
                elif 'inv-2' in mp2rage_file:
                    inv = 2
                else:
                    raise ValueError(f'Unknown inv: {mp2rage_file}')

                mp2rage_json = mp2rage_file.replace('.nii.gz', '.json')
                with open(mp2rage_json, 'r') as f:
                    metadata = json.load(f)

                assert f'INV{inv}_' in metadata['SeriesDescription']
                new_mp2rage_file = mp2rage_file.replace('MP2RAGE_ph.', 'MP2RAGE.')
                os.rename(mp2rage_file, new_mp2rage_file)
                new_mp2rage_json = mp2rage_json.replace('MP2RAGE_ph.', 'MP2RAGE.')
                os.rename(mp2rage_json, new_mp2rage_json)
