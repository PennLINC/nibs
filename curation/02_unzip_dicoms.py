"""Expand dicom zip files in order to heudiconv.

Rename the folders before running this.
"""

import os
import zipfile
from glob import glob

if __name__ == '__main__':
    status_file = '/cbica/projects/nibs/code/curation/status_unzip_dicoms.txt'
    if os.path.exists(status_file):
        with open(status_file, 'r') as f:
            unzipped_subjects = f.read().splitlines()
    else:
        unzipped_subjects = []

    subjects = sorted(glob('/cbica/projects/nibs/sourcedata/scitran/bbl/NIBS_857664/*'))
    subjects = [os.path.basename(subject) for subject in subjects]

    for subject in subjects:
        if subject in unzipped_subjects:
            print(f'Subject {subject} already processed, skipping...')
            continue

        zip_files = sorted(
            glob(
                f'/cbica/projects/nibs/sourcedata/scitran/bbl/NIBS_857664/{subject}/*/*/*.dicom.zip'
            )
        )
        print(f'Processing {subject}...')
        for zip_file in zip_files:
            with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                zip_ref.extractall(os.path.dirname(zip_file))

            os.remove(zip_file)

        with open(status_file, 'a') as f:
            f.write(f'{subject}\n')
