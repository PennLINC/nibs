"""Expand dicom zip files in order to heudiconv."""

import os
import zipfile
from glob import glob

if __name__ == '__main__':
    status_file = '/cbica/projects/nibs/code/curation/status_unzip_dicom_zips.txt'
    if os.path.exists(status_file):
        with open(status_file, 'r') as f:
            unzipped_subjects = f.read().splitlines()
    else:
        unzipped_subjects = []

    zip_files = sorted(glob('/cbica/projects/nibs/sourcedata/*.zip'))
    for zip_file in zip_files:
        subject = os.path.basename(zip_file).split('.')[0]
        if subject in unzipped_subjects:
            print(f'Subject {subject} already processed, skipping...')
        else:
            print(f'Processing {subject}...')
            with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                zip_ref.extractall(os.path.dirname(zip_file))
            print(f'Subject {subject} processed successfully.')
            with open(status_file, 'a') as f:
                f.write(f'{subject}\n')
