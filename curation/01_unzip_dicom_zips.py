"""Expand dicom zip files in order to heudiconv."""

import os
import zipfile
from glob import glob

import yaml

if __name__ == '__main__':
    _cfg_path = os.path.join(os.path.dirname(__file__), '..', 'paths.yaml')
    with open(_cfg_path) as f:
        _cfg = yaml.safe_load(f)
    _root = _cfg['project_root']

    status_file = os.path.join(_root, _cfg['code_dir'], 'curation', 'status_unzip_dicom_zips.txt')
    if os.path.exists(status_file):
        with open(status_file, 'r') as f:
            unzipped_subjects = f.read().splitlines()
    else:
        unzipped_subjects = []

    zip_files = sorted(glob(os.path.join(_root, _cfg['sourcedata']['root'], '*.zip')))
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
