"""Remove unneeded fields from bottom-level JSON files."""

import json
import os
from glob import glob

import yaml


if __name__ == '__main__':
    _cfg_path = os.path.join(os.path.dirname(__file__), '..', 'paths.yaml')
    with open(_cfg_path) as f:
        _cfg = yaml.safe_load(f)
    _root = _cfg['project_root']

    dset_dir = os.path.join(_root, _cfg['bids_dir'])
    drop_keys = [
        'AcquisitionTime',
        'AcquisitionDateTime',
        'CogAtlasID',
        'EchoTime1',
        'EchoTime2',
        'InstitutionAddress',
        'TaskName',
        'ImageComments',
    ]

    json_files = sorted(glob(os.path.join(dset_dir, 'sub-*/ses-*/*/*.json')))
    for json_file in json_files:
        print(json_file)
        with open(json_file, 'r') as fo:
            json_data = json.load(fo)

        for drop_key in drop_keys:
            if drop_key in json_data.keys():
                json_data.pop(drop_key)

        with open(json_file, 'w') as fo:
            json.dump(json_data, fo, indent=4, sort_keys=True)
