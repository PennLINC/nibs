#!/cbica/home/salot/miniconda3/envs/salot/bin/python
"""Remove unneeded fields from bottom-level JSON files."""

import json
import os
from glob import glob


if __name__ == '__main__':
    dset_dir = '/cbica/projects/nibs/dset'
    # dset_dir = "/Users/taylor/Documents/datasets/nibs/dset"
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
