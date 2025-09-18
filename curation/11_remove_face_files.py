"""Remove face files produced by AFNI's refacer from dataset."""

import json
import os
from glob import glob


if __name__ == '__main__':
    dset_dir = '/cbica/projects/nibs/dset'
    face_files = glob(os.path.join(dset_dir, 'sub-*/ses-*/anat/*.face.nii.gz'))
    for face_file in face_files:
        os.remove(face_file)

        # Add "Defaced": true to JSON file
        json_file = face_file.replace('.face.nii.gz', '.json')
        if not os.path.isfile(json_file):
            raise FileNotFoundError(f"JSON file not found: {json_file}")

        with open(json_file, 'r') as fobj:
            data = json.load(fobj)

        data['Defaced'] = True
        with open(json_file, 'w') as fobj:
            json.dump(data, fobj, indent=4, sort_keys=True)
