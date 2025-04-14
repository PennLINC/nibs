#!/cbica/home/salot/miniconda3/envs/salot/bin/python
"""Fix BIDS files after heudiconv conversion.

The necessary steps are:

1.  Drop part-phase bvec and bval files from fmap and dwi directories.
    - QSIPrep actually uses the part-mag bvec and bval files.
2.  Drop part entity from part-mag bvec and bval filenames in fmap and dwi directories.
3.  Add "Units": "arbitrary" to all phase JSONs.
4.  Add the following fields to MP2RAGE JSONs:
    - "NumberShots": 176
    - "RepetitionTimeExcitation": base on RepetitionTime
    - "RepetitionTimePreparation": base on RepetitionTime
"""

import json
import os
import shutil
from glob import glob


if __name__ == "__main__":
    # dset_dir = "/cbica/projects/nibs/dset"
    dset_dir = "/Users/taylor/Documents/datasets/nibs/dset"

    subject_dirs = sorted(glob(os.path.join(dset_dir, "sub-*")))
    for subject_dir in subject_dirs:
        sub_id = os.path.basename(subject_dir)
        session_dirs = sorted(glob(os.path.join(subject_dir, "ses-*")))
        for session_dir in session_dirs:
            ses_id = os.path.basename(session_dir)

            dwi_dir = os.path.join(session_dir, "dwi")

            # Remove part-phase bvec and bval DWI files
            part_phase_bvecs = sorted(glob(os.path.join(dwi_dir, "*_part-phase*.bvec")))
            for part_phase_bvec in part_phase_bvecs:
                os.remove(part_phase_bvec)
            part_phase_bvals = sorted(glob(os.path.join(dwi_dir, "*_part-phase*.bval")))
            for part_phase_bval in part_phase_bvals:
                os.remove(part_phase_bval)

            # Copy part-mag bvec and bval DWI files to versions without part entity
            part_mag_bvecs = sorted(glob(os.path.join(dwi_dir, "*_part-mag*.bvec")))
            for part_mag_bvec in part_mag_bvecs:
                new_part_mag_bvec = part_mag_bvec.replace("_part-mag", "")
                shutil.copyfile(part_mag_bvec, new_part_mag_bvec)
            part_mag_bvals = sorted(glob(os.path.join(dwi_dir, "*_part-mag*.bval")))
            for part_mag_bval in part_mag_bvals:
                new_part_mag_bval = part_mag_bval.replace("_part-mag", "")
                shutil.copyfile(part_mag_bval, new_part_mag_bval)

            # Add Units: arbitrary to all phase JSONs
            phase_jsons = sorted(glob(os.path.join(session_dir, "*", "*part-phase*.json")))
            for phase_json in phase_jsons:
                with open(phase_json, "r") as f:
                    data = json.load(f)
                data["Units"] = "arbitrary"
                with open(phase_json, "w") as f:
                    json.dump(data, f, indent=4, sort_keys=True)

            # Add fields to MP2RAGE JSONs
            mp2rage_jsons = sorted(glob(os.path.join(session_dir, "anat", "*_MP2RAGE.json")))
            for mp2rage_json in mp2rage_jsons:
                with open(mp2rage_json, "r") as f:
                    data = json.load(f)
                data["NumberShots"] = 176  # SlicesPerSlab
                data["RepetitionTimeExcitation"] = 0.0063  # 6.3 ms
                data["RepetitionTimePreparation"] = data["RepetitionTime"]
                with open(mp2rage_json, "w") as f:
                    json.dump(data, f, indent=4, sort_keys=True)

    # Add multi-echo field maps to .bidsignore.
    bidsignore_file = os.path.join(dset_dir, ".bidsignore")
    with open(bidsignore_file, "a") as f:
        f.write("\n*_ihMTRAGE.*\n")
        f.write("swi\n")
