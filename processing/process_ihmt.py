"""Calculate derivatives for ihMTRAGE files."""

import json
import os
from glob import glob

from nilearn import image


def to_bidsuri(filename, dataset_dir, dataset_name):
    return f"bids:{dataset_name}:{os.path.relpath(filename, dataset_dir)}"


if __name__ == "__main__":
    # in_dir = "/cbica/projects/nibs/dset"
    in_dir = "/Users/taylor/Documents/datasets/nibs/dset"
    # out_dir = "/cbica/projects/nibs/derivatives/ihmt"
    out_dir = "/Users/taylor/Documents/datasets/nibs/derivatives/ihmt"

    smriprep_dir = "/cbica/projects/nibs/derivatives/smriprep"

    os.makedirs(out_dir, exist_ok=True)

    patterns = {
        "M0": "_acq-nosat_{run}_mt-off_ihMTRAGE.nii.gz",
        "MT+": "_acq-singlepos_{run}_mt-on_ihMTRAGE.nii.gz",
        "MTdual1": "_acq-dual1_{run}_mt-on_ihMTRAGE.nii.gz",
        "MT-": "_acq-singleneg_{run}_mt-on_ihMTRAGE.nii.gz",
        "MTdual2": "_acq-dual2_{run}_mt-on_ihMTRAGE.nii.gz",
    }
    dataset_description = {
        "Name": "NIBS",
        "BIDSVersion": "1.10.0",
        "DatasetType": "derivative",
        "DatasetLinks": {
            "raw": in_dir,
        },
        "GeneratedBy": [
            {
                "Name": "Custom code",
                "Description": "Custom Python code to calculate ihMTw and MTR.",
                "CodeURL": "https://github.com/PennLINC/nibs",
            }
        ],
    }
    with open(os.path.join(out_dir, "dataset_description.json"), "w") as f:
        json.dump(dataset_description, f, sort_keys=True, indent=4)

    subject_dirs = sorted(glob(os.path.join(in_dir, "sub-*")))
    for subject_dir in subject_dirs:
        subject_id = os.path.basename(subject_dir)
        session_dirs = sorted(glob(os.path.join(subject_dir, "ses-*")))
        for session_dir in session_dirs:
            session_id = os.path.basename(session_dir)
            in_files = sorted(glob(os.path.join(session_dir, "anat", "sub-*_mt-off_ihMTRAGE.nii.gz")))
            for in_file in in_files:
                run_out_dir = os.path.join(out_dir, subject_id, session_id, "anat")
                os.makedirs(run_out_dir, exist_ok=True)

                imgs = {}
                basename = os.path.basename(in_file)
                base_entities = basename.split("_")
                base_entities = base_entities[:-1]
                base_entities = [e for e in base_entities if not e.startswith("acq")]
                base_entities = [e for e in base_entities if not e.startswith("mt")]
                run = [e for e in base_entities if e.startswith("run")]
                run = run[0]
                basename = "_".join(base_entities)

                for name, pattern in patterns.items():
                    pattern_file = in_file.replace(
                        f"_acq-nosat_{run}_mt-off_ihMTRAGE.nii.gz",
                        pattern.format(run=run),
                    )
                    imgs[name] = pattern_file

                # Calculate ihMTw
                ihmt_img = image.math_img(
                    "mtplus + mtminus - (mtdual1 + mtdual2)",
                    mtplus=imgs["MT+"],
                    mtminus=imgs["MT-"],
                    mtdual1=imgs["MTdual1"],
                    mtdual2=imgs["MTdual2"],
                )
                ihmt_file = f"{basename}_ihMTw.nii.gz"
                ihmt_file = os.path.join(run_out_dir, ihmt_file)
                ihmt_img.to_filename(ihmt_file)
                metadata = {
                    "Sources": [
                        to_bidsuri(imgs["MT+"], in_dir, "raw"),
                        to_bidsuri(imgs["MT-"], in_dir, "raw"),
                        to_bidsuri(imgs["MTdual1"], in_dir, "raw"),
                        to_bidsuri(imgs["MTdual2"], in_dir, "raw"),
                    ],
                }
                metadata_file = ihmt_file.replace(".nii.gz", ".json")
                with open(metadata_file, "w") as f:
                    json.dump(metadata, f, sort_keys=True, indent=4)

                # Calculate MTR
                mtr_img = image.math_img("ihmt / m0", ihmt=ihmt_img, m0=imgs["M0"])
                mtr_file = f"{basename}_MTR.nii.gz"
                mtr_file = os.path.join(run_out_dir, mtr_file)
                mtr_img.to_filename(mtr_file)
                metadata = {
                    "Sources": [
                        to_bidsuri(ihmt_file, out_dir, ""),
                        to_bidsuri(imgs["M0"], in_dir, "raw"),
                    ],
                }
                metadata_file = mtr_file.replace(".nii.gz", ".json")
                with open(metadata_file, "w") as f:
                    json.dump(metadata, f, sort_keys=True, indent=4)

                # Calculate rigid transform to T1w
                t1_file = os.path.join(session_dir, "anat", f"{subject_id}_{session_id}_T1w.nii.gz")
