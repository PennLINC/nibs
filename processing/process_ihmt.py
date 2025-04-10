"""Calculate derivatives for ihMTRAGE files."""

import os
from glob import glob

import nibabel as nb
from nilearn import image


if __name__ == "__main__":
    in_dir = "/Users/taylor/Downloads/flywheel/bbl/dset"
    out_dir = "/Users/taylor/Downloads/flywheel/bbl/derivatives/ihmt"
    patterns = {
        "M0": "_acq-nosat_mt-off_ihMTRAGE.nii.gz",
        "MT+": "_acq-singlepos_mt-on_ihMTRAGE.nii.gz",
        "MTdual1": "_acq-dual1_mt-on_ihMTRAGE.nii.gz",
        "MT-": "_acq-singleneg_mt-on_ihMTRAGE.nii.gz",
        "MTdual2": "_acq-dual2_mt-on_ihMTRAGE.nii.gz",
    }

    in_files = sorted(glob(os.path.join(in_dir, "sub-*", "ses-*", "anat", "sub-*_acq-nosat_mt-off_ihMTRAGE.nii.gz")))
    for in_file in in_files:
        imgs = {}
        print(in_file)
        basename = os.path.basename(in_file)
        for name, pattern in patterns.items():
            pattern_file = in_file.replace("_acq-nosat_mt-off_ihMTRAGE.nii.gz", pattern)
            imgs[name] = nb.load(pattern_file)

        ihmt_img = image.math_img(
            "mtplus + mtminus - (mtdual1 + mtdual2)",
            mtplus=imgs["MT+"],
            mtminus=imgs["MT-"],
            mtdual1=imgs["MTdual1"],
            mtdual2=imgs["MTdual2"],
        )
        ihmt_file = basename.replace("_acq-nosat_mt-off_ihMTRAGE.nii.gz", "_ihMTw.nii.gz")
        ihmt_img.to_filename(os.path.join(out_dir, ihmt_file))
        mtr_img = image.math_img("ihmt / m0", ihmt=ihmt_img, m0=imgs["M0"])
        mtr_file = basename.replace("_acq-nosat_mt-off_ihMTRAGE.nii.gz", "_MTR.nii.gz")
        mtr_img.to_filename(os.path.join(out_dir, mtr_file))
