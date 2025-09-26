"""Plot scalar maps from myelin measures."""

import os
import warnings
from glob import glob

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import templateflow.api as tflow
from nilearn import image, maskers, plotting


if __name__ == "__main__":
    in_dir = "/cbica/projects/nibs/derivatives"
    out_dir = "../figures"
    template = tflow.get("MNI152NLin2009cAsym", resolution="01", desc="brain", suffix="T1w", extension="nii.gz")
    mask = tflow.get("MNI152NLin2009cAsym", resolution="01", desc="brain", suffix="mask", extension="nii.gz")

    patterns = {
        # dMRI
        "Fractional Anisotropy": "qsirecon/derivatives/qsirecon-DIPYDKI/sub-*/ses-{ses}/dwi/*_space-MNI152NLin2009cAsym_model-tensor_param-fa_dwimap.nii.gz",
        "Axial Diffusivity": "qsirecon/derivatives/qsirecon-DIPYDKI/sub-*/ses-{ses}/dwi/*_space-MNI152NLin2009cAsym_model-dki_param-ad_dwimap.nii.gz",
        "Mean Diffusivity": "qsirecon/derivatives/qsirecon-DIPYDKI/sub-*/ses-{ses}/dwi/*_space-MNI152NLin2009cAsym_model-dki_param-md_dwimap.nii.gz",
        "Mean Kurtosis": "qsirecon/derivatives/qsirecon-DIPYDKI/sub-*/ses-{ses}/dwi/*_space-MNI152NLin2009cAsym_model-dki_param-mk_dwimap.nii.gz",
        "Radial Diffusivity": "qsirecon/derivatives/qsirecon-DIPYDKI/sub-*/ses-{ses}/dwi/*_space-MNI152NLin2009cAsym_model-dki_param-rd_dwimap.nii.gz",
        "Radial Kurtosis": "qsirecon/derivatives/qsirecon-DIPYDKI/sub-*/ses-{ses}/dwi/*_space-MNI152NLin2009cAsym_model-dki_param-rk_dwimap.nii.gz",
        # ihMT
        "Inhomogeneous Magnetization Transfer-Weighted": "ihmt/sub-*/ses-{ses}/anat/*_space-MNI152NLin2009cAsym_ihMTw.nii.gz",
        "Inhomogeneous Magnetization Transfer Ratio": "ihmt/sub-*/ses-{ses}/anat/*_space-MNI152NLin2009cAsym_ihMTR.nii.gz",
        "Magnetization Transfer Ratio": "ihmt/sub-*/ses-{ses}/anat/*_space-MNI152NLin2009cAsym_MTRmap.nii.gz",
        "Inhomogeneous Magnetization Transfer Saturation": "ihmt/sub-*/ses-{ses}/anat/*_space-MNI152NLin2009cAsym_ihMTsat.nii.gz",
        "B1-Corrected Inhomogeneous Magnetization Transfer Saturation": "ihmt/sub-*/ses-{ses}/anat/*_space-MNI152NLin2009cAsym_ihMTsatB1sq.nii.gz",
        # MP2RAGE
        "R1": "pymp2rage/sub-*/ses-{ses}/anat/*_space-MNI152NLin2009cAsym_R1map.nii.gz",
        "B1-Corrected R1": "pymp2rage/sub-*/ses-{ses}/anat/*_space-MNI152NLin2009cAsym_desc-B1corrected_R1map.nii.gz",
        # T1w/T2w Ratio
        "MPRAGE T1w/SPACE T2w Ratio": "t1wt2w_ratio/sub-*/ses-{ses}/anat/*_space-MNI152NLin2009cAsym_desc-MPRAGEunscaled_myelinw.nii.gz",
        "SPACE T1w/SPACE T2w Ratio": "t1wt2w_ratio/sub-*/ses-{ses}/anat/*_space-MNI152NLin2009cAsym_desc-SPACEunscaled_myelinw.nii.gz",
        "Scaled MPRAGE T1w/SPACE T2w Ratio": "t1wt2w_ratio/sub-*/ses-{ses}/anat/*_space-MNI152NLin2009cAsym_desc-MPRAGEscaled_myelinw.nii.gz",
        "Scaled SPACE T1w/SPACE T2w Ratio": "t1wt2w_ratio/sub-*/ses-{ses}/anat/*_space-MNI152NLin2009cAsym_desc-SPACEscaled_myelinw.nii.gz",
    }
    for title, pattern in patterns.items():
        temp_pattern = pattern.format(ses='*')

        # Get all scalar maps
        scalar_maps = sorted(glob(os.path.join(in_dir, temp_pattern)))
        scalar_maps = [f for f in scalar_maps if "PILOT" not in f]
        print(f"{title}: {len(scalar_maps)}")

        # Mask out non-brain voxels
        masker = maskers.NiftiMasker(mask, resampling_target="data")
        mean_img = image.mean_img(scalar_maps, copy_header=True)
        sd_img = image.math_img("np.std(img, axis=3)", img=scalar_maps)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mean_arr = masker.fit_transform(mean_img)
            sd_arr = masker.fit_transform(sd_img)

        # Get vmax (98th percentile) across both sessions
        mean_arr[np.isnan(mean_arr)] = 0
        mean_arr[np.isinf(mean_arr)] = 0
        sd_arr[np.isnan(sd_arr)] = 0
        sd_arr[np.isinf(sd_arr)] = 0
        vmax0 = np.round(np.percentile(mean_arr, 98), 2)
        vmax1 = np.round(np.percentile(sd_arr, 98), 2)
        print(f"\t{vmax0}, {vmax1}")

        for ses in ['01', '02']:
            temp_pattern = pattern.format(ses=ses)

            # Get all scalar maps
            scalar_maps = sorted(glob(os.path.join(in_dir, temp_pattern)))
            scalar_maps = [f for f in scalar_maps if "PILOT" not in f]
            print(f"\t{ses}: {len(scalar_maps)}")

            # Mask out non-brain voxels
            masker = maskers.NiftiMasker(mask, resampling_target="data")
            mean_img = image.mean_img(scalar_maps, copy_header=True)
            sd_img = image.math_img("np.std(img, axis=3)", img=scalar_maps)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                mean_arr = masker.fit_transform(mean_img)
                sd_arr = masker.fit_transform(sd_img)
                mean_arr[np.isnan(mean_arr)] = 0
                mean_arr[np.isinf(mean_arr)] = 0
                sd_arr[np.isnan(sd_arr)] = 0
                sd_arr[np.isinf(sd_arr)] = 0
                mean_img = masker.inverse_transform(mean_arr)
                sd_img = masker.inverse_transform(sd_arr)

            # Plot mean and SD
            fig, axs = plt.subplots(2, 1, figsize=(10, 5))
            plotting.plot_stat_map(
                mean_img,
                bg_img=template,
                display_mode="z",
                cut_coords=[-30, -15, 0, 15, 30, 45, 60],
                axes=axs[0],
                figure=fig,
                symmetric_cbar=False,
                vmin=0,
                vmax=vmax0,
                cmap="viridis",
                annotate=False,
                black_bg=False,
                resampling_interpolation="nearest",
                colorbar=False,
            )
            plotting.plot_stat_map(
                sd_img,
                bg_img=template,
                display_mode="z",
                cut_coords=[-30, -15, 0, 15, 30, 45, 60],
                axes=axs[1],
                figure=fig,
                symmetric_cbar=False,
                vmin=0,
                vmax=vmax1,
                cmap="viridis",
                annotate=False,
                black_bg=False,
                resampling_interpolation="nearest",
                colorbar=False,
            )
            fig.savefig(
                os.path.join(
                    out_dir,
                    f"{title.lower().replace('/', '_').replace(' ', '_')}_ses-{ses}.png",
                ),
                bbox_inches="tight",
            )
            plt.close()

            # Plot the colorbars
            fig, axs = plt.subplots(2, 1, figsize=(10, 1.5))
            cmap = mpl.cm.viridis

            norm = mpl.colors.Normalize(vmin=0, vmax=vmax0)
            cbar = fig.colorbar(
                mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
                cax=axs[0],
                orientation='horizontal',
            )
            cbar.set_ticks([0, np.mean([0, vmax0]), vmax0])

            norm = mpl.colors.Normalize(vmin=0, vmax=vmax1)
            cbar = fig.colorbar(
                mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
                cax=axs[1],
                orientation='horizontal',
            )
            cbar.set_ticks([0, np.mean([0, vmax1]), vmax1])

            fig.tight_layout()
            fig.savefig(
                os.path.join(
                    out_dir,
                    f"{title.lower().replace('/', '_').replace(' ', '_')}_ses-{ses}_colorbar.png",
                ),
                bbox_inches="tight",
            )
            plt.close()
