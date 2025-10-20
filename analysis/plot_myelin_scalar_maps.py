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
        # G-Ratio
        "G-Ratio MPRAGE": "g_ratio/sub-*/ses-{ses}/anat/*_space-MNI152NLin2009cAsym_desc-MPRAGET1wT2w+ISOVF+ICVF_gratio.nii.gz",
        "G-Ratio SPACE": "g_ratio/sub-*/ses-{ses}/anat/*_space-MNI152NLin2009cAsym_desc-SPACET1wT2w+ISOVF+ICVF_gratio.nii.gz",
        "G-Ratio ihMTsat": "g_ratio/sub-*/ses-{ses}/anat/*_space-MNI152NLin2009cAsym_desc-MTsat+ISOVF+ICVF_gratio.nii.gz",
        "G-Ratio ihMTR": "g_ratio/sub-*/ses-{ses}/anat/*_space-MNI152NLin2009cAsym_desc-ihMTR+ISOVF+ICVF_gratio.nii.gz",
        # QSM
        "QSM SEPIA 5-Echo Chi Map": "qsm/sub-*/ses-{ses}/anat/*_space-MNI152NLin2009cAsym_desc-E12345+sepia_Chimap.nii.gz",
        "QSM SEPIA 4-Echo Chi Map": "qsm/sub-*/ses-{ses}/anat/*_space-MNI152NLin2009cAsym_desc-E2345+sepia_Chimap.nii.gz",
        "QSM Chi-separation+R2' 5-Echo Chi Map": "qsm/sub-*/ses-{ses}/anat/*_space-MNI152NLin2009cAsym_desc-E12345+chisep+r2p_Chimap.nii.gz",
        "QSM Chi-separation+R2' 4-Echo Chi Map": "qsm/sub-*/ses-{ses}/anat/*_space-MNI152NLin2009cAsym_desc-E2345+chisep+r2p_Chimap.nii.gz",
        "QSM Chi-separation+R2' 5-Echo Iron Map": "qsm/sub-*/ses-{ses}/anat/*_space-MNI152NLin2009cAsym_desc-E12345+chisep+r2p_ironw.nii.gz",
        "QSM Chi-separation+R2' 4-Echo Iron Map": "qsm/sub-*/ses-{ses}/anat/*_space-MNI152NLin2009cAsym_desc-E2345+chisep+r2p_ironw.nii.gz",
        "QSM Chi-separation+R2' 5-Echo Myelin Map": "qsm/sub-*/ses-{ses}/anat/*_space-MNI152NLin2009cAsym_desc-E12345+chisep+r2p_myelinw.nii.gz",
        "QSM Chi-separation+R2' 4-Echo Myelin Map": "qsm/sub-*/ses-{ses}/anat/*_space-MNI152NLin2009cAsym_desc-E2345+chisep+r2p_myelinw.nii.gz",
        "QSM Chi-separation+R2pnet 5-Echo Chi Map": "qsm/sub-*/ses-{ses}/anat/*_space-MNI152NLin2009cAsym_desc-E12345+chisep+r2primenet_Chimap.nii.gz",
        "QSM Chi-separation+R2pnet 4-Echo Chi Map": "qsm/sub-*/ses-{ses}/anat/*_space-MNI152NLin2009cAsym_desc-E2345+chisep+r2primenet_Chimap.nii.gz",
        "QSM Chi-separation+R2pnet 5-Echo Iron Map": "qsm/sub-*/ses-{ses}/anat/*_space-MNI152NLin2009cAsym_desc-E12345+chisep+r2primenet_ironw.nii.gz",
        "QSM Chi-separation+R2pnet 4-Echo Iron Map": "qsm/sub-*/ses-{ses}/anat/*_space-MNI152NLin2009cAsym_desc-E2345+chisep+r2primenet_ironw.nii.gz",
        "QSM Chi-separation+R2pnet 5-Echo Myelin Map": "qsm/sub-*/ses-{ses}/anat/*_space-MNI152NLin2009cAsym_desc-E12345+chisep+r2primenet_myelinw.nii.gz",
        "QSM Chi-separation+R2pnet 4-Echo Myelin Map": "qsm/sub-*/ses-{ses}/anat/*_space-MNI152NLin2009cAsym_desc-E2345+chisep+r2primenet_myelinw.nii.gz",
        "QSM Chi-separation+R2* 5-Echo Chi Map": "qsm/sub-*/ses-{ses}/anat/*_space-MNI152NLin2009cAsym_desc-E12345+chisep+r2s_Chimap.nii.gz",
        "QSM Chi-separation+R2* 4-Echo Chi Map": "qsm/sub-*/ses-{ses}/anat/*_space-MNI152NLin2009cAsym_desc-E2345+chisep+r2s_Chimap.nii.gz",
        "QSM Chi-separation+R2* 5-Echo Iron Map": "qsm/sub-*/ses-{ses}/anat/*_space-MNI152NLin2009cAsym_desc-E12345+chisep+r2s_ironw.nii.gz",
        "QSM Chi-separation+R2* 4-Echo Iron Map": "qsm/sub-*/ses-{ses}/anat/*_space-MNI152NLin2009cAsym_desc-E2345+chisep+r2s_ironw.nii.gz",
        "QSM Chi-separation+R2* 5-Echo Myelin Map": "qsm/sub-*/ses-{ses}/anat/*_space-MNI152NLin2009cAsym_desc-E12345+chisep+r2s_myelinw.nii.gz",
        "QSM Chi-separation+R2* 4-Echo Myelin Map": "qsm/sub-*/ses-{ses}/anat/*_space-MNI152NLin2009cAsym_desc-E2345+chisep+r2s_myelinw.nii.gz",
    }
    for title, pattern in patterns.items():
        if 'G-Ratio' not in title:
            # Temporarily skip
            continue
        temp_pattern = pattern.format(ses='*')

        # Get all scalar maps
        scalar_maps = sorted(glob(os.path.join(in_dir, temp_pattern)))
        scalar_maps = [f for f in scalar_maps if "PILOT" not in f]
        print(f"{title}: {len(scalar_maps)}")

        # Mask out non-brain voxels
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mask_img = image.resample_to_img(mask, scalar_maps[0], interpolation="nearest")

        masker = maskers.NiftiMasker(mask_img, resampling_target="data")
        mean_img = image.mean_img(scalar_maps, copy_header=True)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mean_arr = masker.fit_transform(mean_img)

        # Get vmax (98th percentile) across both sessions
        mean_arr[np.isnan(mean_arr)] = 0
        mean_arr[np.isinf(mean_arr)] = 0
        vmax0 = np.percentile(mean_arr, 98)
        print(f"\t{vmax0}")
        if "Chi Map" in title:
            # Use two-directional colorbar
            kwargs = {'symmetric_cbar': True, 'vmin': None}
            vmin = -vmax0
        else:
            kwargs = {'symmetric_cbar': False, 'vmin': 0}
            vmin = 0

        session_mean_imgs = []
        for ses in ['01', '02']:
            temp_pattern = pattern.format(ses=ses)

            # Get all scalar maps
            scalar_maps = sorted(glob(os.path.join(in_dir, temp_pattern)))
            scalar_maps = [f for f in scalar_maps if "PILOT" not in f]
            print(f"\t{ses}: {len(scalar_maps)}")

            # Mask out non-brain voxels
            mean_img = image.mean_img(scalar_maps, copy_header=True)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                mean_arr = masker.transform(mean_img)
                mean_arr[np.isnan(mean_arr)] = 0
                mean_arr[np.isinf(mean_arr)] = 0
                mean_img = masker.inverse_transform(mean_arr)
                session_mean_imgs.append(mean_img)

        # Plot mean from each session
        fig, axs = plt.subplots(3, 1, figsize=(12.5, 5), height_ratios=[1, 1, 0.25])
        # Increase vertical space between the first two rows
        fig.subplots_adjust(hspace=0.5)
        plotting.plot_stat_map(
            session_mean_imgs[0],
            bg_img=template,
            display_mode="z",
            cut_coords=[-30, -15, 0, 15, 30, 45, 60],
            axes=axs[0],
            figure=fig,
            vmax=vmax0,
            cmap="viridis",
            annotate=False,
            black_bg=False,
            resampling_interpolation="nearest",
            colorbar=False,
            **kwargs,
        )
        axs[0].set_title("Session 01", fontsize=16)
        plotting.plot_stat_map(
            session_mean_imgs[1],
            bg_img=template,
            display_mode="z",
            cut_coords=[-30, -15, 0, 15, 30, 45, 60],
            axes=axs[1],
            figure=fig,
            vmax=vmax0,
            cmap="viridis",
            annotate=False,
            black_bg=False,
            resampling_interpolation="nearest",
            colorbar=False,
            **kwargs,
        )
        axs[1].set_title("Session 02", fontsize=16)

        # Plot the colorbars
        # Resize colorbar axis to be shorter and narrower
        cax = axs[2]
        pos = cax.get_position()
        new_width = pos.width * 0.75
        new_height = 0.05
        center_x = pos.x0 + pos.width / 2.0
        new_x0 = center_x - new_width / 2.0
        new_y0 = pos.y0 + (pos.height - new_height) / 2.0
        cax.set_position([new_x0, new_y0, new_width, new_height])

        cmap = mpl.cm.viridis

        norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax0)
        cbar = fig.colorbar(
            mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
            cax=cax,
            orientation='horizontal',
        )
        if vmin == 0:
            cbar.set_ticks([0, np.mean([0, vmax0]), vmax0])
        else:
            cbar.set_ticks([vmin, 0, vmax0])

        cbar.ax.tick_params(labelsize=12)

        fig.savefig(
            os.path.join(
                out_dir,
                f"{title.lower().replace('/', '_').replace(' ', '_').replace('*', 'star')}.png",
            ),
            bbox_inches="tight",
        )
        plt.close()
