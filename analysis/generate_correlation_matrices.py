"""Plot correlation matrices between myelin measures."""

import os
import warnings
from glob import glob

import nibabel as nb
import numpy as np
import pandas as pd
from nilearn import image, masking


if __name__ == "__main__":
    bids_dir = "/cbica/projects/nibs/dset"
    deriv_dir = "/cbica/projects/nibs/derivatives"
    out_dir = "../data"

    patterns = {
        # dMRI
        "Fractional Anisotropy": "qsirecon/derivatives/qsirecon-DIPYDKI/sub-{subject}/ses-01/dwi/*_space-MNI152NLin2009cAsym_model-tensor_param-fa_dwimap.nii.gz",
        "Axial Diffusivity": "qsirecon/derivatives/qsirecon-DIPYDKI/sub-{subject}/ses-01/dwi/*_space-MNI152NLin2009cAsym_model-dki_param-ad_dwimap.nii.gz",
        "Mean Diffusivity": "qsirecon/derivatives/qsirecon-DIPYDKI/sub-{subject}/ses-01/dwi/*_space-MNI152NLin2009cAsym_model-dki_param-md_dwimap.nii.gz",
        "Mean Kurtosis": "qsirecon/derivatives/qsirecon-DIPYDKI/sub-{subject}/ses-01/dwi/*_space-MNI152NLin2009cAsym_model-dki_param-mk_dwimap.nii.gz",
        "Radial Diffusivity": "qsirecon/derivatives/qsirecon-DIPYDKI/sub-{subject}/ses-01/dwi/*_space-MNI152NLin2009cAsym_model-dki_param-rd_dwimap.nii.gz",
        "Radial Kurtosis": "qsirecon/derivatives/qsirecon-DIPYDKI/sub-{subject}/ses-01/dwi/*_space-MNI152NLin2009cAsym_model-dki_param-rk_dwimap.nii.gz",
        # ihMT
        "Inhomogeneous Magnetization Transfer-Weighted": "ihmt/sub-{subject}/ses-01/anat/*_space-MNI152NLin2009cAsym_ihMTw.nii.gz",
        "Inhomogeneous Magnetization Transfer Ratio": "ihmt/sub-{subject}/ses-01/anat/*_space-MNI152NLin2009cAsym_ihMTR.nii.gz",
        "Magnetization Transfer Ratio": "ihmt/sub-{subject}/ses-01/anat/*_space-MNI152NLin2009cAsym_MTRmap.nii.gz",
        "Inhomogeneous Magnetization Transfer Saturation": "ihmt/sub-{subject}/ses-01/anat/*_space-MNI152NLin2009cAsym_ihMTsat.nii.gz",
        "B1-Corrected Inhomogeneous Magnetization Transfer Saturation": "ihmt/sub-{subject}/ses-01/anat/*_space-MNI152NLin2009cAsym_ihMTsatB1sq.nii.gz",
        # MP2RAGE
        "R1": "pymp2rage/sub-{subject}/ses-01/anat/*_space-MNI152NLin2009cAsym_R1map.nii.gz",
        "B1-Corrected R1": "pymp2rage/sub-{subject}/ses-01/anat/*_space-MNI152NLin2009cAsym_desc-B1corrected_R1map.nii.gz",
        # T1w/T2w Ratio
        "MPRAGE T1w/SPACE T2w Ratio": "t1wt2w_ratio/sub-{subject}/ses-01/anat/*_space-MNI152NLin2009cAsym_desc-MPRAGEunscaled_myelinw.nii.gz",
        "SPACE T1w/SPACE T2w Ratio": "t1wt2w_ratio/sub-{subject}/ses-01/anat/*_space-MNI152NLin2009cAsym_desc-SPACEunscaled_myelinw.nii.gz",
        "Scaled MPRAGE T1w/SPACE T2w Ratio": "t1wt2w_ratio/sub-{subject}/ses-01/anat/*_space-MNI152NLin2009cAsym_desc-MPRAGEscaled_myelinw.nii.gz",
        "Scaled SPACE T1w/SPACE T2w Ratio": "t1wt2w_ratio/sub-{subject}/ses-01/anat/*_space-MNI152NLin2009cAsym_desc-SPACEscaled_myelinw.nii.gz",
    }
    gm_idx = 1
    wm_idx = 2

    wb_corr_mats = []
    gm_corr_mats = []
    wm_corr_mats = []
    for subject in sorted(glob(os.path.join(bids_dir, 'sub-*'))):
        subject = os.path.basename(subject)
        subject = subject.split('-')[1]
        if subject.startswith('PILOT'):
            continue

        dseg = os.path.join(
            deriv_dir,
            'smriprep',
            f'sub-{subject}',
            'anat',
            f'sub-{subject}_acq-MPRAGE_rec-refaced_run-01_space-MNI152NLin2009cAsym_dseg.nii.gz',
        )
        if not os.path.isfile(dseg):
            # Try session-wise
            dseg = os.path.join(
                deriv_dir,
                'smriprep',
                f'sub-{subject}',
                'ses-01',
                'anat',
                f'sub-{subject}_ses-01_acq-MPRAGE_rec-refaced_run-01_space-MNI152NLin2009cAsym_dseg.nii.gz',
            )
            if not os.path.isfile(dseg):
                print(f'No dseg found for {subject}')
                continue

        dseg_img = nb.load(dseg)
        dseg_data = dseg_img.get_fdata()
        wm_mask = (dseg_data == wm_idx).astype(int)
        wm_img = nb.Nifti1Image(wm_mask, dseg_img.affine, dseg_img.header)
        gm_mask = (dseg_data == gm_idx).astype(int)
        gm_img = nb.Nifti1Image(gm_mask, dseg_img.affine, dseg_img.header)
        wb_mask = wm_mask + gm_mask
        wb_img = nb.Nifti1Image(wb_mask, dseg_img.affine, dseg_img.header)
        wb_n_voxels = np.sum(wb_mask)
        gm_n_voxels = np.sum(gm_mask)
        wm_n_voxels = np.sum(wm_mask)

        print(os.path.basename(dseg))
        wb_arr = np.zeros((len(patterns), wb_n_voxels))
        gm_arr = np.zeros((len(patterns), gm_n_voxels))
        wm_arr = np.zeros((len(patterns), wm_n_voxels))
        for i_file, (title, pattern) in enumerate(patterns.items()):
            pattern = pattern.format(subject=subject)
            files = sorted(glob(os.path.join(deriv_dir, pattern)))
            if len(files) == 0:
                print(f"No files found for {pattern}")
                wb_arr[i_file, :] = np.nan
                gm_arr[i_file, :] = np.nan
                wm_arr[i_file, :] = np.nan
                continue
            elif len(files) != 1:
                print(f"Multiple files found for {pattern}")
                wb_arr[i_file, :] = np.nan
                gm_arr[i_file, :] = np.nan
                wm_arr[i_file, :] = np.nan
                continue
            else:
                # Resample image to same resolution as dseg
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    img = image.resample_to_img(files[0], dseg_img, interpolation="nearest")
                wb_arr[i_file, :] = masking.apply_mask(img, wb_img)
                gm_arr[i_file, :] = masking.apply_mask(img, gm_img)
                wm_arr[i_file, :] = masking.apply_mask(img, wm_img)

        # Calculate correlation matrices
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            wb_corr_mat = np.atanh(np.corrcoef(wb_arr))
            gm_corr_mat = np.atanh(np.corrcoef(gm_arr))
            wm_corr_mat = np.atanh(np.corrcoef(wm_arr))

        del wb_arr, gm_arr, wm_arr
        wb_corr_mats.append(wb_corr_mat)
        gm_corr_mats.append(gm_corr_mat)
        wm_corr_mats.append(wm_corr_mat)

    mean_wb_corr_mat = np.nanmean(np.stack(wb_corr_mats), axis=0)
    mean_gm_corr_mat = np.nanmean(np.stack(gm_corr_mats), axis=0)
    mean_wm_corr_mat = np.nanmean(np.stack(wm_corr_mats), axis=0)
    mean_wb_df = pd.DataFrame(mean_wb_corr_mat, columns=patterns.keys(), index=patterns.keys())
    mean_gm_df = pd.DataFrame(mean_gm_corr_mat, columns=patterns.keys(), index=patterns.keys())
    mean_wm_df = pd.DataFrame(mean_wm_corr_mat, columns=patterns.keys(), index=patterns.keys())
    mean_wb_df.to_csv(os.path.join(out_dir, 'mean_wb_corr_mat.tsv'), sep='\t', index=True, index_label='Image')
    mean_gm_df.to_csv(os.path.join(out_dir, 'mean_gm_corr_mat.tsv'), sep='\t', index=True, index_label='Image')
    mean_wm_df.to_csv(os.path.join(out_dir, 'mean_wm_corr_mat.tsv'), sep='\t', index=True, index_label='Image')

    print(f"Found {len(wb_corr_mats)} correlation matrices")
