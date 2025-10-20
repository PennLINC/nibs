"""Plot correlation matrices between myelin measures."""

import os
import warnings
from glob import glob

import ants
import nibabel as nb
import numpy as np
import pandas as pd
from nilearn import masking


if __name__ == "__main__":
    bids_dir = "/cbica/projects/nibs/dset"
    deriv_dir = "/cbica/projects/nibs/derivatives"
    temp_dir = "/cbica/projects/nibs/work/correlation_matrices"
    os.makedirs(temp_dir, exist_ok=True)
    out_dir = "../data"

    patterns = {
        # dMRI
        "dMRI": {
            "DKI Micro AD": "qsirecon/derivatives/qsirecon-DIPYDKI/sub-{subject}/ses-01/dwi/*_space-MNI152NLin2009cAsym_model-dkimicro_param-ad_dwimap.nii.gz",
            "DKI Micro ADE": "qsirecon/derivatives/qsirecon-DIPYDKI/sub-{subject}/ses-01/dwi/*_space-MNI152NLin2009cAsym_model-dkimicro_param-ade_dwimap.nii.gz",
            "DKI Micro AWF": "qsirecon/derivatives/qsirecon-DIPYDKI/sub-{subject}/ses-01/dwi/*_space-MNI152NLin2009cAsym_model-dkimicro_param-awf_dwimap.nii.gz",
            "DKI Micro AxonalD": "qsirecon/derivatives/qsirecon-DIPYDKI/sub-{subject}/ses-01/dwi/*_space-MNI152NLin2009cAsym_model-dkimicro_param-axonald_dwimap.nii.gz",
            "DKI Micro KFA": "qsirecon/derivatives/qsirecon-DIPYDKI/sub-{subject}/ses-01/dwi/*_space-MNI152NLin2009cAsym_model-dkimicro_param-kfa_dwimap.nii.gz",
            "DKI Micro MD": "qsirecon/derivatives/qsirecon-DIPYDKI/sub-{subject}/ses-01/dwi/*_space-MNI152NLin2009cAsym_model-dkimicro_param-md_dwimap.nii.gz",
            "DKI Micro RD": "qsirecon/derivatives/qsirecon-DIPYDKI/sub-{subject}/ses-01/dwi/*_space-MNI152NLin2009cAsym_model-dkimicro_param-rd_dwimap.nii.gz",
            "DKI Micro RDE": "qsirecon/derivatives/qsirecon-DIPYDKI/sub-{subject}/ses-01/dwi/*_space-MNI152NLin2009cAsym_model-dkimicro_param-rde_dwimap.nii.gz",
            "DKI Micro Tortuosity": "qsirecon/derivatives/qsirecon-DIPYDKI/sub-{subject}/ses-01/dwi/*_space-MNI152NLin2009cAsym_model-dkimicro_param-tortuosity_dwimap.nii.gz",
            "DKI Micro Trace": "qsirecon/derivatives/qsirecon-DIPYDKI/sub-{subject}/ses-01/dwi/*_space-MNI152NLin2009cAsym_model-dkimicro_param-trace_dwimap.nii.gz",
            "DKI AD": "qsirecon/derivatives/qsirecon-DIPYDKI/sub-{subject}/ses-01/dwi/*_space-MNI152NLin2009cAsym_model-dki_param-ad_dwimap.nii.gz",
            "DKI AK": "qsirecon/derivatives/qsirecon-DIPYDKI/sub-{subject}/ses-01/dwi/*_space-MNI152NLin2009cAsym_model-dki_param-ak_dwimap.nii.gz",
            "DKI KFA": "qsirecon/derivatives/qsirecon-DIPYDKI/sub-{subject}/ses-01/dwi/*_space-MNI152NLin2009cAsym_model-dki_param-kfa_dwimap.nii.gz",
            "DKI Linearity": "qsirecon/derivatives/qsirecon-DIPYDKI/sub-{subject}/ses-01/dwi/*_space-MNI152NLin2009cAsym_model-dki_param-linearity_dwimap.nii.gz",
            "DKI MD": "qsirecon/derivatives/qsirecon-DIPYDKI/sub-{subject}/ses-01/dwi/*_space-MNI152NLin2009cAsym_model-dki_param-md_dwimap.nii.gz",
            "DKI MK": "qsirecon/derivatives/qsirecon-DIPYDKI/sub-{subject}/ses-01/dwi/*_space-MNI152NLin2009cAsym_model-dki_param-mk_dwimap.nii.gz",
            "DKI MKT": "qsirecon/derivatives/qsirecon-DIPYDKI/sub-{subject}/ses-01/dwi/*_space-MNI152NLin2009cAsym_model-dki_param-mkt_dwimap.nii.gz",
            "DKI Planarity": "qsirecon/derivatives/qsirecon-DIPYDKI/sub-{subject}/ses-01/dwi/*_space-MNI152NLin2009cAsym_model-dki_param-planarity_dwimap.nii.gz",
            "DKI RD": "qsirecon/derivatives/qsirecon-DIPYDKI/sub-{subject}/ses-01/dwi/*_space-MNI152NLin2009cAsym_model-dki_param-rd_dwimap.nii.gz",
            "DKI RK": "qsirecon/derivatives/qsirecon-DIPYDKI/sub-{subject}/ses-01/dwi/*_space-MNI152NLin2009cAsym_model-dki_param-rk_dwimap.nii.gz",
            "DKI Sphericity": "qsirecon/derivatives/qsirecon-DIPYDKI/sub-{subject}/ses-01/dwi/*_space-MNI152NLin2009cAsym_model-dki_param-sphericity_dwimap.nii.gz",
            "DKI Tensor FA": "qsirecon/derivatives/qsirecon-DIPYDKI/sub-{subject}/ses-01/dwi/*_space-MNI152NLin2009cAsym_model-tensor_param-fa_dwimap.nii.gz",
            "DSIStudio GQI GFA": "qsirecon/derivatives/qsirecon-DSIStudio/sub-{subject}/ses-01/dwi/*_space-MNI152NLin2009cAsym_model-gqi_param-gfa_dwimap.nii.gz",
            "DSIStudio GQI ISO": "qsirecon/derivatives/qsirecon-DSIStudio/sub-{subject}/ses-01/dwi/*_space-MNI152NLin2009cAsym_model-gqi_param-iso_dwimap.nii.gz",
            "DSIStudio GQI QA": "qsirecon/derivatives/qsirecon-DSIStudio/sub-{subject}/ses-01/dwi/*_space-MNI152NLin2009cAsym_model-gqi_param-qa_dwimap.nii.gz",
            "DSIStudio Tensor AD": "qsirecon/derivatives/qsirecon-DSIStudio/sub-{subject}/ses-01/dwi/*_space-MNI152NLin2009cAsym_model-tensor_param-ad_dwimap.nii.gz",
            "DSIStudio Tensor FA": "qsirecon/derivatives/qsirecon-DSIStudio/sub-{subject}/ses-01/dwi/*_space-MNI152NLin2009cAsym_model-tensor_param-fa_dwimap.nii.gz",
            "DSIStudio Tensor HA": "qsirecon/derivatives/qsirecon-DSIStudio/sub-{subject}/ses-01/dwi/*_space-MNI152NLin2009cAsym_model-tensor_param-ha_dwimap.nii.gz",
            "DSIStudio Tensor MD": "qsirecon/derivatives/qsirecon-DSIStudio/sub-{subject}/ses-01/dwi/*_space-MNI152NLin2009cAsym_model-tensor_param-md_dwimap.nii.gz",
            "DSIStudio Tensor RD1": "qsirecon/derivatives/qsirecon-DSIStudio/sub-{subject}/ses-01/dwi/*_space-MNI152NLin2009cAsym_model-tensor_param-rd1_dwimap.nii.gz",
            "DSIStudio Tensor RD2": "qsirecon/derivatives/qsirecon-DSIStudio/sub-{subject}/ses-01/dwi/*_space-MNI152NLin2009cAsym_model-tensor_param-rd2_dwimap.nii.gz",
            "DSIStudio Tensor RD": "qsirecon/derivatives/qsirecon-DSIStudio/sub-{subject}/ses-01/dwi/*_space-MNI152NLin2009cAsym_model-tensor_param-rd_dwimap.nii.gz",
            "DSIStudio Tensor TXX": "qsirecon/derivatives/qsirecon-DSIStudio/sub-{subject}/ses-01/dwi/*_space-MNI152NLin2009cAsym_model-tensor_param-txx_dwimap.nii.gz",
            "DSIStudio Tensor TXY": "qsirecon/derivatives/qsirecon-DSIStudio/sub-{subject}/ses-01/dwi/*_space-MNI152NLin2009cAsym_model-tensor_param-txy_dwimap.nii.gz",
            "DSIStudio Tensor TXZ": "qsirecon/derivatives/qsirecon-DSIStudio/sub-{subject}/ses-01/dwi/*_space-MNI152NLin2009cAsym_model-tensor_param-txz_dwimap.nii.gz",
            "DSIStudio Tensor TYY": "qsirecon/derivatives/qsirecon-DSIStudio/sub-{subject}/ses-01/dwi/*_space-MNI152NLin2009cAsym_model-tensor_param-tyy_dwimap.nii.gz",
            "DSIStudio Tensor TYZ": "qsirecon/derivatives/qsirecon-DSIStudio/sub-{subject}/ses-01/dwi/*_space-MNI152NLin2009cAsym_model-tensor_param-tyz_dwimap.nii.gz",
            "DSIStudio Tensor TZZ": "qsirecon/derivatives/qsirecon-DSIStudio/sub-{subject}/ses-01/dwi/*_space-MNI152NLin2009cAsym_model-tensor_param-tzz_dwimap.nii.gz",
            "NODDI ICVF Modulated": "qsirecon/derivatives/qsirecon-NODDI/sub-{subject}/ses-01/dwi/*_space-MNI152NLin2009cAsym_model-noddi_param-icvf_desc-modulated_dwimap.nii.gz",
            "NODDI ICVF": "qsirecon/derivatives/qsirecon-NODDI/sub-{subject}/ses-01/dwi/*_space-MNI152NLin2009cAsym_model-noddi_param-icvf_dwimap.nii.gz",
            "NODDI ISOVF": "qsirecon/derivatives/qsirecon-NODDI/sub-{subject}/ses-01/dwi/*_space-MNI152NLin2009cAsym_model-noddi_param-isovf_dwimap.nii.gz",
            "NODDI NRMSE": "qsirecon/derivatives/qsirecon-NODDI/sub-{subject}/ses-01/dwi/*_space-MNI152NLin2009cAsym_model-noddi_param-nrmse_dwimap.nii.gz",
            "NODDI OD Modulated": "qsirecon/derivatives/qsirecon-NODDI/sub-{subject}/ses-01/dwi/*_space-MNI152NLin2009cAsym_model-noddi_param-od_desc-modulated_dwimap.nii.gz",
            "NODDI OD": "qsirecon/derivatives/qsirecon-NODDI/sub-{subject}/ses-01/dwi/*_space-MNI152NLin2009cAsym_model-noddi_param-od_dwimap.nii.gz",
            "NODDI RMSE": "qsirecon/derivatives/qsirecon-NODDI/sub-{subject}/ses-01/dwi/*_space-MNI152NLin2009cAsym_model-noddi_param-rmse_dwimap.nii.gz",
            "NODDI TF": "qsirecon/derivatives/qsirecon-NODDI/sub-{subject}/ses-01/dwi/*_space-MNI152NLin2009cAsym_model-noddi_param-tf_dwimap.nii.gz",
            "TORTOISE MAPMRI NG": "qsirecon/derivatives/qsirecon-TORTOISE_model-MAPMRI/sub-{subject}/ses-01/dwi/*_space-MNI152NLin2009cAsym_model-mapmri_param-ng_dwimap.nii.gz",
            "TORTOISE MAPMRI NG Par": "qsirecon/derivatives/qsirecon-TORTOISE_model-MAPMRI/sub-{subject}/ses-01/dwi/*_space-MNI152NLin2009cAsym_model-mapmri_param-ngpar_dwimap.nii.gz",
            "TORTOISE MAPMRI NG Perp": "qsirecon/derivatives/qsirecon-TORTOISE_model-MAPMRI/sub-{subject}/ses-01/dwi/*_space-MNI152NLin2009cAsym_model-mapmri_param-ngperp_dwimap.nii.gz",
            "TORTOISE MAPMRI PA": "qsirecon/derivatives/qsirecon-TORTOISE_model-MAPMRI/sub-{subject}/ses-01/dwi/*_space-MNI152NLin2009cAsym_model-mapmri_param-pa_dwimap.nii.gz",
            "TORTOISE MAPMRI Path": "qsirecon/derivatives/qsirecon-TORTOISE_model-MAPMRI/sub-{subject}/ses-01/dwi/*_space-MNI152NLin2009cAsym_model-mapmri_param-path_dwimap.nii.gz",
            "TORTOISE MAPMRI RTA": "qsirecon/derivatives/qsirecon-TORTOISE_model-MAPMRI/sub-{subject}/ses-01/dwi/*_space-MNI152NLin2009cAsym_model-mapmri_param-rtap_dwimap.nii.gz",
            "TORTOISE MAPMRI RTO": "qsirecon/derivatives/qsirecon-TORTOISE_model-MAPMRI/sub-{subject}/ses-01/dwi/*_space-MNI152NLin2009cAsym_model-mapmri_param-rtop_dwimap.nii.gz",
            "TORTOISE MAPMRI RTP": "qsirecon/derivatives/qsirecon-TORTOISE_model-MAPMRI/sub-{subject}/ses-01/dwi/*_space-MNI152NLin2009cAsym_model-mapmri_param-rtpp_dwimap.nii.gz",
            "TORTOISE MAPMRI AD": "qsirecon/derivatives/qsirecon-TORTOISE_model-MAPMRI/sub-{subject}/ses-01/dwi/*_space-MNI152NLin2009cAsym_model-tensor_param-ad_dwimap.nii.gz",
            "TORTOISE MAPMRI AM": "qsirecon/derivatives/qsirecon-TORTOISE_model-MAPMRI/sub-{subject}/ses-01/dwi/*_space-MNI152NLin2009cAsym_model-tensor_param-am_dwimap.nii.gz",
            "TORTOISE MAPMRI FA": "qsirecon/derivatives/qsirecon-TORTOISE_model-MAPMRI/sub-{subject}/ses-01/dwi/*_space-MNI152NLin2009cAsym_model-tensor_param-fa_dwimap.nii.gz",
            "TORTOISE MAPMRI LI": "qsirecon/derivatives/qsirecon-TORTOISE_model-MAPMRI/sub-{subject}/ses-01/dwi/*_space-MNI152NLin2009cAsym_model-tensor_param-li_dwimap.nii.gz",
            "TORTOISE MAPMRI RD": "qsirecon/derivatives/qsirecon-TORTOISE_model-MAPMRI/sub-{subject}/ses-01/dwi/*_space-MNI152NLin2009cAsym_model-tensor_param-rd_dwimap.nii.gz",
            "TORTOISE Tensor AD": "qsirecon/derivatives/qsirecon-TORTOISE_model-tensor/sub-{subject}/ses-01/dwi/*_space-MNI152NLin2009cAsym_model-tensor_param-ad_dwimap.nii.gz",
            "TORTOISE Tensor AM": "qsirecon/derivatives/qsirecon-TORTOISE_model-tensor/sub-{subject}/ses-01/dwi/*_space-MNI152NLin2009cAsym_model-tensor_param-am_dwimap.nii.gz",
            "TORTOISE Tensor FA": "qsirecon/derivatives/qsirecon-TORTOISE_model-tensor/sub-{subject}/ses-01/dwi/*_space-MNI152NLin2009cAsym_model-tensor_param-fa_dwimap.nii.gz",
            "TORTOISE Tensor LI": "qsirecon/derivatives/qsirecon-TORTOISE_model-tensor/sub-{subject}/ses-01/dwi/*_space-MNI152NLin2009cAsym_model-tensor_param-li_dwimap.nii.gz",
            "TORTOISE Tensor RD": "qsirecon/derivatives/qsirecon-TORTOISE_model-tensor/sub-{subject}/ses-01/dwi/*_space-MNI152NLin2009cAsym_model-tensor_param-rd_dwimap.nii.gz",
        },
        # ihMT
        "ihMT": {
            "ihMTw": "ihmt/sub-{subject}/ses-01/anat/*_space-MNI152NLin2009cAsym_ihMTw.nii.gz",
            "ihMTR": "ihmt/sub-{subject}/ses-01/anat/*_space-MNI152NLin2009cAsym_ihMTR.nii.gz",
            "MTR": "ihmt/sub-{subject}/ses-01/anat/*_space-MNI152NLin2009cAsym_MTRmap.nii.gz",
            "ihMTsat": "ihmt/sub-{subject}/ses-01/anat/*_space-MNI152NLin2009cAsym_ihMTsat.nii.gz",
            "ihMTsat-B1c": "ihmt/sub-{subject}/ses-01/anat/*_space-MNI152NLin2009cAsym_ihMTsatB1sq.nii.gz",
        },
        # MP2RAGE
        "MP2RAGE": {
            "R1": "pymp2rage/sub-{subject}/ses-01/anat/*_space-MNI152NLin2009cAsym_R1map.nii.gz",
            "R1-B1c": "pymp2rage/sub-{subject}/ses-01/anat/*_space-MNI152NLin2009cAsym_desc-B1corrected_R1map.nii.gz",
        },
        # T1w/T2w Ratio
        "T1w/T2w Ratio": {
            "MPRAGE-MyelinW": "t1wt2w_ratio/sub-{subject}/ses-01/anat/*_space-MNI152NLin2009cAsym_desc-MPRAGEunscaled_myelinw.nii.gz",
            "SPACE-MyelinW": "t1wt2w_ratio/sub-{subject}/ses-01/anat/*_space-MNI152NLin2009cAsym_desc-SPACEunscaled_myelinw.nii.gz",
            "Scaled MPRAGE-MyelinW": "t1wt2w_ratio/sub-{subject}/ses-01/anat/*_space-MNI152NLin2009cAsym_desc-MPRAGEscaled_myelinw.nii.gz",
            "Scaled SPACE-MyelinW": "t1wt2w_ratio/sub-{subject}/ses-01/anat/*_space-MNI152NLin2009cAsym_desc-SPACEscaled_myelinw.nii.gz",
        },
        # G-Ratio
        "G-Ratio": {
            "G-MPRAGE-MyelinW": "g_ratio/sub-{subject}/ses-01/anat/*_space-MNI152NLin2009cAsym_desc-MPRAGET1wT2w+ISOVF+ICVF_gratio.nii.gz",
            "G-SPACE-MyelinW": "g_ratio/sub-{subject}/ses-01/anat/*_space-MNI152NLin2009cAsym_desc-SPACET1wT2w+ISOVF+ICVF_gratio.nii.gz",
            "G-ihMTsat": "g_ratio/sub-{subject}/ses-01/anat/*_space-MNI152NLin2009cAsym_desc-MTsat+ISOVF+ICVF_gratio.nii.gz",
            "G-ihMTR": "g_ratio/sub-{subject}/ses-01/anat/*_space-MNI152NLin2009cAsym_desc-ihMTR+ISOVF+ICVF_gratio.nii.gz",
        },
        # QSM
        "QSM": {
            "QSM-SEPIA-E5": "qsm/sub-{subject}/ses-01/anat/*_space-MNI152NLin2009cAsym_desc-E12345+sepia_Chimap.nii.gz",
            "QSM-SEPIA-E4": "qsm/sub-{subject}/ses-01/anat/*_space-MNI152NLin2009cAsym_desc-E2345+sepia_Chimap.nii.gz",
            "QSM-X-R2'-E5-Chi": "qsm/sub-{subject}/ses-01/anat/*_space-MNI152NLin2009cAsym_desc-E12345+chisep+r2p_Chimap.nii.gz",
            "QSM-X-R2'-E4-Chi": "qsm/sub-{subject}/ses-01/anat/*_space-MNI152NLin2009cAsym_desc-E2345+chisep+r2p_Chimap.nii.gz",
            "QSM-X-R2'-E5-Iron": "qsm/sub-{subject}/ses-01/anat/*_space-MNI152NLin2009cAsym_desc-E12345+chisep+r2p_ironw.nii.gz",
            "QSM-X-R2'-E4-Iron": "qsm/sub-{subject}/ses-01/anat/*_space-MNI152NLin2009cAsym_desc-E2345+chisep+r2p_ironw.nii.gz",
            "QSM-X-R2'-E5-Myelin": "qsm/sub-{subject}/ses-01/anat/*_space-MNI152NLin2009cAsym_desc-E12345+chisep+r2p_myelinw.nii.gz",
            "QSM-X-R2'-E4-Myelin": "qsm/sub-{subject}/ses-01/anat/*_space-MNI152NLin2009cAsym_desc-E2345+chisep+r2p_myelinw.nii.gz",
            "QSM-X-R2pnet-E5-Chi": "qsm/sub-{subject}/ses-01/anat/*_space-MNI152NLin2009cAsym_desc-E12345+chisep+r2primenet_Chimap.nii.gz",
            "QSM-X-R2pnet-E4-Chi": "qsm/sub-{subject}/ses-01/anat/*_space-MNI152NLin2009cAsym_desc-E2345+chisep+r2primenet_Chimap.nii.gz",
            "QSM-X-R2pnet-E5-Iron": "qsm/sub-{subject}/ses-01/anat/*_space-MNI152NLin2009cAsym_desc-E12345+chisep+r2primenet_ironw.nii.gz",
            "QSM-X-R2pnet-E4-Iron": "qsm/sub-{subject}/ses-01/anat/*_space-MNI152NLin2009cAsym_desc-E2345+chisep+r2primenet_ironw.nii.gz",
            "QSM-X-R2pnet-E5-Myelin": "qsm/sub-{subject}/ses-01/anat/*_space-MNI152NLin2009cAsym_desc-E12345+chisep+r2primenet_myelinw.nii.gz",
            "QSM-X-R2pnet-E4-Myelin": "qsm/sub-{subject}/ses-01/anat/*_space-MNI152NLin2009cAsym_desc-E2345+chisep+r2primenet_myelinw.nii.gz",
            "QSM-X-R2*-E5-Chi": "qsm/sub-{subject}/ses-01/anat/*_space-MNI152NLin2009cAsym_desc-E12345+chisep+r2s_Chimap.nii.gz",
            "QSM-X-R2*-E4-Chi": "qsm/sub-{subject}/ses-01/anat/*_space-MNI152NLin2009cAsym_desc-E2345+chisep+r2s_Chimap.nii.gz",
            "QSM-X-R2*-E5-Iron": "qsm/sub-{subject}/ses-01/anat/*_space-MNI152NLin2009cAsym_desc-E12345+chisep+r2s_ironw.nii.gz",
            "QSM-X-R2*-E4-Iron": "qsm/sub-{subject}/ses-01/anat/*_space-MNI152NLin2009cAsym_desc-E2345+chisep+r2s_ironw.nii.gz",
            "QSM-X-R2*-E5-Myelin": "qsm/sub-{subject}/ses-01/anat/*_space-MNI152NLin2009cAsym_desc-E12345+chisep+r2s_myelinw.nii.gz",
            "QSM-X-R2*-E4-Myelin": "qsm/sub-{subject}/ses-01/anat/*_space-MNI152NLin2009cAsym_desc-E2345+chisep+r2s_myelinw.nii.gz",
        },
    }
    scalar_names = [list(v) for v in patterns.values()]
    scalar_names = [item for sublist in scalar_names for item in sublist]
    n_scalars = len(scalar_names)

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

        print(f'Processing {subject}', flush=True)
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
                print(f'No dseg found for {subject}', flush=True)
                continue

        dseg_ants_img = ants.image_read(dseg)
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

        wb_arr = np.zeros((n_scalars, wb_n_voxels))
        gm_arr = np.zeros((n_scalars, gm_n_voxels))
        wm_arr = np.zeros((n_scalars, wm_n_voxels))

        scalar_counter = -1
        for i_modality, (modality, modality_patterns) in enumerate(patterns.items()):
            for scalar_name, scalar_pattern in modality_patterns.items():
                pattern = scalar_pattern.format(subject=subject)
                files = sorted(glob(os.path.join(deriv_dir, pattern)))
                scalar_counter += 1
                if len(files) == 0:
                    print(f"No files found for {pattern}", flush=True)
                    wb_arr[scalar_counter, :] = np.nan
                    gm_arr[scalar_counter, :] = np.nan
                    wm_arr[scalar_counter, :] = np.nan
                    continue
                elif len(files) != 1:
                    print(f"Multiple files found for {pattern}", flush=True)
                    wb_arr[scalar_counter, :] = np.nan
                    gm_arr[scalar_counter, :] = np.nan
                    wm_arr[scalar_counter, :] = np.nan
                    continue
                else:
                    # Resample image to same resolution as dseg
                    resampled_ants_img = ants.image_read(files[0]).resample_image_to_target(
                        dseg_ants_img,
                        interp_type='lanczosWindowedSinc',
                    )
                    resampled_file = os.path.join(temp_dir, f'{subject}_{scalar_name}.nii.gz')
                    ants.image_write(resampled_ants_img, resampled_file)

                    img = nb.load(resampled_file)
                    scalar_wb_arr = masking.apply_mask(img, wb_img)
                    if scalar_wb_arr.ndim != 1:
                        print(
                            f"Scalar {scalar_name} has {scalar_wb_arr.ndim} dimensions",
                            flush=True,
                        )
                        wb_arr[scalar_counter, :] = np.nan
                        gm_arr[scalar_counter, :] = np.nan
                        wm_arr[scalar_counter, :] = np.nan
                        continue

                    wb_arr[scalar_counter, :] = scalar_wb_arr
                    gm_arr[scalar_counter, :] = masking.apply_mask(img, gm_img)
                    wm_arr[scalar_counter, :] = masking.apply_mask(img, wm_img)

                    os.remove(resampled_file)

        # Calculate correlation matrices
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            wb_corr_mat = np.arctanh(np.corrcoef(wb_arr))
            gm_corr_mat = np.arctanh(np.corrcoef(gm_arr))
            wm_corr_mat = np.arctanh(np.corrcoef(wm_arr))

        del wb_arr, gm_arr, wm_arr
        wb_corr_mats.append(wb_corr_mat)
        gm_corr_mats.append(gm_corr_mat)
        wm_corr_mats.append(wm_corr_mat)

    mean_wb_corr_mat = np.nanmean(np.stack(wb_corr_mats), axis=0)
    mean_gm_corr_mat = np.nanmean(np.stack(gm_corr_mats), axis=0)
    mean_wm_corr_mat = np.nanmean(np.stack(wm_corr_mats), axis=0)
    mean_wb_df = pd.DataFrame(mean_wb_corr_mat, columns=scalar_names, index=scalar_names)
    mean_gm_df = pd.DataFrame(mean_gm_corr_mat, columns=scalar_names, index=scalar_names)
    mean_wm_df = pd.DataFrame(mean_wm_corr_mat, columns=scalar_names, index=scalar_names)
    mean_wb_df.to_csv(
        os.path.join(out_dir, 'mean_wb_corr_mat.tsv'),
        sep='\t',
        index=True,
        index_label='Image',
    )
    mean_gm_df.to_csv(
        os.path.join(out_dir, 'mean_gm_corr_mat.tsv'),
        sep='\t',
        index=True,
        index_label='Image',
    )
    mean_wm_df.to_csv(
        os.path.join(out_dir, 'mean_wm_corr_mat.tsv'),
        sep='\t',
        index=True,
        index_label='Image',
    )

    print(f"Found {len(wb_corr_mats)} correlation matrices")
