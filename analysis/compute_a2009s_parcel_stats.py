"""Compute parcel-wise summary statistics and coverage for scalar maps.

Runs per subject, writing one statistics CSV and one long-format coverage CSV
per subject/session/run for aparc.a2009s parcels. Scalar maps remain in their
native grids; the corresponding label image is resampled to each map with
generic-label interpolation.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from collections import defaultdict
from glob import glob

import ants
import numpy as np
import pandas as pd

PATTERNS_SUBJECT: dict[str, str] = {
    # DWI DKI
    "DKI-FA": "qsirecon/derivatives/qsirecon-DIPYDKI/sub-*/ses-*/dwi/sub-*_ses-*_acq-HBCD75_run-01_space-ACPC_model-dki_param-fa_dwimap.nii.gz",
    "DKI-KFA": "qsirecon/derivatives/qsirecon-DIPYDKI/sub-*/ses-*/dwi/sub-*_ses-*_acq-HBCD75_run-01_space-ACPC_model-dki_param-kfa_dwimap.nii.gz",
    "DKI-KFA-Micro": "qsirecon/derivatives/qsirecon-DIPYDKI/sub-*/ses-*/dwi/sub-*_ses-*_acq-HBCD75_run-01_space-ACPC_model-dkimicro_param-kfa_dwimap.nii.gz",
    "DKI-AD": "qsirecon/derivatives/qsirecon-DIPYDKI/sub-*/ses-*/dwi/sub-*_ses-*_acq-HBCD75_run-01_space-ACPC_model-dki_param-ad_dwimap.nii.gz",
    "DKI-AD-Micro": "qsirecon/derivatives/qsirecon-DIPYDKI/sub-*/ses-*/dwi/sub-*_ses-*_acq-HBCD75_run-01_space-ACPC_model-dkimicro_param-ad_dwimap.nii.gz",
    "DKI-ADE-Micro": "qsirecon/derivatives/qsirecon-DIPYDKI/sub-*/ses-*/dwi/sub-*_ses-*_acq-HBCD75_run-01_space-ACPC_model-dkimicro_param-ade_dwimap.nii.gz",
    "DKI-AWF-Micro": "qsirecon/derivatives/qsirecon-DIPYDKI/sub-*/ses-*/dwi/sub-*_ses-*_acq-HBCD75_run-01_space-ACPC_model-dkimicro_param-awf_dwimap.nii.gz",
    "DKI-AxonALD-Micro": "qsirecon/derivatives/qsirecon-DIPYDKI/sub-*/ses-*/dwi/sub-*_ses-*_acq-HBCD75_run-01_space-ACPC_model-dkimicro_param-axonald_dwimap.nii.gz",
    "DKI-AK": "qsirecon/derivatives/qsirecon-DIPYDKI/sub-*/ses-*/dwi/sub-*_ses-*_acq-HBCD75_run-01_space-ACPC_model-dki_param-ak_dwimap.nii.gz",
    "DKI-MD": "qsirecon/derivatives/qsirecon-DIPYDKI/sub-*/ses-*/dwi/sub-*_ses-*_acq-HBCD75_run-01_space-ACPC_model-dki_param-md_dwimap.nii.gz",
    "DKI-MD-Micro": "qsirecon/derivatives/qsirecon-DIPYDKI/sub-*/ses-*/dwi/sub-*_ses-*_acq-HBCD75_run-01_space-ACPC_model-dkimicro_param-md_dwimap.nii.gz",
    "DKI-MK": "qsirecon/derivatives/qsirecon-DIPYDKI/sub-*/ses-*/dwi/sub-*_ses-*_acq-HBCD75_run-01_space-ACPC_model-dki_param-mk_dwimap.nii.gz",
    "DKI-MKT": "qsirecon/derivatives/qsirecon-DIPYDKI/sub-*/ses-*/dwi/sub-*_ses-*_acq-HBCD75_run-01_space-ACPC_model-dki_param-mkt_dwimap.nii.gz",
    "DKI-RD": "qsirecon/derivatives/qsirecon-DIPYDKI/sub-*/ses-*/dwi/sub-*_ses-*_acq-HBCD75_run-01_space-ACPC_model-dki_param-rd_dwimap.nii.gz",
    "DKI-RD-Micro": "qsirecon/derivatives/qsirecon-DIPYDKI/sub-*/ses-*/dwi/sub-*_ses-*_acq-HBCD75_run-01_space-ACPC_model-dkimicro_param-rd_dwimap.nii.gz",
    "DKI-RDE-Micro": "qsirecon/derivatives/qsirecon-DIPYDKI/sub-*/ses-*/dwi/sub-*_ses-*_acq-HBCD75_run-01_space-ACPC_model-dkimicro_param-rde_dwimap.nii.gz",
    "DKI-RK": "qsirecon/derivatives/qsirecon-DIPYDKI/sub-*/ses-*/dwi/sub-*_ses-*_acq-HBCD75_run-01_space-ACPC_model-dki_param-rk_dwimap.nii.gz",
    "DKI-Linearity": "qsirecon/derivatives/qsirecon-DIPYDKI/sub-*/ses-*/dwi/sub-*_ses-*_acq-HBCD75_run-01_space-ACPC_model-dki_param-linearity_dwimap.nii.gz",
    "DKI-Planarity": "qsirecon/derivatives/qsirecon-DIPYDKI/sub-*/ses-*/dwi/sub-*_ses-*_acq-HBCD75_run-01_space-ACPC_model-dki_param-planarity_dwimap.nii.gz",
    "DKI-Sphericity": "qsirecon/derivatives/qsirecon-DIPYDKI/sub-*/ses-*/dwi/sub-*_ses-*_acq-HBCD75_run-01_space-ACPC_model-dki_param-sphericity_dwimap.nii.gz",
    "DKI-Trace-Micro": "qsirecon/derivatives/qsirecon-DIPYDKI/sub-*/ses-*/dwi/sub-*_ses-*_acq-HBCD75_run-01_space-ACPC_model-dkimicro_param-trace_dwimap.nii.gz",
    "DKI-Tortuosity-Micro": "qsirecon/derivatives/qsirecon-DIPYDKI/sub-*/ses-*/dwi/sub-*_ses-*_acq-HBCD75_run-01_space-ACPC_model-dkimicro_param-tortuosity_dwimap.nii.gz",
    "DKI-MSDKI-AWF": "qsirecon/derivatives/qsirecon-DIPYDKI/sub-*/ses-*/dwi/sub-*_ses-*_acq-HBCD75_run-01_space-ACPC_model-msdki_param-awf_dwimap.nii.gz",
    "DKI-MSDKI-DI": "qsirecon/derivatives/qsirecon-DIPYDKI/sub-*/ses-*/dwi/sub-*_ses-*_acq-HBCD75_run-01_space-ACPC_model-msdki_param-di_dwimap.nii.gz",
    "DKI-MSDKI-MFA": "qsirecon/derivatives/qsirecon-DIPYDKI/sub-*/ses-*/dwi/sub-*_ses-*_acq-HBCD75_run-01_space-ACPC_model-msdki_param-mfa_dwimap.nii.gz",
    "DKI-MSDKI-MSD": "qsirecon/derivatives/qsirecon-DIPYDKI/sub-*/ses-*/dwi/sub-*_ses-*_acq-HBCD75_run-01_space-ACPC_model-msdki_param-msd_dwimap.nii.gz",
    "DKI-MSDKI-MSK": "qsirecon/derivatives/qsirecon-DIPYDKI/sub-*/ses-*/dwi/sub-*_ses-*_acq-HBCD75_run-01_space-ACPC_model-msdki_param-msk_dwimap.nii.gz",
    # DWI DSIStudio
    "DSIStudio-GQI-GFA": "qsirecon/derivatives/qsirecon-DSIStudio/sub-*/ses-*/dwi/sub-*_ses-*_acq-HBCD75_run-01_space-ACPC_model-gqi_param-gfa_dwimap.nii.gz",
    "DSIStudio-GQI-ISO": "qsirecon/derivatives/qsirecon-DSIStudio/sub-*/ses-*/dwi/sub-*_ses-*_acq-HBCD75_run-01_space-ACPC_model-gqi_param-iso_dwimap.nii.gz",
    "DSIStudio-GQI-QA": "qsirecon/derivatives/qsirecon-DSIStudio/sub-*/ses-*/dwi/sub-*_ses-*_acq-HBCD75_run-01_space-ACPC_model-gqi_param-qa_dwimap.nii.gz",
    "DSIStudio-GQI-RDI": "qsirecon/derivatives/qsirecon-DSIStudio/sub-*/ses-*/dwi/sub-*_ses-*_acq-HBCD75_run-01_space-ACPC_model-gqi_param-rdi_dwimap.nii.gz",
    "DSIStudio-Tensor-AD": "qsirecon/derivatives/qsirecon-DSIStudio/sub-*/ses-*/dwi/sub-*_ses-*_acq-HBCD75_run-01_space-ACPC_model-tensor_param-ad_dwimap.nii.gz",
    "DSIStudio-Tensor-FA": "qsirecon/derivatives/qsirecon-DSIStudio/sub-*/ses-*/dwi/sub-*_ses-*_acq-HBCD75_run-01_space-ACPC_model-tensor_param-fa_dwimap.nii.gz",
    "DSIStudio-Tensor-MD": "qsirecon/derivatives/qsirecon-DSIStudio/sub-*/ses-*/dwi/sub-*_ses-*_acq-HBCD75_run-01_space-ACPC_model-tensor_param-md_dwimap.nii.gz",
    "DSIStudio-Tensor-RD": "qsirecon/derivatives/qsirecon-DSIStudio/sub-*/ses-*/dwi/sub-*_ses-*_acq-HBCD75_run-01_space-ACPC_model-tensor_param-rd_dwimap.nii.gz",
    "DSIStudio-Tensor-RD1": "qsirecon/derivatives/qsirecon-DSIStudio/sub-*/ses-*/dwi/sub-*_ses-*_acq-HBCD75_run-01_space-ACPC_model-tensor_param-rd1_dwimap.nii.gz",
    "DSIStudio-Tensor-RD2": "qsirecon/derivatives/qsirecon-DSIStudio/sub-*/ses-*/dwi/sub-*_ses-*_acq-HBCD75_run-01_space-ACPC_model-tensor_param-rd2_dwimap.nii.gz",
    # DWI NODDI
    "NODDI-ICVF-Modulated": "qsirecon/derivatives/qsirecon-NODDI/sub-*/ses-*/dwi/sub-*_ses-*_acq-HBCD75_run-01_space-ACPC_model-noddi_param-icvf_desc-modulated_dwimap.nii.gz",
    "NODDI-ICVF": "qsirecon/derivatives/qsirecon-NODDI/sub-*/ses-*/dwi/sub-*_ses-*_acq-HBCD75_run-01_space-ACPC_model-noddi_param-icvf_dwimap.nii.gz",
    "NODDI-ISOVF": "qsirecon/derivatives/qsirecon-NODDI/sub-*/ses-*/dwi/sub-*_ses-*_acq-HBCD75_run-01_space-ACPC_model-noddi_param-isovf_dwimap.nii.gz",
    "NODDI-NRMSE": "qsirecon/derivatives/qsirecon-NODDI/sub-*/ses-*/dwi/sub-*_ses-*_acq-HBCD75_run-01_space-ACPC_model-noddi_param-nrmse_dwimap.nii.gz",
    "NODDI-RMSE": "qsirecon/derivatives/qsirecon-NODDI/sub-*/ses-*/dwi/sub-*_ses-*_acq-HBCD75_run-01_space-ACPC_model-noddi_param-rmse_dwimap.nii.gz",
    "NODDI-OD-Modulated": "qsirecon/derivatives/qsirecon-NODDI/sub-*/ses-*/dwi/sub-*_ses-*_acq-HBCD75_run-01_space-ACPC_model-noddi_param-od_desc-modulated_dwimap.nii.gz",
    "NODDI-OD": "qsirecon/derivatives/qsirecon-NODDI/sub-*/ses-*/dwi/sub-*_ses-*_acq-HBCD75_run-01_space-ACPC_model-noddi_param-od_dwimap.nii.gz",
    "NODDI-TF": "qsirecon/derivatives/qsirecon-NODDI/sub-*/ses-*/dwi/sub-*_ses-*_acq-HBCD75_run-01_space-ACPC_model-noddi_param-tf_dwimap.nii.gz",
    # DWI MAPMRI
    "MAPMRI-NG": "qsirecon/derivatives/qsirecon-TORTOISE_model-MAPMRI/sub-*/ses-*/dwi/sub-*_ses-*_acq-HBCD75_run-01_space-ACPC_model-mapmri_param-ng_dwimap.nii.gz",
    "MAPMRI-NGPar": "qsirecon/derivatives/qsirecon-TORTOISE_model-MAPMRI/sub-*/ses-*/dwi/sub-*_ses-*_acq-HBCD75_run-01_space-ACPC_model-mapmri_param-ngpar_dwimap.nii.gz",
    "MAPMRI-NGPerp": "qsirecon/derivatives/qsirecon-TORTOISE_model-MAPMRI/sub-*/ses-*/dwi/sub-*_ses-*_acq-HBCD75_run-01_space-ACPC_model-mapmri_param-ngperp_dwimap.nii.gz",
    "MAPMRI-PA": "qsirecon/derivatives/qsirecon-TORTOISE_model-MAPMRI/sub-*/ses-*/dwi/sub-*_ses-*_acq-HBCD75_run-01_space-ACPC_model-mapmri_param-pa_dwimap.nii.gz",
    "MAPMRI-PAth": "qsirecon/derivatives/qsirecon-TORTOISE_model-MAPMRI/sub-*/ses-*/dwi/sub-*_ses-*_acq-HBCD75_run-01_space-ACPC_model-mapmri_param-path_dwimap.nii.gz",
    "MAPMRI-RTAP": "qsirecon/derivatives/qsirecon-TORTOISE_model-MAPMRI/sub-*/ses-*/dwi/sub-*_ses-*_acq-HBCD75_run-01_space-ACPC_model-mapmri_param-rtap_dwimap.nii.gz",
    "MAPMRI-RTOP": "qsirecon/derivatives/qsirecon-TORTOISE_model-MAPMRI/sub-*/ses-*/dwi/sub-*_ses-*_acq-HBCD75_run-01_space-ACPC_model-mapmri_param-rtop_dwimap.nii.gz",
    "MAPMRI-RTPP": "qsirecon/derivatives/qsirecon-TORTOISE_model-MAPMRI/sub-*/ses-*/dwi/sub-*_ses-*_acq-HBCD75_run-01_space-ACPC_model-mapmri_param-rtpp_dwimap.nii.gz",
    # DWI TORTOISE tensor
    "TORTOISE-FullShell-AD": "qsirecon/derivatives/qsirecon-TORTOISE_model-MAPMRI/sub-*/ses-*/dwi/sub-*_ses-*_acq-HBCD75_run-01_space-ACPC_model-tensor_param-ad_dwimap.nii.gz",
    "TORTOISE-FullShell-FA": "qsirecon/derivatives/qsirecon-TORTOISE_model-MAPMRI/sub-*/ses-*/dwi/sub-*_ses-*_acq-HBCD75_run-01_space-ACPC_model-tensor_param-fa_dwimap.nii.gz",
    "TORTOISE-FullShell-LI": "qsirecon/derivatives/qsirecon-TORTOISE_model-MAPMRI/sub-*/ses-*/dwi/sub-*_ses-*_acq-HBCD75_run-01_space-ACPC_model-tensor_param-li_dwimap.nii.gz",
    "TORTOISE-FullShell-MD": "qsirecon/derivatives/qsirecon-TORTOISE_model-MAPMRI/sub-*/ses-*/dwi/sub-*_ses-*_acq-HBCD75_run-01_space-ACPC_model-tensor_param-md_dwimap.nii.gz",
    "TORTOISE-FullShell-RD": "qsirecon/derivatives/qsirecon-TORTOISE_model-MAPMRI/sub-*/ses-*/dwi/sub-*_ses-*_acq-HBCD75_run-01_space-ACPC_model-tensor_param-rd_dwimap.nii.gz",
    "TORTOISE-InnerShell-AD": "qsirecon/derivatives/qsirecon-TORTOISE_model-tensor/sub-*/ses-*/dwi/sub-*_ses-*_acq-HBCD75_run-01_space-ACPC_model-tensor_param-ad_dwimap.nii.gz",
    "TORTOISE-InnerShell-FA": "qsirecon/derivatives/qsirecon-TORTOISE_model-tensor/sub-*/ses-*/dwi/sub-*_ses-*_acq-HBCD75_run-01_space-ACPC_model-tensor_param-fa_dwimap.nii.gz",
    "TORTOISE-InnerShell-LI": "qsirecon/derivatives/qsirecon-TORTOISE_model-tensor/sub-*/ses-*/dwi/sub-*_ses-*_acq-HBCD75_run-01_space-ACPC_model-tensor_param-li_dwimap.nii.gz",
    "TORTOISE-InnerShell-MD": "qsirecon/derivatives/qsirecon-TORTOISE_model-tensor/sub-*/ses-*/dwi/sub-*_ses-*_acq-HBCD75_run-01_space-ACPC_model-tensor_param-md_dwimap.nii.gz",
    "TORTOISE-InnerShell-RD": "qsirecon/derivatives/qsirecon-TORTOISE_model-tensor/sub-*/ses-*/dwi/sub-*_ses-*_acq-HBCD75_run-01_space-ACPC_model-tensor_param-rd_dwimap.nii.gz",
    # QSM
    "QSM-SEPIA-E5": "qsm/sub-*/ses-*/anat/*_space-T1w_desc-E12345+sepia_Chimap.nii.gz",
    "QSM-X-R2p-E5-X": "qsm/sub-*/ses-*/anat/*_space-T1w_desc-E12345+chisep+r2p_Chimap.nii.gz",
    "QSM-X-R2p-E5-Para": "qsm/sub-*/ses-*/anat/*_space-T1w_desc-E12345+chisep+r2p_para.nii.gz",
    "QSM-X-R2p-E5-Dia": "qsm/sub-*/ses-*/anat/*_space-T1w_desc-E12345+chisep+r2p_dia.nii.gz",
    # ihMT
    "ihMTw": "ihmt/sub-*/ses-*/anat/sub-*_ses-*_run-01_space-T1w_ihMTw.nii.gz",
    "ihMTR": "ihmt/sub-*/ses-*/anat/sub-*_ses-*_run-01_space-T1w_ihMTR.nii.gz",
    "MTR": "ihmt/sub-*/ses-*/anat/sub-*_ses-*_run-01_space-T1w_MTRmap.nii.gz",
    "ihMTsat": "ihmt/sub-*/ses-*/anat/sub-*_ses-*_run-01_space-T1w_ihMTsat.nii.gz",
    "ihMTsat-B1c": "ihmt/sub-*/ses-*/anat/sub-*_ses-*_run-01_space-T1w_ihMTsatB1sq.nii.gz",
    # MP2RAGE
    "R1": "pymp2rage/sub-*/ses-*/anat/sub-*_ses-*_run-01_space-T1w_R1map.nii.gz",
    "R1-B1c": "pymp2rage/sub-*/ses-*/anat/sub-*_ses-*_run-01_space-T1w_desc-B1corrected_R1map.nii.gz",
    # T1/T2
    "MPRAGE-MyelinW": "t1wt2w_ratio/sub-*/ses-*/anat/sub-*_ses-*_run-01_space-T1w_desc-MPRAGEunscaled_myelinw.nii.gz",
    "SPACE-MyelinW": "t1wt2w_ratio/sub-*/ses-*/anat/sub-*_ses-*_run-01_space-T1w_desc-SPACEunscaled_myelinw.nii.gz",
    "Scaled MPRAGE-MyelinW": "t1wt2w_ratio/sub-*/ses-*/anat/sub-*_ses-*_run-01_space-T1w_desc-MPRAGEscaled_myelinw.nii.gz",
    "Scaled SPACE-MyelinW": "t1wt2w_ratio/sub-*/ses-*/anat/sub-*_ses-*_run-01_space-T1w_desc-SPACEscaled_myelinw.nii.gz",
    # g-ratio
    "G-ihMTsat": "g_ratio/sub-*/ses-*/anat/sub-*_ses-*_run-01_space-T1w_desc-MTsat+ISOVF+ICVF_gratio.nii.gz",
    "G-ihMTR": "g_ratio/sub-*/ses-*/anat/sub-*_ses-*_run-01_space-T1w_desc-ihMTR+ISOVF+ICVF_gratio.nii.gz",
}

STATS = ("mean", "median", "std", "min", "max")
KEY_RE = re.compile(r"(ses-[A-Za-z0-9]+)|(run-[A-Za-z0-9]+)")
EXCLUDED_LABELS = {11142, 12142}  # Medial wall labels


def _read_lut_subset(lut_file: str) -> pd.DataFrame:
    """Read a2009s cortical labels from FreeSurferColorLUT.

    Keeps only 11101-11175 (lh) and 12101-12175 (rh).
    """
    rows: list[dict[str, object]] = []
    with open(lut_file) as fobj:
        for line in fobj:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            parts = stripped.split()
            if len(parts) < 2:
                continue
            try:
                intensity = int(parts[0])
            except ValueError:
                continue
            name = parts[1]
            is_lh = 11101 <= intensity <= 11175
            is_rh = 12101 <= intensity <= 12175
            if not (is_lh or is_rh):
                continue
            if intensity in EXCLUDED_LABELS:
                continue
            rows.append(
                {
                    "parcel_intensity": intensity,
                    "parcel_name": name,
                    "parcel_hemi": "lh" if is_lh else "rh",
                }
            )
    if not rows:
        raise RuntimeError(f"No target labels parsed from {lut_file}")
    df = pd.DataFrame(rows).drop_duplicates(subset=["parcel_intensity", "parcel_name"])
    return df.sort_values("parcel_intensity").reset_index(drop=True)


def _parse_ses_run(path: str) -> tuple[str, str]:
    matches = [m.group(0) for m in KEY_RE.finditer(os.path.basename(path))]
    ses = "ses-unknown"
    run = "run-01"
    for token in matches:
        if token.startswith("ses-"):
            ses = token
        elif token.startswith("run-"):
            run = token
    return ses, run


def _build_metric_files(subject: str, deriv_dir: str) -> dict[tuple[str, str], dict[str, str]]:
    metric_files_by_key: dict[tuple[str, str], dict[str, str]] = defaultdict(dict)
    subject_tok = f"sub-{subject}"
    for metric_name, rel_pattern in PATTERNS_SUBJECT.items():
        subj_pattern = rel_pattern.replace("sub-*", subject_tok)
        matches = sorted(glob(os.path.join(deriv_dir, subj_pattern)))
        if not matches:
            continue
        for map_file in matches:
            ses, run = _parse_ses_run(map_file)
            key = (ses, run)
            if metric_name in metric_files_by_key[key]:
                # Keep the first deterministic match for duplicate paths.
                continue
            metric_files_by_key[key][metric_name] = map_file
    return metric_files_by_key


def _space_from_path(path: str) -> str:
    fname = os.path.basename(path)
    has_acpc = "_space-ACPC_" in fname
    has_t1w = "_space-T1w_" in fname
    if has_acpc and not has_t1w:
        return "ACPC"
    if has_t1w and not has_acpc:
        return "T1w"
    raise ValueError(
        "Could not unambiguously determine space from filename "
        f"(expected _space-ACPC_ or _space-T1w_): {fname}"
    )


def _expected_space_for_metric(metric_name: str) -> str:
    pattern = PATTERNS_SUBJECT[metric_name]
    has_acpc = "_space-ACPC_" in pattern
    has_t1w = "_space-T1w_" in pattern
    if has_acpc and not has_t1w:
        return "ACPC"
    if has_t1w and not has_acpc:
        return "T1w"
    raise ValueError(
        "Metric pattern must include exactly one of _space-ACPC_ or _space-T1w_: "
        f"{metric_name} -> {pattern}"
    )


def _images_share_grid(image_a: ants.ANTsImage, image_b: ants.ANTsImage) -> bool:
    """Return True when two ANTs images use the same voxel grid."""
    return (
        image_a.shape == image_b.shape
        and np.allclose(image_a.spacing, image_b.spacing)
        and np.allclose(image_a.origin, image_b.origin)
        and np.allclose(image_a.direction, image_b.direction)
    )


def _compute_stats(values: np.ndarray) -> dict[str, float]:
    if values.size == 0:
        return {
            "mean": np.nan,
            "median": np.nan,
            "std": np.nan,
            "min": np.nan,
            "max": np.nan,
        }
    return {
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "std": float(np.std(values)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
    }


def process_subject(
    subject: str,
    deriv_dir: str,
    zero_is_missing: bool = True,
) -> None:
    t1w_reg_dir = os.path.join(deriv_dir, "t1w_registration", f"sub-{subject}", "anat")
    out_dir = os.path.join(deriv_dir, "parcel_myelin_stats", f"sub-{subject}")
    os.makedirs(out_dir, exist_ok=True)

    lut_file = os.path.normpath(
        os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "..",
            "processing",
            "FreeSurferColorLUT.txt",
        )
    )
    dseg_t1w = os.path.join(t1w_reg_dir, f"sub-{subject}_space-T1w_desc-a2009s_dseg.nii.gz")
    dseg_acpc = os.path.join(t1w_reg_dir, f"sub-{subject}_space-ACPC_desc-a2009s_dseg.nii.gz")

    required_files = [lut_file, dseg_t1w, dseg_acpc]
    for required in required_files:
        if not os.path.exists(required):
            raise FileNotFoundError(required)
    dseg_imgs = {
        "T1w": ants.image_read(dseg_t1w),
        "ACPC": ants.image_read(dseg_acpc),
    }
    dseg_arrays = {space: img.numpy().astype(np.int64) for space, img in dseg_imgs.items()}
    parcel_df = _read_lut_subset(lut_file)
    label_ids = parcel_df["parcel_intensity"].astype(int).to_numpy()
    available_labels = set(np.unique(dseg_arrays["T1w"]).astype(int)) | set(
        np.unique(dseg_arrays["ACPC"]).astype(int)
    )
    missing_label_ids = [label_id for label_id in label_ids if label_id not in available_labels]
    if missing_label_ids:
        print(
            f"{len(missing_label_ids)} LUT labels absent from subject dseg volumes.",
            flush=True,
        )

    t1w_counts = np.array(
        [int(np.count_nonzero(dseg_arrays["T1w"] == label_id)) for label_id in label_ids],
        dtype=np.int64,
    )
    acpc_counts = np.array(
        [int(np.count_nonzero(dseg_arrays["ACPC"] == label_id)) for label_id in label_ids],
        dtype=np.int64,
    )

    metric_files_by_key = _build_metric_files(subject, deriv_dir)
    if not metric_files_by_key:
        print(f"No scalar maps found for sub-{subject}", flush=True)
        return

    expected_space_by_metric = {
        metric_name: _expected_space_for_metric(metric_name) for metric_name in PATTERNS_SUBJECT
    }

    for (ses, run), metric_files in sorted(metric_files_by_key.items()):
        out_df = parcel_df.copy()
        out_df.insert(3, "parcel_count_t1w", t1w_counts)
        out_df.insert(4, "parcel_count_acpc", acpc_counts)

        for metric_name in PATTERNS_SUBJECT:
            for stat in STATS:
                out_df[f"{metric_name}_{stat}"] = np.nan

        # Long-format coverage output: one row per metric and parcel.
        coverage_rows: list[dict[str, object]] = []

        for metric_name, metric_file in metric_files.items():
            actual_space = _space_from_path(metric_file)
            expected_space = expected_space_by_metric[metric_name]
            if actual_space != expected_space:
                raise ValueError(
                    f"Space mismatch for {metric_name}: expected {expected_space}, "
                    f"got {actual_space} from {metric_file}"
                )
            space = actual_space
            dseg_img = dseg_imgs[space]

            # Keep the quantitative scalar map in its native grid. Move only
            # the categorical parcellation to that grid with label-aware
            # interpolation. This avoids smoothing, ringing, and mixing of
            # scalar values across parcel or missing-data boundaries.
            map_img = ants.image_read(metric_file)
            map_data = map_img.numpy()

            if _images_share_grid(dseg_img, map_img):
                metric_dseg_img = dseg_img
            else:
                metric_dseg_img = ants.resample_image_to_target(
                    image=dseg_img,
                    target=map_img,
                    interp_type="genericLabel",
                )

            # genericLabel should preserve integer labels. Rounding before
            # conversion guards against floating-point storage artifacts.
            metric_dseg_data = np.rint(metric_dseg_img.numpy()).astype(np.int64)

            valid_data = np.isfinite(map_data)
            if zero_is_missing:
                valid_data &= map_data != 0

            for label_id in label_ids:
                row_idx = out_df["parcel_intensity"] == label_id
                parcel_row = out_df.loc[row_idx].iloc[0]

                parcel_mask = metric_dseg_data == label_id
                parcel_values = map_data[parcel_mask]
                parcel_valid = valid_data[parcel_mask]
                valid_values = parcel_values[parcel_valid]
                n_total = int(parcel_values.size)
                n_valid = int(np.count_nonzero(parcel_valid))
                coverage = n_valid / n_total if n_total > 0 else np.nan

                stats = _compute_stats(valid_values)
                for stat_name, stat_val in stats.items():
                    out_df.loc[row_idx, f"{metric_name}_{stat_name}"] = stat_val

                coverage_rows.append(
                    {
                        "subject": f"sub-{subject}",
                        "session": ses,
                        "run": run,
                        "metric": metric_name,
                        "space": space,
                        "parcel_intensity": int(label_id),
                        "parcel_name": str(parcel_row["parcel_name"]),
                        "parcel_hemi": str(parcel_row["parcel_hemi"]),
                        "parcel_count": n_total,
                        "valid_count": n_valid,
                        "coverage": coverage,
                    }
                )

        out_file = os.path.join(
            out_dir,
            f"sub-{subject}_{ses}_{run}_desc-a2009s_scalarstats.csv",
        )
        out_df.to_csv(out_file, index=False)
        print(f"Wrote {out_file}", flush=True)

        coverage_file = os.path.join(
            out_dir,
            f"sub-{subject}_{ses}_{run}_desc-a2009s_coverage.csv",
        )
        coverage_df = pd.DataFrame(coverage_rows).sort_values(
            ["metric", "parcel_intensity"]
        )
        coverage_df.to_csv(coverage_file, index=False)
        print(f"Wrote {coverage_file}", flush=True)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--subject-id",
        required=True,
        help="Subject ID without the sub- prefix",
    )
    parser.add_argument(
        "--include-zero",
        action="store_true",
        help=(
            "Include exact zero values in parcel statistics and coverage. "
            "By default, zero and all nonfinite values are treated as invalid."
        ),
    )
    return parser


if __name__ == "__main__":
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
    from configuration.config import load_config

    args = _build_parser().parse_args()
    cfg = load_config()
    derivatives_dir = os.path.join(cfg["project_root"], "derivatives")
    process_subject(
        args.subject_id,
        derivatives_dir,
        zero_is_missing=not args.include_zero,
    )