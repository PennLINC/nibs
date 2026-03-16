#!/usr/bin/env python3
"""
Combined GM + WM parcel-wise ICC analysis in MNI152NLin2009cAsym space.

- WM: JHU ICBM white-matter atlas (single label image)
- GM: Vogt atlas (LH + RH combined into one GM analysis;
      parcel IDs retain hemisphere prefixes, but GM is output as one set)

Outputs are generated separately for GM and WM:
- parcel means
- ICC summary
- ICC heatmap (rows ordered by mean ICC)
- violin plots colored by model family
"""

from __future__ import annotations

import argparse
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import nibabel as nib
from nilearn.image import resample_to_img

# headless plotting
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

try:
    import pingouin as pg
    _HAVE_PG = True
except Exception:
    _HAVE_PG = False

try:
    from tqdm import tqdm
except Exception:
    def tqdm(x, **kwargs):
        return x


# ==========================
# Root paths
# ==========================
DERIV_ROOT = Path("/cbica/projects/nibs/derivatives")

ATLAS_ROOT = Path(
    "/cbica/projects/nibs/projects/myelin_reliability/code/myelin_atlas"
)

WM_LABEL = ATLAS_ROOT / "JHU-ICBM-labels-1mm-MNI152NLin2009cAsym.nii.gz"
LH_LABEL = (
    ATLAS_ROOT
    / "MYATLAS_package_new/maps/Volume/vogt_multilabel_lh.nii"
)
RH_LABEL = (
    ATLAS_ROOT
    / "MYATLAS_package_new/maps/Volume/vogt_multilabel_rh.nii"
)


# ==========================
# Regex + labels
# ==========================
BIDS_SUB_RE = re.compile(r"sub-(?P<sub>[^_/]+)")
BIDS_SES_RE = re.compile(r"ses-(?P<ses>[^_/]+)")


# ==========================
# Metric patterns (MNI)
# relative to DERIV_ROOT
# ==========================
PATTERNS_MNI: Dict[str, str] = {
    # DWI DKI
    "FA": "qsirecon/derivatives/qsirecon-DIPYDKI/sub-*/ses-*/dwi/sub-*_ses-*_acq-HBCD75_run-01_space-MNI152NLin2009cAsym_model-tensor_param-fa_dwimap.nii.gz",
    "KFA": "qsirecon/derivatives/qsirecon-DIPYDKI/sub-*/ses-*/dwi/sub-*_ses-*_acq-HBCD75_run-01_space-MNI152NLin2009cAsym_model-dki_param-kfa_dwimap.nii.gz",
    "KFA-Micro": "qsirecon/derivatives/qsirecon-DIPYDKI/sub-*/ses-*/dwi/sub-*_ses-*_acq-HBCD75_run-01_space-MNI152NLin2009cAsym_model-dkimicro_param-kfa_dwimap.nii.gz",
    "AD": "qsirecon/derivatives/qsirecon-DIPYDKI/sub-*/ses-*/dwi/sub-*_ses-*_acq-HBCD75_run-01_space-MNI152NLin2009cAsym_model-dki_param-ad_dwimap.nii.gz",
    "AD-Micro": "qsirecon/derivatives/qsirecon-DIPYDKI/sub-*/ses-*/dwi/sub-*_ses-*_acq-HBCD75_run-01_space-MNI152NLin2009cAsym_model-dkimicro_param-ad_dwimap.nii.gz",
    "ADE-Micro": "qsirecon/derivatives/qsirecon-DIPYDKI/sub-*/ses-*/dwi/sub-*_ses-*_acq-HBCD75_run-01_space-MNI152NLin2009cAsym_model-dkimicro_param-ade_dwimap.nii.gz",
    "AWF-Micro": "qsirecon/derivatives/qsirecon-DIPYDKI/sub-*/ses-*/dwi/sub-*_ses-*_acq-HBCD75_run-01_space-MNI152NLin2009cAsym_model-dkimicro_param-awf_dwimap.nii.gz",
    "AxonALD-Micro": "qsirecon/derivatives/qsirecon-DIPYDKI/sub-*/ses-*/dwi/sub-*_ses-*_acq-HBCD75_run-01_space-MNI152NLin2009cAsym_model-dkimicro_param-axonald_dwimap.nii.gz",
    "AK": "qsirecon/derivatives/qsirecon-DIPYDKI/sub-*/ses-*/dwi/sub-*_ses-*_acq-HBCD75_run-01_space-MNI152NLin2009cAsym_model-dki_param-ak_dwimap.nii.gz",
    "MD": "qsirecon/derivatives/qsirecon-DIPYDKI/sub-*/ses-*/dwi/sub-*_ses-*_acq-HBCD75_run-01_space-MNI152NLin2009cAsym_model-dki_param-md_dwimap.nii.gz",
    "MD-Micro": "qsirecon/derivatives/qsirecon-DIPYDKI/sub-*/ses-*/dwi/sub-*_ses-*_acq-HBCD75_run-01_space-MNI152NLin2009cAsym_model-dkimicro_param-md_dwimap.nii.gz",
    "MK": "qsirecon/derivatives/qsirecon-DIPYDKI/sub-*/ses-*/dwi/sub-*_ses-*_acq-HBCD75_run-01_space-MNI152NLin2009cAsym_model-dki_param-mk_dwimap.nii.gz",
    "MKT": "qsirecon/derivatives/qsirecon-DIPYDKI/sub-*/ses-*/dwi/sub-*_ses-*_acq-HBCD75_run-01_space-MNI152NLin2009cAsym_model-dki_param-mkt_dwimap.nii.gz",
    "RD": "qsirecon/derivatives/qsirecon-DIPYDKI/sub-*/ses-*/dwi/sub-*_ses-*_acq-HBCD75_run-01_space-MNI152NLin2009cAsym_model-dki_param-rd_dwimap.nii.gz",
    "RD-Micro": "qsirecon/derivatives/qsirecon-DIPYDKI/sub-*/ses-*/dwi/sub-*_ses-*_acq-HBCD75_run-01_space-MNI152NLin2009cAsym_model-dkimicro_param-rd_dwimap.nii.gz",
    "RDE-Micro": "qsirecon/derivatives/qsirecon-DIPYDKI/sub-*/ses-*/dwi/sub-*_ses-*_acq-HBCD75_run-01_space-MNI152NLin2009cAsym_model-dkimicro_param-rde_dwimap.nii.gz",
    "RK": "qsirecon/derivatives/qsirecon-DIPYDKI/sub-*/ses-*/dwi/sub-*_ses-*_acq-HBCD75_run-01_space-MNI152NLin2009cAsym_model-dki_param-rk_dwimap.nii.gz",
    "Linearity": "qsirecon/derivatives/qsirecon-DIPYDKI/sub-*/ses-*/dwi/sub-*_ses-*_acq-HBCD75_run-01_space-MNI152NLin2009cAsym_model-dki_param-linearity_dwimap.nii.gz",
    "Planarity": "qsirecon/derivatives/qsirecon-DIPYDKI/sub-*/ses-*/dwi/sub-*_ses-*_acq-HBCD75_run-01_space-MNI152NLin2009cAsym_model-dki_param-planarity_dwimap.nii.gz",
    "Sphericity": "qsirecon/derivatives/qsirecon-DIPYDKI/sub-*/ses-*/dwi/sub-*_ses-*_acq-HBCD75_run-01_space-MNI152NLin2009cAsym_model-dki_param-sphericity_dwimap.nii.gz",
    "Trace-Micro": "qsirecon/derivatives/qsirecon-DIPYDKI/sub-*/ses-*/dwi/sub-*_ses-*_acq-HBCD75_run-01_space-MNI152NLin2009cAsym_model-dkimicro_param-trace_dwimap.nii.gz",
    "Tortuosity-Micro": "qsirecon/derivatives/qsirecon-DIPYDKI/sub-*/ses-*/dwi/sub-*_ses-*_acq-HBCD75_run-01_space-MNI152NLin2009cAsym_model-dkimicro_param-tortuosity_dwimap.nii.gz",

    # DWI NODDI
    "ICVF-Modulated": "qsirecon/derivatives/qsirecon-NODDI/sub-*/ses-*/dwi/sub-*_ses-*_acq-HBCD75_run-01_space-MNI152NLin2009cAsym_model-noddi_param-icvf_desc-modulated_dwimap.nii.gz",
    "ICVF": "qsirecon/derivatives/qsirecon-NODDI/sub-*/ses-*/dwi/sub-*_ses-*_acq-HBCD75_run-01_space-MNI152NLin2009cAsym_model-noddi_param-icvf_dwimap.nii.gz",
    "ISOVF": "qsirecon/derivatives/qsirecon-NODDI/sub-*/ses-*/dwi/sub-*_ses-*_acq-HBCD75_run-01_space-MNI152NLin2009cAsym_model-noddi_param-isovf_dwimap.nii.gz",
    "NRMSE": "qsirecon/derivatives/qsirecon-NODDI/sub-*/ses-*/dwi/sub-*_ses-*_acq-HBCD75_run-01_space-MNI152NLin2009cAsym_model-noddi_param-nrmse_dwimap.nii.gz",
    "RMSE": "qsirecon/derivatives/qsirecon-NODDI/sub-*/ses-*/dwi/sub-*_ses-*_acq-HBCD75_run-01_space-MNI152NLin2009cAsym_model-noddi_param-rmse_dwimap.nii.gz",
    "OD-Modulated": "qsirecon/derivatives/qsirecon-NODDI/sub-*/ses-*/dwi/sub-*_ses-*_acq-HBCD75_run-01_space-MNI152NLin2009cAsym_model-noddi_param-od_desc-modulated_dwimap.nii.gz",
    "OD": "qsirecon/derivatives/qsirecon-NODDI/sub-*/ses-*/dwi/sub-*_ses-*_acq-HBCD75_run-01_space-MNI152NLin2009cAsym_model-noddi_param-od_dwimap.nii.gz",
    "TF": "qsirecon/derivatives/qsirecon-NODDI/sub-*/ses-*/dwi/sub-*_ses-*_acq-HBCD75_run-01_space-MNI152NLin2009cAsym_model-noddi_param-tf_dwimap.nii.gz",

    # DWI MAPMRI
    "NG": "qsirecon/derivatives/qsirecon-TORTOISE_model-MAPMRI/sub-*/ses-*/dwi/sub-*_ses-*_acq-HBCD75_run-01_space-MNI152NLin2009cAsym_model-mapmri_param-ng_dwimap.nii.gz",
    "NGPar": "qsirecon/derivatives/qsirecon-TORTOISE_model-MAPMRI/sub-*/ses-*/dwi/sub-*_ses-*_acq-HBCD75_run-01_space-MNI152NLin2009cAsym_model-mapmri_param-ngpar_dwimap.nii.gz",
    "NGPerp": "qsirecon/derivatives/qsirecon-TORTOISE_model-MAPMRI/sub-*/ses-*/dwi/sub-*_ses-*_acq-HBCD75_run-01_space-MNI152NLin2009cAsym_model-mapmri_param-ngperp_dwimap.nii.gz",
    "PA": "qsirecon/derivatives/qsirecon-TORTOISE_model-MAPMRI/sub-*/ses-*/dwi/sub-*_ses-*_acq-HBCD75_run-01_space-MNI152NLin2009cAsym_model-mapmri_param-pa_dwimap.nii.gz",
    "PAth": "qsirecon/derivatives/qsirecon-TORTOISE_model-MAPMRI/sub-*/ses-*/dwi/sub-*_ses-*_acq-HBCD75_run-01_space-MNI152NLin2009cAsym_model-mapmri_param-path_dwimap.nii.gz",
    "RTAP": "qsirecon/derivatives/qsirecon-TORTOISE_model-MAPMRI/sub-*/ses-*/dwi/sub-*_ses-*_acq-HBCD75_run-01_space-MNI152NLin2009cAsym_model-mapmri_param-rtap_dwimap.nii.gz",
    "RTOP": "qsirecon/derivatives/qsirecon-TORTOISE_model-MAPMRI/sub-*/ses-*/dwi/sub-*_ses-*_acq-HBCD75_run-01_space-MNI152NLin2009cAsym_model-mapmri_param-rtop_dwimap.nii.gz",
    "RTPP": "qsirecon/derivatives/qsirecon-TORTOISE_model-MAPMRI/sub-*/ses-*/dwi/sub-*_ses-*_acq-HBCD75_run-01_space-MNI152NLin2009cAsym_model-mapmri_param-rtpp_dwimap.nii.gz",

    # QSM
    "MEGRE": "qsm/sub-*/ses-*/anat/sub-*_ses-*_acq-QSM_run-01_space-MNI152NLin2009cAsym_desc-mean_MEGRE.nii.gz",
    "QSM-SEPIA-E5": "qsm/sub-*/ses-*/anat/*_space-MNI152NLin2009cAsym_desc-E12345+sepia_Chimap.nii.gz",
    "QSM-X-R2p-E5-X": "qsm/sub-*/ses-*/anat/*_space-MNI152NLin2009cAsym_desc-E12345+chisep+r2p_Chimap.nii.gz",
    "QSM-X-R2p-E5-Para": "qsm/sub-*/ses-*/anat/*_space-MNI152NLin2009cAsym_desc-E12345+chisep+r2p_ironw.nii.gz",
    "QSM-X-R2p-E5-Dia": "qsm/sub-*/ses-*/anat/*_space-MNI152NLin2009cAsym_desc-E12345+chisep+r2p_myelinw.nii.gz",

    # ihMT
    "ihMTw": "ihmt/sub-*/ses-*/anat/sub-*_ses-*_run-01_space-MNI152NLin2009cAsym_ihMTw.nii.gz",
    "ihMTR": "ihmt/sub-*/ses-*/anat/sub-*_ses-*_run-01_space-MNI152NLin2009cAsym_ihMTR.nii.gz",
    "MTR": "ihmt/sub-*/ses-*/anat/sub-*_ses-*_run-01_space-MNI152NLin2009cAsym_MTRmap.nii.gz",
    "ihMTsat": "ihmt/sub-*/ses-*/anat/sub-*_ses-*_run-01_space-MNI152NLin2009cAsym_ihMTsat.nii.gz",
    "ihMTsat-B1c": "ihmt/sub-*/ses-*/anat/sub-*_ses-*_run-01_space-MNI152NLin2009cAsym_ihMTsatB1sq.nii.gz",

    # MP2RAGE
    "R1": "pymp2rage/sub-*/ses-*/anat/sub-*_ses-*_run-01_space-MNI152NLin2009cAsym_R1map.nii.gz",
    "R1-B1c": "pymp2rage/sub-*/ses-*/anat/sub-*_ses-*_run-01_space-MNI152NLin2009cAsym_desc-B1corrected_R1map.nii.gz",

    # T1/T2
    "MPRAGE-MyelinW": "t1wt2w_ratio/sub-*/ses-*/anat/sub-*_ses-*_run-01_space-MNI152NLin2009cAsym_desc-MPRAGEunscaled_myelinw.nii.gz",
    "SPACE-MyelinW": "t1wt2w_ratio/sub-*/ses-*/anat/sub-*_ses-*_run-01_space-MNI152NLin2009cAsym_desc-SPACEunscaled_myelinw.nii.gz",
    "Scaled MPRAGE-MyelinW": "t1wt2w_ratio/sub-*/ses-*/anat/sub-*_ses-*_run-01_space-MNI152NLin2009cAsym_desc-MPRAGEscaled_myelinw.nii.gz",
    "Scaled SPACE-MyelinW": "t1wt2w_ratio/sub-*/ses-*/anat/sub-*_ses-*_run-01_space-MNI152NLin2009cAsym_desc-SPACEscaled_myelinw.nii.gz",

    # g-ratio
    "G-MPRAGE-MyelinW": "g_ratio/sub-*/ses-*/anat/sub-*_ses-*_run-01_space-MNI152NLin2009cAsym_desc-MPRAGET1wT2w+ISOVF+ICVF_gratio.nii.gz",
    "G-SPACE-MyelinW": "g_ratio/sub-*/ses-*/anat/sub-*_ses-*_run-01_space-MNI152NLin2009cAsym_desc-SPACET1wT2w+ISOVF+ICVF_gratio.nii.gz",
    "G-ihMTsat": "g_ratio/sub-*/ses-*/anat/sub-*_ses-*_run-01_space-MNI152NLin2009cAsym_desc-MTsat+ISOVF+ICVF_gratio.nii.gz",
    "G-ihMTR": "g_ratio/sub-*/ses-*/anat/sub-*_ses-*_run-01_space-MNI152NLin2009cAsym_desc-ihMTR+ISOVF+ICVF_gratio.nii.gz",
}


MEASURE_TO_MODEL = {
    # DWI DKI
    "FA": "DKI",
    "KFA": "DKI",
    "AD": "DKI",
    "MD": "DKI",
    "MK": "DKI",
    "RD": "DKI",
    "RK": "DKI",
    "AK": "DKI",
    "MKT": "DKI",
    "Linearity": "DKI",
    "Planarity": "DKI",
    "Sphericity": "DKI",

    # DKI micro
    "KFA-Micro": "DKI-micro",
    "AD-Micro": "DKI-micro",
    "ADE-Micro": "DKI-micro",
    "AWF-Micro": "DKI-micro",
    "AxonALD-Micro": "DKI-micro",
    "MD-Micro": "DKI-micro",
    "RD-Micro": "DKI-micro",
    "RDE-Micro": "DKI-micro",
    "Trace-Micro": "DKI-micro",
    "Tortuosity-Micro": "DKI-micro",

    # DWI NODDI
    "ICVF": "NODDI",
    "ICVF-Modulated": "NODDI",
    "ISOVF": "NODDI",
    "OD": "NODDI",
    "OD-Modulated": "NODDI",
    "TF": "NODDI",
    "RMSE": "NODDI",
    "NRMSE": "NODDI",

    # MAPMRI
    "NG": "MAPMRI",
    "NGPar": "MAPMRI",
    "NGPerp": "MAPMRI",
    "PA": "MAPMRI",
    "PAth": "MAPMRI",
    "RTAP": "MAPMRI",
    "RTOP": "MAPMRI",
    "RTPP": "MAPMRI",

    # QSM
    "MEGRE": "QSM",
    "QSM-SEPIA-E5": "QSM",
    "QSM-X-R2p-E5-X": "QSM",
    "QSM-X-R2p-E5-Para": "QSM",
    "QSM-X-R2p-E5-Dia": "QSM",

    # ihMT
    "ihMTw": "ihMT",
    "ihMTR": "ihMT",
    "MTR": "ihMT",
    "ihMTsat": "ihMT",
    "ihMTsat-B1c": "ihMT",

    # MP2RAGE
    "R1": "MP2RAGE",
    "R1-B1c": "MP2RAGE",

    # T1/T2
    "MPRAGE-MyelinW": "T1w/T2w",
    "SPACE-MyelinW": "T1w/T2w",
    "Scaled MPRAGE-MyelinW": "T1w/T2w",
    "Scaled SPACE-MyelinW": "T1w/T2w",

    # g-ratio
    "G-MPRAGE-MyelinW": "g-ratio",
    "G-SPACE-MyelinW": "g-ratio",
    "G-ihMTsat": "g-ratio",
    "G-ihMTR": "g-ratio",
}

MODEL_COLORS = {
    "DKI": "#1f77b4",
    "DKI-micro": "#4fa3d1",
    "NODDI": "#ff7f0e",
    "MAPMRI": "#2ca02c",
    "QSM": "#d62728",
    "ihMT": "#9467bd",
    "MP2RAGE": "#8c564b",
    "T1w/T2w": "#e377c2",
    "g-ratio": "#7f7f7f",
}


# ==========================
# Helpers
# ==========================
def parse_sub_ses(p: Path) -> Tuple[str, str | None]:
    msub = BIDS_SUB_RE.search(p.as_posix())
    msea = BIDS_SES_RE.search(p.as_posix())
    return (
        msub.group("sub") if msub else "unknown",
        msea.group("ses") if msea else None,
    )


def collect_metric_files(patterns: Dict[str, str], deriv_root: Path) -> pd.DataFrame:
    rows = []
    for meas, globpat in patterns.items():
        for fp in deriv_root.glob(globpat):
            sub, ses = parse_sub_ses(fp)
            if sub.upper().startswith("PILOT"):
                continue
            rows.append(
                {"measure": meas, "sub": sub, "ses": ses, "path": fp}
            )

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(["measure", "sub", "ses", "path"]).reset_index(drop=True)
    return df


def compute_icc_fallback(y, subs, sess):
    subs_u, si = np.unique(subs, return_inverse=True)
    sess_u, sj = np.unique(sess, return_inverse=True)
    n, k = len(subs_u), len(sess_u)
    if n < 2 or k < 2:
        return np.nan

    M = np.full((n, k), np.nan)
    for v, i, j in zip(y, si, sj):
        M[i, j] = v

    M = M[~np.any(np.isnan(M), axis=1)]
    if M.shape[0] < 2:
        return np.nan

    n = M.shape[0]
    GM = M.mean()
    rm = M.mean(axis=1)
    cm = M.mean(axis=0)

    SSR = k * np.sum((rm - GM) ** 2)
    SSC = n * np.sum((cm - GM) ** 2)
    SSE = np.sum((M - GM) ** 2) - SSR - SSC

    MSR = SSR / (n - 1)
    MSC = SSC / (k - 1)
    MSE = SSE / ((n - 1) * (k - 1))

    denom = MSR + (k - 1) * MSE + k * (MSC - MSE) / n
    if denom == 0:
        return np.nan

    return (MSR - MSE) / denom


# ==========================
# Atlas cache (grid-safe)
# ==========================
_ATLAS_CACHE = {}
_ATLAS_CACHE_LOCK = Lock()


def _grid_key(img: nib.spatialimages.SpatialImage):
    return (
        tuple(img.shape),
        tuple(np.round(img.affine, 5).ravel()),
    )


def get_resampled_atlases(metric_img, wm_atlas, lh_atlas, rh_atlas):
    """
    Resample atlases to the metric image grid, caching by shape + affine.
    This fixes mixed-grid datasets without resampling thousands of times.
    """
    key = _grid_key(metric_img)

    with _ATLAS_CACHE_LOCK:
        cached = _ATLAS_CACHE.get(key)
    if cached is not None:
        return cached

    wm_labels = resample_to_img(
        wm_atlas,
        metric_img,
        interpolation="nearest",
        force_resample=True,
    ).get_fdata().astype(np.int32)

    lh_labels = resample_to_img(
        lh_atlas,
        metric_img,
        interpolation="nearest",
        force_resample=True,
    ).get_fdata().astype(np.int32)

    rh_labels = resample_to_img(
        rh_atlas,
        metric_img,
        interpolation="nearest",
        force_resample=True,
    ).get_fdata().astype(np.int32)

    result = (wm_labels, lh_labels, rh_labels)

    with _ATLAS_CACHE_LOCK:
        _ATLAS_CACHE.setdefault(key, result)
        return _ATLAS_CACHE[key]


# ==========================
# Tissue-specific extraction
# ==========================
def parcel_means(data: np.ndarray, labels: np.ndarray):
    """
    Vectorized parcel means with NaN-safe behavior:
    only finite voxels contribute, matching np.nanmean semantics.
    """
    if data.shape != labels.shape:
        raise ValueError(
            f"Shape mismatch: metric data {data.shape} vs labels {labels.shape}"
        )

    mask = (labels > 0) & np.isfinite(data)
    if not np.any(mask):
        return np.array([], dtype=int), np.array([], dtype=float)

    labs = labels[mask].astype(np.int64, copy=False)
    vals = np.asarray(data[mask], dtype=np.float64)

    sums = np.bincount(labs, weights=vals)
    counts = np.bincount(labs)

    means = np.divide(
        sums,
        counts,
        out=np.full_like(sums, np.nan, dtype=np.float64),
        where=counts > 0,
    )

    parcels = np.arange(len(means), dtype=int)
    valid = (parcels > 0) & (counts > 0)

    return parcels[valid], means[valid]


def extract_wm_means(data: np.ndarray, labels: np.ndarray) -> pd.DataFrame:
    parcels, means = parcel_means(data, labels)
    return pd.DataFrame(
        {
            "parcel": [f"WM-{int(p)}" for p in parcels],
            "value": means,
        }
    )


def extract_gm_means(
    data: np.ndarray, lh_labels: np.ndarray, rh_labels: np.ndarray
) -> pd.DataFrame:
    rows = []

    for hemi, labs in (("L", lh_labels), ("R", rh_labels)):
        parcels, means = parcel_means(data, labs)
        rows.append(
            pd.DataFrame(
                {
                    "parcel": [f"{hemi}-{int(p)}" for p in parcels],
                    "value": means,
                }
            )
        )

    if rows:
        return pd.concat(rows, ignore_index=True)
    return pd.DataFrame(columns=["parcel", "value"])


def process_file(row, wm_atlas, lh_atlas, rh_atlas):
    """
    Worker for a single metric image.
    Safe for parallel use with ThreadPoolExecutor.
    """
    img = nib.load(row.path)
    data = np.asarray(img.dataobj)

    if data.ndim != 3:
        raise ValueError(f"Expected 3D image, got shape {data.shape} for {row.path}")

    wm_labels, lh_labels, rh_labels = get_resampled_atlases(
        img, wm_atlas, lh_atlas, rh_atlas
    )

    wm = extract_wm_means(data, wm_labels)
    gm = extract_gm_means(data, lh_labels, rh_labels)

    for df in (wm, gm):
        df["sub"] = row.sub
        df["ses"] = row.ses
        df["measure"] = row.measure

    return wm, gm


# ==========================
# ICC + plotting
# ==========================
def compute_icc_table(df_means: pd.DataFrame) -> pd.DataFrame:
    icc_rows = []

    for (meas, parc), dfg in df_means.groupby(["measure", "parcel"]):
        dfg = dfg[dfg["ses"].notna()].copy()
        if dfg.empty:
            continue

        valid_subs = dfg.groupby("sub")["ses"].nunique()
        valid_subs = valid_subs[valid_subs >= 2].index
        dfg = dfg[dfg["sub"].isin(valid_subs)]

        if dfg.empty:
            continue

        y = dfg["value"].to_numpy()
        subs = dfg["sub"].to_numpy()
        sess = dfg["ses"].to_numpy()

        if _HAVE_PG:
            try:
                tab = pd.DataFrame(
                    {"targets": subs, "raters": sess, "scores": y}
                ).dropna()

                if (
                    tab["targets"].nunique() < 2
                    or tab["raters"].nunique() < 2
                ):
                    icc = np.nan
                else:
                    icc = (
                        pg.intraclass_corr(
                            tab,
                            targets="targets",
                            raters="raters",
                            ratings="scores",
                        )
                        .query("Type=='ICC2'")["ICC"]
                        .iloc[0]
                    )
            except Exception:
                icc = compute_icc_fallback(y, subs, sess)
        else:
            icc = compute_icc_fallback(y, subs, sess)

        icc_rows.append(
            {
                "measure": meas,
                "parcel": parc,
                "ICC2_1": icc,
                "n_subjects": dfg["sub"].nunique(),
                "n_sessions": dfg["ses"].nunique(),
            }
        )

    return pd.DataFrame(icc_rows)


def plot_icc_outputs(df_icc, figs: Path, tissue_tag: str, tissue_label: str):
    if df_icc.empty:
        print(f"[WARN] No ICC values to plot for {tissue_tag}")
        return

    # Heatmap
    piv = df_icc.pivot(index="measure", columns="parcel", values="ICC2_1")
    piv = piv.loc[piv.mean(axis=1).sort_values(ascending=False).index]

    plt.figure(figsize=(12, 6))
    plt.imshow(piv, aspect="auto", vmin=0, vmax=1)
    plt.colorbar(label="ICC(2,1)")
    plt.yticks(range(len(piv.index)), piv.index)
    plt.xticks(range(len(piv.columns)), piv.columns, rotation=90)
    plt.title(f"ICC heatmap ({tissue_label})\nRows ordered by mean ICC")
    plt.tight_layout()
    plt.savefig(figs / f"icc_heatmap_mni_{tissue_tag}.png", dpi=150)
    plt.close()

    # Violin
    mean_icc = (
        df_icc.groupby("measure")["ICC2_1"]
        .mean()
        .sort_values(ascending=False)
    )
    measures = []
    data = []
    for m in mean_icc.index.tolist():
        vals = df_icc.loc[df_icc["measure"] == m, "ICC2_1"].dropna().values
        if len(vals) > 0:
            measures.append(m)
            data.append(vals)

    if not data:
        print(f"[WARN] No non-NaN ICC values for violin plot: {tissue_tag}")
        return

    fig, ax = plt.subplots(figsize=(max(12, 1.2 * len(measures)), 6))
    vp = ax.violinplot(data, showmedians=True, showextrema=False)

    for body, meas in zip(vp["bodies"], measures):
        model = MEASURE_TO_MODEL.get(meas, "Other")
        body.set_facecolor(MODEL_COLORS.get(model, "#cccccc"))
        body.set_edgecolor("black")
        body.set_alpha(0.85)

    ax.set_xticks(range(1, len(measures) + 1))
    ax.set_xticklabels(measures, rotation=45, ha="right")
    ax.set_ylim(0, 1)
    ax.set_ylabel("ICC(2,1)")
    ax.set_title(f"Parcel-wise ICC distributions ({tissue_label})")

    legend = [
        Patch(facecolor=c, edgecolor="black", label=m)
        for m, c in MODEL_COLORS.items()
    ]
    ax.legend(
        handles=legend,
        title="Model",
        bbox_to_anchor=(1.01, 1),
        loc="upper left",
    )

    plt.tight_layout()
    plt.savefig(figs / f"icc_violins_mni_{tissue_tag}_by_model.png", dpi=300)
    plt.close()


# ==========================
# Main
# ==========================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", required=True)
    ap.add_argument(
        "--n-jobs",
        type=int,
        default=1,
        help="Number of worker threads for image extraction. "
             "On shared storage, 8-12 is often faster than 32.",
    )
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    print("[INFO] Collecting metric files")
    df_files = collect_metric_files(PATTERNS_MNI, DERIV_ROOT)
    if df_files.empty:
        raise RuntimeError("No metric files found")

    print(f"[INFO] Found {len(df_files)} files")

    print("[INFO] Loading atlases once")
    wm_atlas = nib.load(WM_LABEL)
    lh_atlas = nib.load(LH_LABEL)
    rh_atlas = nib.load(RH_LABEL)

    print(f"[INFO] Extracting parcel means with n_jobs={args.n_jobs}")

    rows_wm = []
    rows_gm = []

    file_rows = list(df_files.itertuples(index=False))

    if args.n_jobs == 1:
        iterator = tqdm(file_rows, total=len(file_rows))
        for row in iterator:
            wm, gm = process_file(row, wm_atlas, lh_atlas, rh_atlas)
            rows_wm.append(wm)
            rows_gm.append(gm)
    else:
        with ThreadPoolExecutor(max_workers=max(1, args.n_jobs)) as ex:
            futures = {
                ex.submit(process_file, row, wm_atlas, lh_atlas, rh_atlas): row
                for row in file_rows
            }
            for fut in tqdm(as_completed(futures), total=len(futures)):
                wm, gm = fut.result()
                rows_wm.append(wm)
                rows_gm.append(gm)

    df_wm = pd.concat(rows_wm, ignore_index=True)
    df_gm = pd.concat(rows_gm, ignore_index=True)

    df_wm.to_csv(outdir / "parcel_means_long_mni_wm.csv", index=False)
    df_gm.to_csv(outdir / "parcel_means_long_mni_gm.csv", index=False)

    print("[INFO] Computing ICC for WM")
    icc_wm = compute_icc_table(df_wm)
    icc_wm.to_csv(outdir / "icc_summary_mni_wm.csv", index=False)

    print("[INFO] Computing ICC for GM")
    icc_gm = compute_icc_table(df_gm)
    icc_gm.to_csv(outdir / "icc_summary_mni_gm.csv", index=False)

    figs_wm = outdir / "figs_wm"
    figs_gm = outdir / "figs_gm"
    figs_wm.mkdir(exist_ok=True)
    figs_gm.mkdir(exist_ok=True)

    print("[INFO] Plotting WM outputs")
    plot_icc_outputs(icc_wm, figs_wm, "wm", "White Matter (JHU)")

    print("[INFO] Plotting GM outputs")
    plot_icc_outputs(icc_gm, figs_gm, "gm", "Gray Matter (Vogt)")

    print("[DONE]")


if __name__ == "__main__":
    main()
