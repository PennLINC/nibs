"""Compute parcel-wise summary statistics for scalar maps.

Runs per subject, writing one CSV per subject/session/run with one row per
aparc.a2009s parcel and columns of the form {METRIC}_{STAT}.
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
    "FA": "qsirecon/derivatives/qsirecon-DIPYDKI/sub-*/ses-*/dwi/sub-*_ses-*_acq-HBCD75_run-01_space-ACPC_model-tensor_param-fa_dwimap.nii.gz",
    "KFA": "qsirecon/derivatives/qsirecon-DIPYDKI/sub-*/ses-*/dwi/sub-*_ses-*_acq-HBCD75_run-01_space-ACPC_model-dki_param-kfa_dwimap.nii.gz",
    "KFA-Micro": "qsirecon/derivatives/qsirecon-DIPYDKI/sub-*/ses-*/dwi/sub-*_ses-*_acq-HBCD75_run-01_space-ACPC_model-dkimicro_param-kfa_dwimap.nii.gz",
    "AD": "qsirecon/derivatives/qsirecon-DIPYDKI/sub-*/ses-*/dwi/sub-*_ses-*_acq-HBCD75_run-01_space-ACPC_model-dki_param-ad_dwimap.nii.gz",
    "AD-Micro": "qsirecon/derivatives/qsirecon-DIPYDKI/sub-*/ses-*/dwi/sub-*_ses-*_acq-HBCD75_run-01_space-ACPC_model-dkimicro_param-ad_dwimap.nii.gz",
    "ADE-Micro": "qsirecon/derivatives/qsirecon-DIPYDKI/sub-*/ses-*/dwi/sub-*_ses-*_acq-HBCD75_run-01_space-ACPC_model-dkimicro_param-ade_dwimap.nii.gz",
    "AWF-Micro": "qsirecon/derivatives/qsirecon-DIPYDKI/sub-*/ses-*/dwi/sub-*_ses-*_acq-HBCD75_run-01_space-ACPC_model-dkimicro_param-awf_dwimap.nii.gz",
    "AxonALD-Micro": "qsirecon/derivatives/qsirecon-DIPYDKI/sub-*/ses-*/dwi/sub-*_ses-*_acq-HBCD75_run-01_space-ACPC_model-dkimicro_param-axonald_dwimap.nii.gz",
    "AK": "qsirecon/derivatives/qsirecon-DIPYDKI/sub-*/ses-*/dwi/sub-*_ses-*_acq-HBCD75_run-01_space-ACPC_model-dki_param-ak_dwimap.nii.gz",
    "MD": "qsirecon/derivatives/qsirecon-DIPYDKI/sub-*/ses-*/dwi/sub-*_ses-*_acq-HBCD75_run-01_space-ACPC_model-dki_param-md_dwimap.nii.gz",
    "MD-Micro": "qsirecon/derivatives/qsirecon-DIPYDKI/sub-*/ses-*/dwi/sub-*_ses-*_acq-HBCD75_run-01_space-ACPC_model-dkimicro_param-md_dwimap.nii.gz",
    "MK": "qsirecon/derivatives/qsirecon-DIPYDKI/sub-*/ses-*/dwi/sub-*_ses-*_acq-HBCD75_run-01_space-ACPC_model-dki_param-mk_dwimap.nii.gz",
    "MKT": "qsirecon/derivatives/qsirecon-DIPYDKI/sub-*/ses-*/dwi/sub-*_ses-*_acq-HBCD75_run-01_space-ACPC_model-dki_param-mkt_dwimap.nii.gz",
    "RD": "qsirecon/derivatives/qsirecon-DIPYDKI/sub-*/ses-*/dwi/sub-*_ses-*_acq-HBCD75_run-01_space-ACPC_model-dki_param-rd_dwimap.nii.gz",
    "RD-Micro": "qsirecon/derivatives/qsirecon-DIPYDKI/sub-*/ses-*/dwi/sub-*_ses-*_acq-HBCD75_run-01_space-ACPC_model-dkimicro_param-rd_dwimap.nii.gz",
    "RDE-Micro": "qsirecon/derivatives/qsirecon-DIPYDKI/sub-*/ses-*/dwi/sub-*_ses-*_acq-HBCD75_run-01_space-ACPC_model-dkimicro_param-rde_dwimap.nii.gz",
    "RK": "qsirecon/derivatives/qsirecon-DIPYDKI/sub-*/ses-*/dwi/sub-*_ses-*_acq-HBCD75_run-01_space-ACPC_model-dki_param-rk_dwimap.nii.gz",
    "Linearity": "qsirecon/derivatives/qsirecon-DIPYDKI/sub-*/ses-*/dwi/sub-*_ses-*_acq-HBCD75_run-01_space-ACPC_model-dki_param-linearity_dwimap.nii.gz",
    "Planarity": "qsirecon/derivatives/qsirecon-DIPYDKI/sub-*/ses-*/dwi/sub-*_ses-*_acq-HBCD75_run-01_space-ACPC_model-dki_param-planarity_dwimap.nii.gz",
    "Sphericity": "qsirecon/derivatives/qsirecon-DIPYDKI/sub-*/ses-*/dwi/sub-*_ses-*_acq-HBCD75_run-01_space-ACPC_model-dki_param-sphericity_dwimap.nii.gz",
    "Trace-Micro": "qsirecon/derivatives/qsirecon-DIPYDKI/sub-*/ses-*/dwi/sub-*_ses-*_acq-HBCD75_run-01_space-ACPC_model-dkimicro_param-trace_dwimap.nii.gz",
    "Tortuosity-Micro": "qsirecon/derivatives/qsirecon-DIPYDKI/sub-*/ses-*/dwi/sub-*_ses-*_acq-HBCD75_run-01_space-ACPC_model-dkimicro_param-tortuosity_dwimap.nii.gz",
    # DWI NODDI
    "ICVF-Modulated": "qsirecon/derivatives/qsirecon-NODDI/sub-*/ses-*/dwi/sub-*_ses-*_acq-HBCD75_run-01_space-ACPC_model-noddi_param-icvf_desc-modulated_dwimap.nii.gz",
    "ICVF": "qsirecon/derivatives/qsirecon-NODDI/sub-*/ses-*/dwi/sub-*_ses-*_acq-HBCD75_run-01_space-ACPC_model-noddi_param-icvf_dwimap.nii.gz",
    "ISOVF": "qsirecon/derivatives/qsirecon-NODDI/sub-*/ses-*/dwi/sub-*_ses-*_acq-HBCD75_run-01_space-ACPC_model-noddi_param-isovf_dwimap.nii.gz",
    "NRMSE": "qsirecon/derivatives/qsirecon-NODDI/sub-*/ses-*/dwi/sub-*_ses-*_acq-HBCD75_run-01_space-ACPC_model-noddi_param-nrmse_dwimap.nii.gz",
    "RMSE": "qsirecon/derivatives/qsirecon-NODDI/sub-*/ses-*/dwi/sub-*_ses-*_acq-HBCD75_run-01_space-ACPC_model-noddi_param-rmse_dwimap.nii.gz",
    "OD-Modulated": "qsirecon/derivatives/qsirecon-NODDI/sub-*/ses-*/dwi/sub-*_ses-*_acq-HBCD75_run-01_space-ACPC_model-noddi_param-od_desc-modulated_dwimap.nii.gz",
    "OD": "qsirecon/derivatives/qsirecon-NODDI/sub-*/ses-*/dwi/sub-*_ses-*_acq-HBCD75_run-01_space-ACPC_model-noddi_param-od_dwimap.nii.gz",
    "TF": "qsirecon/derivatives/qsirecon-NODDI/sub-*/ses-*/dwi/sub-*_ses-*_acq-HBCD75_run-01_space-ACPC_model-noddi_param-tf_dwimap.nii.gz",
    # DWI MAPMRI
    "NG": "qsirecon/derivatives/qsirecon-TORTOISE_model-MAPMRI/sub-*/ses-*/dwi/sub-*_ses-*_acq-HBCD75_run-01_space-ACPC_model-mapmri_param-ng_dwimap.nii.gz",
    "NGPar": "qsirecon/derivatives/qsirecon-TORTOISE_model-MAPMRI/sub-*/ses-*/dwi/sub-*_ses-*_acq-HBCD75_run-01_space-ACPC_model-mapmri_param-ngpar_dwimap.nii.gz",
    "NGPerp": "qsirecon/derivatives/qsirecon-TORTOISE_model-MAPMRI/sub-*/ses-*/dwi/sub-*_ses-*_acq-HBCD75_run-01_space-ACPC_model-mapmri_param-ngperp_dwimap.nii.gz",
    "PA": "qsirecon/derivatives/qsirecon-TORTOISE_model-MAPMRI/sub-*/ses-*/dwi/sub-*_ses-*_acq-HBCD75_run-01_space-ACPC_model-mapmri_param-pa_dwimap.nii.gz",
    "PAth": "qsirecon/derivatives/qsirecon-TORTOISE_model-MAPMRI/sub-*/ses-*/dwi/sub-*_ses-*_acq-HBCD75_run-01_space-ACPC_model-mapmri_param-path_dwimap.nii.gz",
    "RTAP": "qsirecon/derivatives/qsirecon-TORTOISE_model-MAPMRI/sub-*/ses-*/dwi/sub-*_ses-*_acq-HBCD75_run-01_space-ACPC_model-mapmri_param-rtap_dwimap.nii.gz",
    "RTOP": "qsirecon/derivatives/qsirecon-TORTOISE_model-MAPMRI/sub-*/ses-*/dwi/sub-*_ses-*_acq-HBCD75_run-01_space-ACPC_model-mapmri_param-rtop_dwimap.nii.gz",
    "RTPP": "qsirecon/derivatives/qsirecon-TORTOISE_model-MAPMRI/sub-*/ses-*/dwi/sub-*_ses-*_acq-HBCD75_run-01_space-ACPC_model-mapmri_param-rtpp_dwimap.nii.gz",
    # QSM
    "MEGRE": "qsm/sub-*/ses-*/anat/sub-*_ses-*_acq-QSM_run-01_space-T1w_desc-mean_MEGRE.nii.gz",
    "QSM-SEPIA-E5": "qsm/sub-*/ses-*/anat/*_space-T1w_desc-E12345+sepia_Chimap.nii.gz",
    "QSM-X-R2p-E5-X": "qsm/sub-*/ses-*/anat/*_space-T1w_desc-E12345+chisep+r2p_Chimap.nii.gz",
    "QSM-X-R2p-E5-Para": "qsm/sub-*/ses-*/anat/*_space-T1w_desc-E12345+chisep+r2p_ironw.nii.gz",
    "QSM-X-R2p-E5-Dia": "qsm/sub-*/ses-*/anat/*_space-T1w_desc-E12345+chisep+r2p_myelinw.nii.gz",
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
    #"G-MPRAGE-MyelinW": "g_ratio/sub-*/ses-*/anat/sub-*_ses-*_run-01_space-T1w_desc-MPRAGET1wT2w+ISOVF+ICVF_gratio.nii.gz",
    #"G-SPACE-MyelinW": "g_ratio/sub-*/ses-*/anat/sub-*_ses-*_run-01_space-T1w_desc-SPACET1wT2w+ISOVF+ICVF_gratio.nii.gz",
    #"G-ihMTsat": "g_ratio/sub-*/ses-*/anat/sub-*_ses-*_run-01_space-T1w_desc-MTsat+ISOVF+ICVF_gratio.nii.gz",
    #"G-ihMTR": "g_ratio/sub-*/ses-*/anat/sub-*_ses-*_run-01_space-T1w_desc-ihMTR+ISOVF+ICVF_gratio.nii.gz",
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


def process_subject(subject: str, deriv_dir: str) -> None:
    t1w_reg_dir = os.path.join(deriv_dir, "t1w_registration", f"sub-{subject}", "anat")
    out_dir = os.path.join(deriv_dir, "parcel_bundle_stats", f"sub-{subject}")
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
            dseg_data = dseg_arrays[space]

            map_img = ants.image_read(metric_file)
            if map_img.shape != dseg_img.shape:
                map_img = ants.apply_transforms(
                    fixed=dseg_img,
                    moving=map_img,
                    transformlist=[],
                    interpolator="linear",
                )
            map_data = map_img.numpy()

            for label_id in label_ids:
                voxels = map_data[dseg_data == label_id]
                voxels = voxels[np.isfinite(voxels)]
                stats = _compute_stats(voxels)
                row_idx = out_df["parcel_intensity"] == label_id
                for stat_name, stat_val in stats.items():
                    out_df.loc[row_idx, f"{metric_name}_{stat_name}"] = stat_val

        out_file = os.path.join(
            out_dir,
            f"sub-{subject}_{ses}_{run}_desc-a2009s_scalarstats.csv",
        )
        out_df.to_csv(out_file, index=False)
        print(f"Wrote {out_file}", flush=True)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--subject-id",
        required=True,
        help="Subject ID without the sub- prefix",
    )
    return parser


if __name__ == "__main__":
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
    from config import load_config

    args = _build_parser().parse_args()
    cfg = load_config()
    derivatives_dir = os.path.join(cfg["project_root"], "derivatives")
    process_subject(args.subject_id, derivatives_dir)
