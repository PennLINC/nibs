#!/usr/bin/env python3
"""Generate QSIRecon-style scalarstats TSVs for warped T1w bundles."""

from __future__ import annotations

import argparse
import os
import re
from glob import glob
from pathlib import Path

import pandas as pd

from bundle_scalar_mapping_utils import summarize_bundles

# Myelin metrics
METRIC_PATTERNS_T1W: dict[str, str] = {
    "MEGRE": "qsm/sub-*/ses-*/anat/sub-*_ses-*_acq-QSM_run-01_space-T1w_desc-mean_MEGRE.nii.gz",
    "QSM-SEPIA-E5": "qsm/sub-*/ses-*/anat/*_space-T1w_desc-E12345+sepia_Chimap.nii.gz",
    "QSM-X-R2p-E5-X": "qsm/sub-*/ses-*/anat/*_space-T1w_desc-E12345+chisep+r2p_Chimap.nii.gz",
    "QSM-X-R2p-E5-Para": "qsm/sub-*/ses-*/anat/*_space-T1w_desc-E12345+chisep+r2p_ironw.nii.gz",
    "QSM-X-R2p-E5-Dia": "qsm/sub-*/ses-*/anat/*_space-T1w_desc-E12345+chisep+r2p_myelinw.nii.gz",
    "ihMTw": "ihmt/sub-*/ses-*/anat/sub-*_ses-*_run-01_space-T1w_ihMTw.nii.gz",
    "ihMTR": "ihmt/sub-*/ses-*/anat/sub-*_ses-*_run-01_space-T1w_ihMTR.nii.gz",
    "MTR": "ihmt/sub-*/ses-*/anat/sub-*_ses-*_run-01_space-T1w_MTRmap.nii.gz",
    "ihMTsat": "ihmt/sub-*/ses-*/anat/sub-*_ses-*_run-01_space-T1w_ihMTsat.nii.gz",
    "ihMTsat-B1c": "ihmt/sub-*/ses-*/anat/sub-*_ses-*_run-01_space-T1w_ihMTsatB1sq.nii.gz",
    "R1": "pymp2rage/sub-*/ses-*/anat/sub-*_ses-*_run-01_space-T1w_R1map.nii.gz",
    "R1-B1c": "pymp2rage/sub-*/ses-*/anat/sub-*_ses-*_run-01_space-T1w_desc-B1corrected_R1map.nii.gz",
    "MPRAGE-MyelinW": "t1wt2w_ratio/sub-*/ses-*/anat/sub-*_ses-*_run-01_space-T1w_desc-MPRAGEunscaled_myelinw.nii.gz",
    "SPACE-MyelinW": "t1wt2w_ratio/sub-*/ses-*/anat/sub-*_ses-*_run-01_space-T1w_desc-SPACEunscaled_myelinw.nii.gz",
    "Scaled MPRAGE-MyelinW": "t1wt2w_ratio/sub-*/ses-*/anat/sub-*_ses-*_run-01_space-T1w_desc-MPRAGEscaled_myelinw.nii.gz",
    "Scaled SPACE-MyelinW": "t1wt2w_ratio/sub-*/ses-*/anat/sub-*_ses-*_run-01_space-T1w_desc-SPACEscaled_myelinw.nii.gz",
}

BUNDLE_RE = re.compile(r"_bundle-(?P<bundle>.+?)_streamlines\.tck(?:\.gz)?$")
UNDERSCORE_PREFIXES = (
    "ProjectionBasalGanglia",
    "ProjectionBrainstem",
    "Association",
    "Cerebellum",
    "Commissure",
    "CranialNerve",
)


def _extract_bundle_name(path: str) -> str:
    match = BUNDLE_RE.search(os.path.basename(path))
    if not match:
        raise ValueError(f"Could not parse bundle name from {path}")
    bundle = match.group("bundle")
    for prefix in UNDERSCORE_PREFIXES:
        if bundle.startswith(prefix + "_"):
            return bundle
        if bundle.startswith(prefix) and len(bundle) > len(prefix):
            return prefix + "_" + bundle[len(prefix) :]
    return bundle


def _resolve_scalar_specs(deriv_dir: str, subject: str, session: str) -> list[dict[str, str]]:
    scalar_specs: list[dict[str, str]] = []
    for metric_name, pattern in METRIC_PATTERNS_T1W.items():
        subj_pattern = pattern.replace("sub-*", f"sub-{subject}").replace("ses-*", session)
        matches = sorted(glob(os.path.join(deriv_dir, subj_pattern)))
        if not matches:
            print(f"[WARN] Missing scalar for {metric_name}: {subj_pattern}", flush=True)
            continue
        if len(matches) > 1:
            print(f"[WARN] Multiple scalar matches for {metric_name}; using first: {matches[0]}", flush=True)
        scalar_specs.append(
            {
                "variable_name": metric_name,
                "path": matches[0],
                "source_file": matches[0],
                "qsirecon_suffix": "myelin_t1w",
            }
        )
    return scalar_specs


def _finalize_qsirecon_style_tsv(
    bundle_stats_file: str,
    out_file: str,
    subject: str,
    session: str,
    bundle_source: str,
    bundle_params_id: str,
) -> None:
    df = pd.read_csv(bundle_stats_file, sep="\t")

    # Ensure QSIRecon-style metadata columns exist.
    df["subject_id"] = f"sub-{subject}"
    df["session_id"] = session
    df["task_id"] = pd.NA
    df["dir_id"] = pd.NA
    df["acq_id"] = "HBCD75"
    df["space_id"] = "T1w"
    df["rec_id"] = pd.NA
    df["run_id"] = "01"
    df["bundle_source"] = bundle_source
    df["bundle_params_id"] = bundle_params_id

    ordered_cols = [
        "bundle",
        "variable_name",
        "qsirecon_suffix",
        "source_file",
        "zero_proportion",
        "mean",
        "stdev",
        "median",
        "masked_mean",
        "masked_median",
        "masked_stdev",
        "weighted_mean",
        "masked_weighted_mean",
        "subject_id",
        "session_id",
        "task_id",
        "dir_id",
        "acq_id",
        "space_id",
        "rec_id",
        "run_id",
        "bundle_source",
        "bundle_params_id",
    ]
    for col in ordered_cols:
        if col not in df.columns:
            df[col] = pd.NA

    df = df[ordered_cols]
    Path(out_file).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_file, sep="\t", index=False)


def process_subject(
    subject: str,
    deriv_dir: str,
    out_root: str,
    bundle_source: str,
    bundle_params_id: str,
) -> None:
    bundles_root = os.path.join(deriv_dir, "warped_bundles", f"sub-{subject}")
    if not os.path.isdir(bundles_root):
        print(f"[WARN] No warped bundles directory for sub-{subject}: {bundles_root}", flush=True)
        return

    session_dirs = sorted(glob(os.path.join(bundles_root, "ses-*")))
    if not session_dirs:
        print(f"[WARN] No sessions found for sub-{subject} in {bundles_root}", flush=True)
        return

    for session_dir in session_dirs:
        session = os.path.basename(session_dir)
        dwi_dir = os.path.join(session_dir, "dwi")
        if not os.path.isdir(dwi_dir):
            print(f"[WARN] Missing dwi directory for sub-{subject} {session}", flush=True)
            continue

        tck_files = sorted(
            glob(
                os.path.join(
                    dwi_dir,
                    f"sub-{subject}_{session}_acq-HBCD75_run-01_space-T1w_model-gqi_bundle-*_streamlines.tck",
                )
            )
        )
        if not tck_files:
            tck_files = sorted(
                glob(
                    os.path.join(
                        dwi_dir,
                        f"sub-{subject}_{session}_acq-HBCD75_run-01_space-T1w_model-gqi_bundle-*_streamlines.tck.gz",
                    )
                )
            )
        if not tck_files:
            print(f"[WARN] No warped bundle TCK files for sub-{subject} {session}", flush=True)
            continue

        bundle_names = [_extract_bundle_name(tck_path) for tck_path in tck_files]
        scalar_specs = _resolve_scalar_specs(deriv_dir, subject, session)
        if not scalar_specs:
            print(f"[WARN] No scalar maps found for sub-{subject} {session}", flush=True)
            continue

        # Use first scalar image as the tckmap template (all are in T1w space).
        dwiref = scalar_specs[0]["path"]
        out_dir = os.path.join(out_root, f"sub-{subject}", session, "dwi")
        bundle_stats_file, _ = summarize_bundles(
            dwiref_image=dwiref,
            tck_files=tck_files,
            bundle_names=bundle_names,
            scalar_specs=scalar_specs,
            out_dir=out_dir,
            bundle_source=bundle_source,
            bundle_params_id=bundle_params_id,
        )

        final_tsv = os.path.join(
            out_dir,
            f"sub-{subject}_{session}_acq-HBCD75_run-01_space-T1w_model-gqi_scalarstats.tsv",
        )
        _finalize_qsirecon_style_tsv(
            bundle_stats_file=bundle_stats_file,
            out_file=final_tsv,
            subject=subject,
            session=session,
            bundle_source=bundle_source,
            bundle_params_id=bundle_params_id,
        )
        print(f"[INFO] Wrote {final_tsv}", flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subject-id", required=True, help="Subject ID without sub- prefix.")
    parser.add_argument(
        "--derivatives-dir",
        default="/cbica/projects/nibs/derivatives",
        help="Derivatives root directory.",
    )
    parser.add_argument(
        "--out-root",
        default="/cbica/projects/nibs/derivatives/bundle_myelin_stats",
        help="Output root for scalarstats TSVs.",
    )
    parser.add_argument(
        "--bundle-source",
        default="warped_gqi",
        help="Value for bundle_source column.",
    )
    parser.add_argument(
        "--bundle-params-id",
        default="default",
        help="Value for bundle_params_id column.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    process_subject(
        subject=args.subject_id,
        deriv_dir=args.derivatives_dir,
        out_root=args.out_root,
        bundle_source=args.bundle_source,
        bundle_params_id=args.bundle_params_id,
    )


if __name__ == "__main__":
    main()
