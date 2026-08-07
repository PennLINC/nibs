#!/usr/bin/env python3
"""Compute voxelwise ICC(2,1) maps in MNI space.

This workflow creates one reliability map per metric rather than separate GM,
WM, and GM+WM ICC maps. Fixed group-level GM and WM masks are built once
from template-space GM and WM probability maps. The analysis mask is the union
of the non-overlapping GM and WM masks.

For each selected metric, the script pairs two sessions within subject and
computes voxelwise ICC(2,1): two-way random effects, absolute agreement, single
measurement. Two analyses are written:

1. Primary: all complete subject pairs that pass scan-level QC.
2. Paired-MAD sensitivity: removes an entire subject pair at a voxel when the
   subject's across-session mean or session difference is an extreme robust
   outlier across subjects at that voxel.

Paired outliers are evaluated separately inside the eroded GM and WM
compartments. The compartment-specific results are then combined into one ICC
map. Thus, GM and WM intensity distributions are never pooled for outlier
determination. Because detection is voxelwise across subjects, the paired
design is preserved and only complete subject pairs are removed.

The same final ICC map is summarized within the fixed eroded GM mask, fixed
eroded WM mask, and their union. Ranked median/IQR figures are colored by source
image.

The implementation avoids ``from __future__ import annotations`` and newer
union/generic annotation syntax for compatibility with older Python 3
environments.
"""

import argparse
import hashlib
import json
import re
import warnings
from collections import OrderedDict, namedtuple
from pathlib import Path

try:
    import numpy as np
except ImportError:
    np = None

try:
    import pandas as pd
except ImportError:
    pd = None

try:
    import nibabel as nib
    from nibabel.processing import resample_from_to
except ImportError:
    nib = None
    resample_from_to = None

try:
    from scipy.ndimage import distance_transform_edt
except ImportError:
    distance_transform_edt = None

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
except ImportError:
    matplotlib = None
    plt = None
    Line2D = None


SubjectPairInputs = namedtuple(
    "SubjectPairInputs",
    ["subject", "metric_a", "metric_b"],
)
HybridSubjectPairInputs = namedtuple(
    "HybridSubjectPairInputs",
    [
        "subject",
        "metric_a",
        "metric_b",
        "gm_metric_a",
        "gm_metric_b",
    ],
)
DsegPairInputs = namedtuple(
    "DsegPairInputs",
    ["subject", "dseg_a", "dseg_b"],
)


try:
    from metric_registry import (
        SOURCE_IMAGE_COLORS,
        build_metric_specs,
        gm_noddi_hybrid_pairs,
        metric_display_labels,
        metric_order,
    )
except ImportError:
    SOURCE_IMAGE_COLORS = None
    build_metric_specs = None
    gm_noddi_hybrid_pairs = None
    metric_display_labels = None
    metric_order = None


ANALYSIS_SETS = (
    "primary",
    "full",
    "both",
)

SPACE = "MNI152NLin2009cAsym"
ROBUST_NORMAL_SCALE = 0.67448975
SUMMARY_TISSUES = ("gm", "wm", "gmwm")
TISSUE_LABELS = {
    "gm": "GM",
    "wm": "WM",
    "gmwm": "GM+WM",
}


def require_dependencies():
    missing = []

    for name, module in (
        ("nibabel", nib),
        ("numpy", np),
        ("pandas", pd),
        ("scipy", distance_transform_edt),
        ("matplotlib", matplotlib),
        ("metric_registry", build_metric_specs),
    ):
        if module is None:
            missing.append(name)

    if missing:
        raise RuntimeError(
            "Missing required Python packages: {0}. Activate the NIBS "
            "processing environment first.".format(", ".join(missing))
        )


def normalize_subject(value):
    token = str(value).strip()

    if token.startswith("sub-"):
        return token

    return "sub-{0}".format(token)


def normalize_session(value):
    token = str(value).strip()

    if token.startswith("ses-"):
        return token

    return "ses-{0}".format(token)


def subject_for_qc(subject):
    return re.sub(r"^sub-", "", str(subject).strip())


def is_pilot_subject(subject):
    return subject_for_qc(subject).upper().startswith("PILOT")


def session_label(session):
    match = re.search(r"(\d+)", str(session))

    if match is None:
        raise ValueError(
            "Could not parse session number from {0}".format(session)
        )

    return "Session {0:02d}".format(int(match.group(1)))


def safe_label(value):
    return re.sub(
        r"[^A-Za-z0-9]+",
        "-",
        str(value),
    ).strip("-")


def metric_slug(metric_spec):
    base = safe_label(metric_spec.label)
    digest = hashlib.sha1(
        str(metric_spec.pattern_key).encode("utf-8")
    ).hexdigest()[:8]
    return "{0}-{1}".format(base, digest)


def assert_unique_metric_slugs(metric_specs):
    by_slug = {}

    for spec in metric_specs:
        slug = metric_slug(spec)
        by_slug.setdefault(slug, []).append(spec.label)

    collisions = {
        slug: labels
        for slug, labels in by_slug.items()
        if len(labels) > 1
    }

    if collisions:
        details = "; ".join(
            "{0}: {1}".format(
                slug,
                ", ".join(labels),
            )
            for slug, labels in sorted(collisions.items())
        )
        raise RuntimeError(
            "Selected metrics do not have unique output slugs: "
            "{0}".format(details)
        )


def number_token(value):
    return ("{0:g}".format(float(value))).replace(".", "p")


def load_patterns(path):
    with path.open() as fobj:
        nested = json.load(fobj)

    return {
        key: value
        for group in nested.values()
        for key, value in group.items()
    }


def first_glob(patterns):
    matches = []

    for pattern in patterns:
        matches.extend(
            sorted(
                pattern.parent.glob(pattern.name)
            )
        )

    unique = sorted(set(matches))

    return unique[0] if unique else None


def pattern_path(
    derivatives,
    rel_pattern,
    subject,
    session,
):
    rel_pattern = rel_pattern.replace(
        "_space-MNI152NLin2009cAsym_",
        "_space-{space}_",
    )

    return derivatives / rel_pattern.format(
        subject=subject,
        session=session,
        space=SPACE,
    )


def discover_subjects(derivatives):
    roots = (
        derivatives / "smriprep",
        derivatives
        / "qsirecon"
        / "derivatives"
        / "qsirecon-DIPYDKI",
        derivatives / "pymp2rage",
        derivatives / "ihmt",
    )

    subjects = set()

    for root in roots:
        if not root.is_dir():
            continue

        for path in root.glob("sub-*"):
            if (
                path.is_dir()
                and not is_pilot_subject(path.name)
            ):
                subjects.add(path.name)

    return sorted(subjects)


def find_dseg(
    derivatives,
    subject,
    session,
):
    return first_glob(
        (
            derivatives
            / "smriprep"
            / subject
            / "anat"
            / (
                "{0}_acq-MPRAGE_rec-refaced_run-01_"
                "space-{1}_dseg.nii*"
            ).format(
                subject,
                SPACE,
            ),
            derivatives
            / "smriprep"
            / subject
            / session
            / "anat"
            / (
                "{0}_{1}_acq-MPRAGE_rec-refaced_run-01_"
                "space-{2}_dseg.nii*"
            ).format(
                subject,
                session,
                SPACE,
            ),
            derivatives
            / "smriprep"
            / subject
            / "anat"
            / "{0}_*space-{1}_dseg.nii*".format(
                subject,
                SPACE,
            ),
            derivatives
            / "smriprep"
            / subject
            / session
            / "anat"
            / "{0}_{1}_*space-{2}_dseg.nii*".format(
                subject,
                session,
                SPACE,
            ),
        )
    )


def load_qc_table(path):
    if path is None:
        return None

    qc = pd.read_csv(
        path,
        sep="\t",
    )

    qc["participant_id"] = qc[
        "participant_id"
    ].map(subject_for_qc)

    qc = qc.loc[
        ~qc["participant_id"].map(is_pilot_subject)
    ].copy()

    return qc.set_index(
        "participant_id",
        drop=False,
    )


def qc_passes(
    qc,
    subject,
    session,
    metric_spec,
):
    if qc is None:
        return True

    if not metric_spec.qc_modalities:
        warnings.warn(
            "No QC modality mapping for {0}; "
            "applying no modality QC".format(
                metric_spec.label
            )
        )
        return True

    subject_id = subject_for_qc(subject)

    if subject_id not in qc.index:
        return False

    row = qc.loc[subject_id]
    prefix = session_label(session)

    for modality in metric_spec.qc_modalities:
        column = "{0}--{1}".format(
            prefix,
            modality,
        )

        if column not in qc.columns:
            raise RuntimeError(
                "QC file is missing required column: "
                "{0}".format(column)
            )

        value = row[column]

        if pd.isna(value) or int(value) != 1:
            return False

    return True


def selected_analysis_sets(analysis_set):
    if analysis_set == "both":
        return [
            "primary",
            "full",
        ]

    return [analysis_set]


def specs_for_analysis_set(
    specs,
    analysis_set,
    tissues=None,
):
    by_label = {
        spec.label: spec
        for spec in specs
    }

    ordered_specs = OrderedDict()
    tissue_values = (
        tuple(tissues)
        if tissues is not None
        else (None,)
    )

    for tissue in tissue_values:
        ordered_labels = metric_order(
            specs,
            analysis_set,
            tissue=tissue,
        )

        for label in ordered_labels:
            if label in by_label:
                ordered_specs.setdefault(
                    label,
                    by_label[label],
                )

    return list(
        ordered_specs.values()
    )


def select_metrics(
    specs,
    analysis_set,
    requested,
    tissues=None,
):
    analysis_sets = selected_analysis_sets(
        analysis_set
    )

    selected = OrderedDict()

    for current_set in analysis_sets:
        for spec in specs_for_analysis_set(
            specs,
            current_set,
            tissues=tissues,
        ):
            selected.setdefault(
                spec.label,
                spec,
            )

    if not requested:
        return list(selected.values())

    by_lower = {}
    for spec in selected.values():
        for label in (
            spec.label,
            spec.primary_label,
            spec.pattern_key,
        ):
            by_lower.setdefault(
                str(label).lower(),
                [],
            ).append(spec)

    requested_specs = []
    unknown = []

    for value in requested:
        matches = by_lower.get(
            str(value).strip().lower()
        )

        if not matches:
            unknown.append(str(value))
            continue

        for spec in matches:
            if spec not in requested_specs:
                requested_specs.append(spec)

    if unknown:
        raise ValueError(
            "Unknown metric label(s): {0}. "
            "Available labels: {1}".format(
                ", ".join(unknown),
                ", ".join(
                    spec.label
                    for spec in selected.values()
                ),
            )
        )

    return requested_specs


def collect_dseg_pairs(
    derivatives,
    subjects,
    session_a,
    session_b,
):
    pairs = []

    for subject in subjects:
        dseg_a = find_dseg(
            derivatives,
            subject,
            session_a,
        )
        dseg_b = find_dseg(
            derivatives,
            subject,
            session_b,
        )

        if dseg_a is None or dseg_b is None:
            continue

        pairs.append(
            DsegPairInputs(
                subject=subject,
                dseg_a=dseg_a,
                dseg_b=dseg_b,
            )
        )

    return pairs


def collect_subject_pairs(
    derivatives,
    patterns,
    qc,
    subjects,
    session_a,
    session_b,
    metric_spec,
):
    rel_pattern = patterns.get(
        metric_spec.pattern_key
    )

    if rel_pattern is None:
        raise RuntimeError(
            "patterns.json has no entry for {0}: "
            "{1}".format(
                metric_spec.label,
                metric_spec.pattern_key,
            )
        )

    pairs = []
    diagnostics = []

    for subject in subjects:
        record = {
            "metric": metric_spec.label,
            "subject": subject,
            "session_a": session_a,
            "session_b": session_b,
            "included": False,
            "reason": "",
            "metric_a": "",
            "metric_b": "",
        }

        if not qc_passes(
            qc,
            subject,
            session_a,
            metric_spec,
        ):
            record["reason"] = (
                "failed_or_missing_qc_session_a"
            )
            diagnostics.append(record)
            continue

        if not qc_passes(
            qc,
            subject,
            session_b,
            metric_spec,
        ):
            record["reason"] = (
                "failed_or_missing_qc_session_b"
            )
            diagnostics.append(record)
            continue

        metric_a = first_glob(
            (
                pattern_path(
                    derivatives,
                    rel_pattern,
                    subject,
                    session_a,
                ),
            )
        )

        metric_b = first_glob(
            (
                pattern_path(
                    derivatives,
                    rel_pattern,
                    subject,
                    session_b,
                ),
            )
        )

        record["metric_a"] = (
            str(metric_a)
            if metric_a is not None
            else ""
        )
        record["metric_b"] = (
            str(metric_b)
            if metric_b is not None
            else ""
        )

        missing = []

        if metric_a is None:
            missing.append("metric_a")

        if metric_b is None:
            missing.append("metric_b")

        if missing:
            record["reason"] = (
                "missing_" + "_".join(missing)
            )
            diagnostics.append(record)
            continue

        record["included"] = True
        record["reason"] = "included"

        diagnostics.append(record)

        pairs.append(
            SubjectPairInputs(
                subject=subject,
                metric_a=metric_a,
                metric_b=metric_b,
            )
        )

    return pairs, diagnostics


def pair_gmwm_hybrid_subject_pairs(
    wm_pairs,
    gm_pairs,
):
    gm_by_subject = {
        pair.subject: pair
        for pair in gm_pairs
    }
    paired = []

    for pair in wm_pairs:
        gm_pair = gm_by_subject.get(
            pair.subject
        )
        if gm_pair is None:
            continue
        paired.append(
            HybridSubjectPairInputs(
                subject=pair.subject,
                metric_a=pair.metric_a,
                metric_b=pair.metric_b,
                gm_metric_a=gm_pair.metric_a,
                gm_metric_b=gm_pair.metric_b,
            )
        )

    return paired


def load_like(
    path,
    reference,
    order,
):
    image = nib.load(str(path))

    if (
        image.shape[:3] != reference.shape[:3]
        or not np.allclose(
            image.affine,
            reference.affine,
            atol=1e-4,
        )
    ):
        image = resample_from_to(
            image,
            reference,
            order=order,
        )

    return np.asarray(
        image.get_fdata(),
        dtype=np.float32,
    )


def write_nifti(
    flat_values,
    reference,
    out_file,
    dtype,
    description,
):
    values = np.asarray(
        flat_values
    ).reshape(
        reference.shape[:3]
    ).astype(
        dtype,
        copy=False,
    )

    header = reference.header.copy()
    header.set_data_dtype(dtype)

    try:
        header["descrip"] = str(
            description
        )[:79]
    except Exception:
        pass

    image = nib.Nifti1Image(
        values,
        reference.affine,
        header,
    )

    nib.save(
        image,
        str(out_file),
    )


def erode_mask_mm(
    mask,
    reference,
    erosion_mm,
):
    mask_3d = np.asarray(
        mask,
        dtype=bool,
    ).reshape(
        reference.shape[:3]
    )

    if erosion_mm <= 0:
        return mask_3d.reshape(-1)

    voxel_sizes = tuple(
        float(value)
        for value in nib.affines.voxel_sizes(
            reference.affine
        )
    )

    distance = distance_transform_edt(
        mask_3d,
        sampling=voxel_sizes,
    )

    # Keep voxel centers more than the requested physical distance from
    # the mask exterior.
    return (
        distance > float(erosion_mm)
    ).reshape(-1)


def build_fixed_tissue_masks(
    gm_probseg,
    wm_probseg,
    gm_threshold,
    wm_threshold,
    gm_erosion_mm,
    wm_erosion_mm,
):
    reference = nib.load(
        str(gm_probseg)
    )

    gm_probability = load_like(
        gm_probseg,
        reference,
        order=1,
    ).reshape(-1)

    wm_probability = load_like(
        wm_probseg,
        reference,
        order=1,
    ).reshape(-1)

    gm_thresholded = (
        gm_probability
        >= float(gm_threshold)
    )
    wm_thresholded = (
        wm_probability
        >= float(wm_threshold)
    )

    gm_eroded = erode_mask_mm(
        gm_thresholded,
        reference,
        gm_erosion_mm,
    )
    wm_eroded = erode_mask_mm(
        wm_thresholded,
        reference,
        wm_erosion_mm,
    )

    overlap = gm_eroded & wm_eroded

    if np.any(overlap):
        overlap_indices = np.flatnonzero(overlap)

        gm_wins = (
            gm_probability[overlap]
            >= wm_probability[overlap]
        )

        gm_eroded[
            overlap_indices[~gm_wins]
        ] = False

        wm_eroded[
            overlap_indices[gm_wins]
        ] = False

    analysis_mask = (
        gm_eroded
        | wm_eroded
    )

    if not np.any(gm_eroded):
        raise RuntimeError(
            "The template GM mask is empty "
            "after thresholding/erosion"
        )

    if not np.any(wm_eroded):
        raise RuntimeError(
            "The template WM mask is empty "
            "after thresholding/erosion"
        )

    return {
        "gm_probability": gm_probability,
        "wm_probability": wm_probability,
        "gm_thresholded": gm_thresholded,
        "wm_thresholded": wm_thresholded,
        "gm": gm_eroded,
        "wm": wm_eroded,
        "gmwm": analysis_mask,
        "reference": reference,
    }


def write_fixed_masks(
    masks,
    reference,
    output_dir,
    gm_threshold,
    wm_threshold,
    gm_erosion_mm,
    wm_erosion_mm,
):
    gm_threshold_token = number_token(
        gm_threshold
    )
    wm_threshold_token = number_token(
        wm_threshold
    )
    gm_erosion_token = number_token(
        gm_erosion_mm
    )
    wm_erosion_token = number_token(
        wm_erosion_mm
    )
    base = "space-{0}".format(SPACE)

    write_nifti(
        masks["gm_probability"],
        reference,
        output_dir
        / "{0}_label-GM_probseg.nii.gz".format(
            base
        ),
        np.float32,
        "Template GM probability",
    )

    write_nifti(
        masks["wm_probability"],
        reference,
        output_dir
        / "{0}_label-WM_probseg.nii.gz".format(
            base
        ),
        np.float32,
        "Template WM probability",
    )

    write_nifti(
        masks["gm_thresholded"].astype(
            np.uint8
        ),
        reference,
        output_dir
        / (
            "{0}_label-GM_desc-templateProb{1}_"
            "mask.nii.gz"
        ).format(
            base,
            gm_threshold_token,
        ),
        np.uint8,
        "Template GM probability mask before erosion",
    )

    write_nifti(
        masks["wm_thresholded"].astype(
            np.uint8
        ),
        reference,
        output_dir
        / (
            "{0}_label-WM_desc-templateProb{1}_"
            "mask.nii.gz"
        ).format(
            base,
            wm_threshold_token,
        ),
        np.uint8,
        "Template WM probability mask before erosion",
    )

    write_nifti(
        masks["gm"].astype(np.uint8),
        reference,
        output_dir
        / (
            "{0}_label-GM_desc-templateProb{1}"
            "Eroded{2}mm_mask.nii.gz"
        ).format(
            base,
            gm_threshold_token,
            gm_erosion_token,
        ),
        np.uint8,
        "Fixed template GM analysis and summary mask",
    )

    write_nifti(
        masks["wm"].astype(np.uint8),
        reference,
        output_dir
        / (
            "{0}_label-WM_desc-templateProb{1}"
            "Eroded{2}mm_mask.nii.gz"
        ).format(
            base,
            wm_threshold_token,
            wm_erosion_token,
        ),
        np.uint8,
        "Fixed template WM analysis and summary mask",
    )

    write_nifti(
        masks["gmwm"].astype(np.uint8),
        reference,
        output_dir
        / (
            "{0}_desc-GMprob{1}WMprob{2}"
            "ErodedGM{3}mmWM{4}mm_mask.nii.gz"
        ).format(
            base,
            gm_threshold_token,
            wm_threshold_token,
            gm_erosion_token,
            wm_erosion_token,
        ),
        np.uint8,
        "Union of fixed template GM and WM compartments",
    )


def build_metric_memmap(
    pairs,
    reference,
    work_dir,
    metric_label,
    gm_mask=None,
):
    n_subjects = len(pairs)
    n_voxels = int(
        np.prod(reference.shape[:3])
    )

    prefix = safe_label(metric_label)

    values_path = (
        work_dir
        / "{0}_values.float32.dat".format(
            prefix
        )
    )

    values = np.memmap(
        str(values_path),
        mode="w+",
        dtype="float32",
        shape=(
            n_subjects,
            2,
            n_voxels,
        ),
    )

    values[:] = np.nan
    gm_mask = (
        np.asarray(gm_mask, dtype=bool)
        if gm_mask is not None
        else None
    )

    for subject_index, pair in enumerate(pairs):
        print(
            "  Loading {0} ({1}/{2})".format(
                pair.subject,
                subject_index + 1,
                n_subjects,
            ),
            flush=True,
        )

        session_a_values = load_like(
            pair.metric_a,
            reference,
            order=1,
        ).reshape(-1)

        session_b_values = load_like(
            pair.metric_b,
            reference,
            order=1,
        ).reshape(-1)

        if (
            gm_mask is not None
            and hasattr(pair, "gm_metric_a")
            and hasattr(pair, "gm_metric_b")
        ):
            gm_a_values = load_like(
                pair.gm_metric_a,
                reference,
                order=1,
            ).reshape(-1)
            gm_b_values = load_like(
                pair.gm_metric_b,
                reference,
                order=1,
            ).reshape(-1)
            session_a_values = session_a_values.copy()
            session_b_values = session_b_values.copy()
            session_a_values[gm_mask] = gm_a_values[
                gm_mask
            ]
            session_b_values[gm_mask] = gm_b_values[
                gm_mask
            ]

        values[
            subject_index,
            0,
            :,
        ] = session_a_values

        values[
            subject_index,
            1,
            :,
        ] = session_b_values

    values.flush()

    return values, values_path


def paired_outlier_masks(
    x1,
    x2,
    valid,
    z_threshold,
):
    """Detect paired subject outliers independently at each voxel.

    For each voxel and subject, calculate:

      pair mean = (session A + session B) / 2
      pair difference = session B - session A

    Each quantity is standardized across subjects at that voxel using a
    modified median/MAD z-score. A complete subject pair is flagged when
    either absolute robust z-score exceeds the specified threshold.

    If the MAD is zero or nonfinite for a criterion at a voxel, no subject
    is flagged by that criterion at that voxel.
    """

    pair_mean = np.where(
        valid,
        (x1 + x2) / 2.0,
        np.nan,
    )

    pair_diff = np.where(
        valid,
        x2 - x1,
        np.nan,
    )

    with warnings.catch_warnings():
        warnings.simplefilter(
            "ignore",
            category=RuntimeWarning,
        )

        median_mean = np.nanmedian(
            pair_mean,
            axis=0,
        )
        median_diff = np.nanmedian(
            pair_diff,
            axis=0,
        )

        mad_mean = np.nanmedian(
            np.abs(
                pair_mean
                - median_mean[None, :]
            ),
            axis=0,
        )

        mad_diff = np.nanmedian(
            np.abs(
                pair_diff
                - median_diff[None, :]
            ),
            axis=0,
        )

    usable_mean = (
        np.isfinite(mad_mean)
        & (mad_mean > 0)
    )
    usable_diff = (
        np.isfinite(mad_diff)
        & (mad_diff > 0)
    )

    outlier_mean = np.zeros(
        valid.shape,
        dtype=bool,
    )
    outlier_diff = np.zeros(
        valid.shape,
        dtype=bool,
    )

    if np.any(usable_mean):
        robust_z_mean = np.zeros(
            pair_mean.shape,
            dtype=np.float32,
        )

        robust_z_mean[
            :,
            usable_mean,
        ] = (
            ROBUST_NORMAL_SCALE
            * (
                pair_mean[:, usable_mean]
                - median_mean[
                    None,
                    usable_mean,
                ]
            )
            / mad_mean[
                None,
                usable_mean,
            ]
        )

        outlier_mean = (
            valid
            & usable_mean[None, :]
            & (
                np.abs(robust_z_mean)
                > z_threshold
            )
        )

    if np.any(usable_diff):
        robust_z_diff = np.zeros(
            pair_diff.shape,
            dtype=np.float32,
        )

        robust_z_diff[
            :,
            usable_diff,
        ] = (
            ROBUST_NORMAL_SCALE
            * (
                pair_diff[:, usable_diff]
                - median_diff[
                    None,
                    usable_diff,
                ]
            )
            / mad_diff[
                None,
                usable_diff,
            ]
        )

        outlier_diff = (
            valid
            & usable_diff[None, :]
            & (
                np.abs(robust_z_diff)
                > z_threshold
            )
        )

    return (
        outlier_mean,
        outlier_diff,
    )


def icc2_1_from_pairs(
    x1,
    x2,
    valid,
    min_subjects,
):
    """Calculate voxelwise ICC(2,1) from complete subject pairs."""

    valid = np.asarray(
        valid,
        dtype=bool,
    )

    n = np.sum(
        valid,
        axis=0,
    ).astype(np.int32)

    n_float = n.astype(np.float64)

    safe_n = np.where(
        n_float > 0,
        n_float,
        np.nan,
    )

    x1d = np.asarray(
        x1,
        dtype=np.float64,
    )
    x2d = np.asarray(
        x2,
        dtype=np.float64,
    )

    x1_zero = np.where(
        valid,
        x1d,
        0.0,
    )
    x2_zero = np.where(
        valid,
        x2d,
        0.0,
    )

    session_mean_1 = (
        np.sum(
            x1_zero,
            axis=0,
        )
        / safe_n
    )
    session_mean_2 = (
        np.sum(
            x2_zero,
            axis=0,
        )
        / safe_n
    )

    grand_mean = (
        session_mean_1
        + session_mean_2
    ) / 2.0

    subject_mean = (
        x1d
        + x2d
    ) / 2.0

    ss_subject = (
        2.0
        * np.sum(
            np.where(
                valid,
                (
                    subject_mean
                    - grand_mean[None, :]
                )
                ** 2,
                0.0,
            ),
            axis=0,
        )
    )

    ms_subject = (
        ss_subject
        / np.where(
            n_float > 1,
            n_float - 1.0,
            np.nan,
        )
    )

    ss_session = (
        n_float
        * (
            (
                session_mean_1
                - grand_mean
            )
            ** 2
            + (
                session_mean_2
                - grand_mean
            )
            ** 2
        )
    )

    # There are two sessions, so df_session = 1.
    ms_session = ss_session

    residual_1 = (
        x1d
        - subject_mean
        - session_mean_1[None, :]
        + grand_mean[None, :]
    )

    residual_2 = (
        x2d
        - subject_mean
        - session_mean_2[None, :]
        + grand_mean[None, :]
    )

    ss_error = np.sum(
        np.where(
            valid,
            residual_1 ** 2
            + residual_2 ** 2,
            0.0,
        ),
        axis=0,
    )

    # For two sessions, df_error = n - 1.
    ms_error = (
        ss_error
        / np.where(
            n_float > 1,
            n_float - 1.0,
            np.nan,
        )
    )

    denominator = (
        ms_subject
        + ms_error
        + 2.0
        * (
            ms_session
            - ms_error
        )
        / safe_n
    )

    icc = (
        ms_subject
        - ms_error
    ) / denominator

    mean_difference = (
        np.sum(
            np.where(
                valid,
                x2d - x1d,
                0.0,
            ),
            axis=0,
        )
        / safe_n
    )

    rmse = np.sqrt(
        np.sum(
            np.where(
                valid,
                (
                    x2d
                    - x1d
                )
                ** 2,
                0.0,
            ),
            axis=0,
        )
        / safe_n
    )

    invalid = (
        (n < int(min_subjects))
        | ~np.isfinite(denominator)
        | (
            np.abs(denominator)
            <= np.finfo(np.float64).eps
        )
    )

    icc[invalid] = np.nan
    mean_difference[n == 0] = np.nan
    rmse[n == 0] = np.nan

    return (
        icc.astype(np.float32),
        n,
        mean_difference.astype(np.float32),
        rmse.astype(np.float32),
    )


def empty_result(n_voxels):
    return {
        "primary_icc": np.full(
            n_voxels,
            np.nan,
            dtype=np.float32,
        ),
        "primary_n": np.zeros(
            n_voxels,
            dtype=np.int16,
        ),
        "mean_difference": np.full(
            n_voxels,
            np.nan,
            dtype=np.float32,
        ),
        "rmse": np.full(
            n_voxels,
            np.nan,
            dtype=np.float32,
        ),
        "sensitivity_icc": np.full(
            n_voxels,
            np.nan,
            dtype=np.float32,
        ),
        "sensitivity_n": np.zeros(
            n_voxels,
            dtype=np.int16,
        ),
        "outlier_mean_n": np.zeros(
            n_voxels,
            dtype=np.int16,
        ),
        "outlier_diff_n": np.zeros(
            n_voxels,
            dtype=np.int16,
        ),
        "outlier_any_n": np.zeros(
            n_voxels,
            dtype=np.int16,
        ),
    }


def process_compartment(
    values,
    compartment_mask,
    compartment_name,
    min_subjects,
    outlier_z,
    min_retained_fraction,
    remove_zeros,
    chunk_size,
    do_outlier_sensitivity,
):
    """Compute voxelwise maps inside one fixed tissue compartment."""

    n_voxels = values.shape[2]

    result = empty_result(
        n_voxels
    )

    compartment_mask = np.asarray(
        compartment_mask,
        dtype=bool,
    )

    for start in range(
        0,
        n_voxels,
        int(chunk_size),
    ):
        stop = min(
            start + int(chunk_size),
            n_voxels,
        )

        chunk_mask = compartment_mask[
            start:stop
        ]

        if not np.any(chunk_mask):
            continue

        print(
            "    {0}: voxels {1:,}-{2:,} / "
            "{3:,}".format(
                compartment_name,
                start + 1,
                stop,
                n_voxels,
            ),
            flush=True,
        )

        x1 = np.asarray(
            values[
                :,
                0,
                start:stop,
            ],
            dtype=np.float32,
        )
        x2 = np.asarray(
            values[
                :,
                1,
                start:stop,
            ],
            dtype=np.float32,
        )

        valid = (
            chunk_mask[None, :]
            & np.isfinite(x1)
            & np.isfinite(x2)
        )

        if remove_zeros:
            valid &= (
                (x1 != 0)
                & (x2 != 0)
            )

        (
            icc,
            n,
            difference_map,
            rmse_map,
        ) = icc2_1_from_pairs(
            x1,
            x2,
            valid,
            min_subjects=min_subjects,
        )

        result["primary_icc"][
            start:stop
        ] = icc

        result["primary_n"][
            start:stop
        ] = np.minimum(
            n,
            np.iinfo(np.int16).max,
        ).astype(np.int16)

        result["mean_difference"][
            start:stop
        ] = difference_map

        result["rmse"][
            start:stop
        ] = rmse_map

        if not do_outlier_sensitivity:
            continue

        (
            outlier_mean,
            outlier_diff,
        ) = paired_outlier_masks(
            x1,
            x2,
            valid,
            z_threshold=outlier_z,
        )

        outlier_any = (
            outlier_mean
            | outlier_diff
        )

        sensitivity_valid = (
            valid
            & ~outlier_any
        )

        (
            sensitivity_icc,
            sensitivity_n,
            _,
            _,
        ) = icc2_1_from_pairs(
            x1,
            x2,
            sensitivity_valid,
            min_subjects=min_subjects,
        )

        retained_fraction = (
            sensitivity_n.astype(np.float64)
            / np.where(
                n > 0,
                n,
                np.nan,
            )
        )

        sensitivity_icc[
            retained_fraction
            < min_retained_fraction
        ] = np.nan

        result["sensitivity_icc"][
            start:stop
        ] = sensitivity_icc

        result["sensitivity_n"][
            start:stop
        ] = np.minimum(
            sensitivity_n,
            np.iinfo(np.int16).max,
        ).astype(np.int16)

        result["outlier_mean_n"][
            start:stop
        ] = np.minimum(
            np.sum(
                outlier_mean,
                axis=0,
            ),
            np.iinfo(np.int16).max,
        ).astype(np.int16)

        result["outlier_diff_n"][
            start:stop
        ] = np.minimum(
            np.sum(
                outlier_diff,
                axis=0,
            ),
            np.iinfo(np.int16).max,
        ).astype(np.int16)

        result["outlier_any_n"][
            start:stop
        ] = np.minimum(
            np.sum(
                outlier_any,
                axis=0,
            ),
            np.iinfo(np.int16).max,
        ).astype(np.int16)

    return result


def combine_compartment_results(
    gm_result,
    wm_result,
    gm_mask,
    wm_mask,
):
    combined = empty_result(
        len(gm_mask)
    )

    for key in combined:
        combined[key][gm_mask] = (
            gm_result[key][gm_mask]
        )
        combined[key][wm_mask] = (
            wm_result[key][wm_mask]
        )

    return combined


def restrict_result_to_metric_tissues(
    result,
    metric_spec,
    masks,
    hybrid=False,
):
    """Blank map values outside the metric's valid tissue contexts."""

    invalid = np.zeros(
        len(masks["gm"]),
        dtype=bool,
    )

    if (
        "gm" not in metric_spec.tissues
        and not (
            hybrid
            and "gmwm" in metric_spec.tissues
        )
    ):
        invalid |= masks["gm"]

    if (
        "wm" not in metric_spec.tissues
        and not (
            hybrid
            and "gmwm" in metric_spec.tissues
        )
    ):
        invalid |= masks["wm"]

    if not np.any(invalid):
        return result

    defaults = empty_result(
        len(invalid)
    )

    for key in result:
        result[key][invalid] = defaults[key][
            invalid
        ]

    return result


def summarize_icc_map(
    metric_spec,
    metric_label,
    analysis_set,
    tissue,
    analysis,
    icc,
    n_map,
    summary_mask,
    n_subject_pairs,
):
    summary_mask = np.asarray(
        summary_mask,
        dtype=bool,
    )

    finite = (
        summary_mask
        & np.isfinite(icc)
    )

    values = icc[
        finite
    ].astype(np.float64)

    n_values = n_map[
        finite
    ].astype(np.float64)

    n_mask_voxels = int(
        np.count_nonzero(summary_mask)
    )

    row = {
        "metric": metric_label,
        "metric_key": metric_spec.label,
        "pattern_key": metric_spec.pattern_key,
        "analysis_set": analysis_set,
        "family": metric_spec.family,
        "source_image": metric_spec.source_image,
        "tissue": tissue,
        "analysis": analysis,
        "n_subject_pairs_available": int(
            n_subject_pairs
        ),
        "n_voxels_in_summary_mask": (
            n_mask_voxels
        ),
        "n_voxels_with_icc": int(
            values.size
        ),
        "proportion_mask_with_icc": (
            float(values.size)
            / float(n_mask_voxels)
            if n_mask_voxels
            else np.nan
        ),
        "mean_icc": np.nan,
        "median_icc": np.nan,
        "q25_icc": np.nan,
        "q75_icc": np.nan,
        "proportion_icc_below_0": np.nan,
        "proportion_icc_ge_0p50": np.nan,
        "proportion_icc_ge_0p75": np.nan,
        "proportion_icc_ge_0p90": np.nan,
        "median_n_subjects_per_voxel": np.nan,
        "mean_n_subjects_per_voxel": np.nan,
        "minimum_n_subjects_per_voxel": np.nan,
    }

    if values.size:
        row.update(
            {
                "mean_icc": float(
                    np.mean(values)
                ),
                "median_icc": float(
                    np.median(values)
                ),
                "q25_icc": float(
                    np.percentile(
                        values,
                        25,
                    )
                ),
                "q75_icc": float(
                    np.percentile(
                        values,
                        75,
                    )
                ),
                "proportion_icc_below_0": float(
                    np.mean(values < 0)
                ),
                "proportion_icc_ge_0p50": float(
                    np.mean(values >= 0.50)
                ),
                "proportion_icc_ge_0p75": float(
                    np.mean(values >= 0.75)
                ),
                "proportion_icc_ge_0p90": float(
                    np.mean(values >= 0.90)
                ),
                "median_n_subjects_per_voxel": float(
                    np.median(n_values)
                ),
                "mean_n_subjects_per_voxel": float(
                    np.mean(n_values)
                ),
                "minimum_n_subjects_per_voxel": int(
                    np.min(n_values)
                ),
            }
        )

    return row


def plot_ranked_summaries(
    summary,
    output_dir,
):
    if summary.empty:
        return

    finite_summary = summary[
        np.isfinite(
            pd.to_numeric(
                summary["median_icc"],
                errors="coerce",
            )
        )
    ].copy()

    for (
        analysis_set,
        analysis,
        tissue,
    ), group in finite_summary.groupby(
        [
            "analysis_set",
            "analysis",
            "tissue",
        ],
        sort=True,
    ):
        data = group.sort_values(
            "median_icc",
            ascending=True,
        ).reset_index(drop=True)

        if data.empty:
            continue

        height = max(
            6.0,
            0.42 * len(data) + 1.8,
        )

        fig, ax = plt.subplots(
            figsize=(
                10.5,
                height,
            )
        )

        y = np.arange(len(data))

        medians = data[
            "median_icc"
        ].to_numpy(dtype=float)

        q25 = data[
            "q25_icc"
        ].to_numpy(dtype=float)

        q75 = data[
            "q75_icc"
        ].to_numpy(dtype=float)

        colors = [
            SOURCE_IMAGE_COLORS.get(
                source,
                SOURCE_IMAGE_COLORS["Other"],
            )
            for source in data["source_image"]
        ]

        for index in range(len(data)):
            ax.hlines(
                y[index],
                q25[index],
                q75[index],
                color=colors[index],
                linewidth=4,
            )

            ax.scatter(
                medians[index],
                y[index],
                s=55,
                color=colors[index],
                edgecolor="black",
                linewidth=0.45,
                zorder=3,
            )

            ax.text(
                min(
                    q75[index] + 0.025,
                    1.01,
                ),
                y[index],
                "{0:.2f} [{1:.2f}, "
                "{2:.2f}]".format(
                    medians[index],
                    q25[index],
                    q75[index],
                ),
                va="center",
                ha="left",
                fontsize=7.5,
            )

        for value in (
            0.0,
            0.50,
            0.75,
            0.90,
        ):
            ax.axvline(
                value,
                color="#888888",
                linestyle=(
                    "-"
                    if value == 0
                    else ":"
                ),
                linewidth=0.8,
            )

        ax.set_yticks(y)
        ax.set_yticklabels(
            data["metric"],
            fontsize=8,
        )

        ax.set_xlim(
            -1.0,
            1.15,
        )

        ax.set_xlabel(
            "Voxelwise ICC(2,1): median [IQR]"
        )
        ax.set_ylabel("")

        ax.set_title(
            "{0} — {1} — {2}".format(
                TISSUE_LABELS.get(
                    tissue,
                    tissue.upper(),
                ),
                analysis_set.title(),
                (
                    "Primary"
                    if analysis == "primary"
                    else analysis
                ),
            )
        )

        ax.grid(False)

        ax.spines[
            "top"
        ].set_visible(False)

        ax.spines[
            "right"
        ].set_visible(False)

        observed_sources = set(
            data["source_image"]
        )

        handles = [
            Line2D(
                [0],
                [0],
                marker="o",
                linestyle="none",
                markersize=7,
                markerfacecolor=color,
                markeredgecolor="black",
                markeredgewidth=0.45,
                label=source,
            )
            for (
                source,
                color,
            ) in SOURCE_IMAGE_COLORS.items()
            if source in observed_sources
        ]

        ax.legend(
            handles=handles,
            title="Source image",
            loc="center left",
            bbox_to_anchor=(
                1.02,
                0.5,
            ),
            frameon=False,
        )

        fig.tight_layout()

        stem = (
            output_dir
            / "voxelwise_icc_ranked_{0}_{1}_{2}".format(
                analysis_set,
                tissue,
                safe_label(analysis),
            )
        )

        for extension in (
            "png",
            "pdf",
        ):
            fig.savefig(
                str(stem)
                + "."
                + extension,
                dpi=300,
                bbox_inches="tight",
            )

        plt.close(fig)


def add_rank_column(summary):
    if summary.empty:
        return summary

    ranked = summary.copy()

    ranked[
        "rank_by_median_icc"
    ] = np.nan

    groups = ranked.groupby(
        [
            "analysis_set",
            "analysis",
            "tissue",
        ]
    ).groups

    for indices in groups.values():
        values = pd.to_numeric(
            ranked.loc[
                indices,
                "median_icc",
            ],
            errors="coerce",
        )

        ranked.loc[
            indices,
            "rank_by_median_icc",
        ] = values.rank(
            method="min",
            ascending=False,
            na_option="bottom",
        )

    ranked[
        "rank_by_median_icc"
    ] = ranked[
        "rank_by_median_icc"
    ].astype("Int64")

    return ranked


def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__
    )

    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(
            "/cbica/projects/nibs"
        ),
        help=(
            "Project root containing "
            "derivatives, code, and data."
        ),
    )

    parser.add_argument(
        "--derivatives-dir",
        type=Path,
        default=None,
    )

    parser.add_argument(
        "--patterns-file",
        type=Path,
        default=None,
    )

    parser.add_argument(
        "--qc-file",
        type=Path,
        default=None,
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
    )

    parser.add_argument(
        "--work-dir",
        type=Path,
        default=None,
    )

    parser.add_argument(
        "--subject-id",
        action="append",
        help=(
            "Subject(s), with or without sub-."
        ),
    )

    parser.add_argument(
        "--metric",
        action="append",
        help=(
            "Selected metric label to process. "
            "Repeat as needed. Default: all "
            "metrics in --analysis-set."
        ),
    )

    parser.add_argument(
        "--analysis-set",
        choices=ANALYSIS_SETS,
        default="both",
        help=(
            "Metric set to process and summarize. "
            "Use primary for the primary-analysis "
            "metrics, full for all metrics in "
            "patterns.json, or both. Default: both."
        ),
    )

    parser.add_argument(
        "--tissue",
        action="append",
        choices=SUMMARY_TISSUES,
        help=(
            "Compartment to include in summary "
            "tables and figures. Repeat as "
            "needed. The ICC map itself is "
            "always computed once over the "
            "union of the fixed eroded GM and "
            "WM masks. Default: gm, wm, gmwm."
        ),
    )

    parser.add_argument(
        "--session-a",
        default="ses-01",
    )

    parser.add_argument(
        "--session-b",
        default="ses-02",
    )

    parser.add_argument(
        "--gm-probseg",
        type=Path,
        default=None,
        help=(
            "Template-space GM probability map. "
            "Defaults to <project-root>/code/data/"
            "tpl-MNI152NLin2009cAsym_res-01_"
            "label-GM_probseg.nii.gz."
        ),
    )

    parser.add_argument(
        "--wm-probseg",
        type=Path,
        default=None,
        help=(
            "Template-space WM probability map. "
            "Defaults to <project-root>/code/data/"
            "tpl-MNI152NLin2009cAsym_res-01_"
            "label-WM_probseg.nii.gz."
        ),
    )

    parser.add_argument(
        "--gm-threshold",
        type=float,
        default=0.50,
        help="Template GM probability threshold. Default: 0.50.",
    )

    parser.add_argument(
        "--wm-threshold",
        type=float,
        default=0.50,
        help="Template WM probability threshold. Default: 0.50.",
    )

    parser.add_argument(
        "--gm-erosion-mm",
        type=float,
        default=0.0,
        help="Physical erosion distance for the GM mask. Default: 0 mm.",
    )

    parser.add_argument(
        "--wm-erosion-mm",
        type=float,
        default=0.0,
        help="Physical erosion distance for the WM mask. Default: 0 mm.",
    )

    parser.add_argument(
        "--min-subjects",
        type=int,
        default=15,
        help=(
            "Minimum complete subject pairs "
            "required for ICC at a voxel."
        ),
    )

    parser.add_argument(
        "--outlier-z",
        type=float,
        default=6.0,
        help=(
            "Absolute modified MAD z threshold "
            "for the paired sensitivity "
            "analysis."
        ),
    )

    parser.add_argument(
        "--min-retained-fraction",
        type=float,
        default=0.80,
        help=(
            "Minimum fraction of initially "
            "complete pairs retained after "
            "outlier filtering."
        ),
    )

    parser.add_argument(
        "--chunk-size",
        type=int,
        default=50000,
        help=(
            "Number of voxels processed per "
            "vectorized chunk."
        ),
    )

    parser.add_argument(
        "--allow-zero",
        action="store_true",
        help=(
            "Treat exact zero as a valid metric "
            "value. By default zeros are "
            "excluded."
        ),
    )

    parser.add_argument(
        "--no-outlier-sensitivity",
        action="store_true",
        help=(
            "Skip the paired-MAD sensitivity "
            "analysis."
        ),
    )

    parser.add_argument(
        "--no-qc",
        action="store_true",
        help=(
            "Do not apply manual modality QC "
            "even if the default QC file exists."
        ),
    )

    parser.add_argument(
        "--keep-work-files",
        action="store_true",
        help=(
            "Retain temporary metric "
            "memory-map files."
        ),
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help=(
            "Overwrite existing maps. Summary "
            "tables and figures are always "
            "regenerated."
        ),
    )

    args = parser.parse_args()

    args.project_root = (
        args.project_root
        .expanduser()
        .resolve()
    )

    args.derivatives_dir = (
        args.derivatives_dir
        .expanduser()
        .resolve()
        if args.derivatives_dir
        else (
            args.project_root
            / "derivatives"
        )
    )

    args.patterns_file = (
        args.patterns_file
        .expanduser()
        .resolve()
        if args.patterns_file
        else (
            args.project_root
            / "code"
            / "configuration"
            / "patterns.json"
        )
    )

    if not args.patterns_file.exists():
        fallback = (
            Path(__file__)
            .resolve()
            .parents[1]
            / "configuration"
            / "patterns.json"
        )

        args.patterns_file = fallback

    if (
        args.qc_file is None
        and not args.no_qc
    ):
        candidates = (
            args.project_root
            / "code"
            / "data"
            / "manual_qc_modality.tsv",
            Path(__file__)
            .resolve()
            .parents[1]
            / "data"
            / "manual_qc_modality.tsv",
        )

        args.qc_file = next(
            (
                path
                for path in candidates
                if path.exists()
            ),
            None,
        )

    elif args.qc_file is not None:
        args.qc_file = (
            args.qc_file
            .expanduser()
            .resolve()
        )

    args.output_dir = (
        args.output_dir
        .expanduser()
        .resolve()
        if args.output_dir
        else (
            args.project_root
            / "derivatives"
            / "mni_voxelwise_icc"
        )
    )

    args.work_dir = (
        args.work_dir
        .expanduser()
        .resolve()
        if args.work_dir
        else (
            args.output_dir
            / "work"
        )
    )

    args.session_a = normalize_session(
        args.session_a
    )
    args.session_b = normalize_session(
        args.session_b
    )

    args.summary_tissues = (
        args.tissue
        if args.tissue
        else list(SUMMARY_TISSUES)
    )

    if args.session_a == args.session_b:
        parser.error(
            "--session-a and --session-b "
            "must be different"
        )

    data_candidates = (
        args.project_root
        / "code"
        / "data",
        Path(__file__)
        .resolve()
        .parents[1]
        / "data",
    )

    if args.gm_probseg is None:
        args.gm_probseg = next(
            (
                directory
                / (
                    "tpl-{0}_res-01_label-GM_"
                    "probseg.nii.gz"
                ).format(SPACE)
                for directory in data_candidates
                if (
                    directory
                    / (
                        "tpl-{0}_res-01_label-GM_"
                        "probseg.nii.gz"
                    ).format(SPACE)
                ).exists()
            ),
            data_candidates[0]
            / (
                "tpl-{0}_res-01_label-GM_"
                "probseg.nii.gz"
            ).format(SPACE),
        )
    else:
        args.gm_probseg = (
            args.gm_probseg
            .expanduser()
            .resolve()
        )

    if args.wm_probseg is None:
        args.wm_probseg = next(
            (
                directory
                / (
                    "tpl-{0}_res-01_label-WM_"
                    "probseg.nii.gz"
                ).format(SPACE)
                for directory in data_candidates
                if (
                    directory
                    / (
                        "tpl-{0}_res-01_label-WM_"
                        "probseg.nii.gz"
                    ).format(SPACE)
                ).exists()
            ),
            data_candidates[0]
            / (
                "tpl-{0}_res-01_label-WM_"
                "probseg.nii.gz"
            ).format(SPACE),
        )
    else:
        args.wm_probseg = (
            args.wm_probseg
            .expanduser()
            .resolve()
        )

    if not args.gm_probseg.exists():
        raise FileNotFoundError(
            "GM probability map not found: "
            "{0}".format(args.gm_probseg)
        )

    if not args.wm_probseg.exists():
        raise FileNotFoundError(
            "WM probability map not found: "
            "{0}".format(args.wm_probseg)
        )

    if not (
        0.0
        < args.gm_threshold
        <= 1.0
    ):
        parser.error(
            "--gm-threshold must be in (0, 1]"
        )

    if not (
        0.0
        < args.wm_threshold
        <= 1.0
    ):
        parser.error(
            "--wm-threshold must be in (0, 1]"
        )

    if (
        args.gm_erosion_mm < 0
        or args.wm_erosion_mm < 0
    ):
        parser.error(
            "Template mask erosion distances "
            "must be nonnegative"
        )

    if args.min_subjects < 3:
        parser.error(
            "--min-subjects must be at least 3"
        )

    if args.outlier_z <= 0:
        parser.error(
            "--outlier-z must be positive"
        )

    if not (
        0
        < args.min_retained_fraction
        <= 1
    ):
        parser.error(
            "--min-retained-fraction must be "
            "in (0, 1]"
        )

    if args.chunk_size < 1:
        parser.error(
            "--chunk-size must be positive"
        )

    return args


def map_paths(
    output_dir,
    metric_spec,
    sensitivity_tag,
    hybrid=False,
):
    prefix = (
        "metric-{0}_space-{1}".format(
            metric_slug(metric_spec),
            SPACE,
        )
    )

    primary_desc = (
        "hybridPrimary"
        if hybrid
        else "primary"
    )

    paths = {
        "primary_icc": (
            output_dir
            / (
                "{0}_desc-{1}_"
                "stat-icc2p1.nii.gz"
            ).format(prefix, primary_desc)
        ),
        "primary_n": (
            output_dir
            / (
                "{0}_desc-{1}_"
                "stat-nsubjects.nii.gz"
            ).format(prefix, primary_desc)
        ),
        "mean_difference": (
            output_dir
            / (
                "{0}_desc-{1}_"
                "stat-meanDifference.nii.gz"
            ).format(prefix, primary_desc)
        ),
        "rmse": (
            output_dir
            / (
                "{0}_desc-{1}_"
                "stat-rmse.nii.gz"
            ).format(prefix, primary_desc)
        ),
    }

    if sensitivity_tag is not None:
        sensitivity_desc = (
            "hybrid{0}".format(
                sensitivity_tag[:1].upper()
                + sensitivity_tag[1:]
            )
            if hybrid
            else sensitivity_tag
        )
        paths.update(
            {
                "sensitivity_icc": (
                    output_dir
                    / (
                        "{0}_desc-{1}_"
                        "stat-icc2p1.nii.gz"
                    ).format(
                        prefix,
                        sensitivity_desc,
                    )
                ),
                "sensitivity_n": (
                    output_dir
                    / (
                        "{0}_desc-{1}_"
                        "stat-nsubjects.nii.gz"
                    ).format(
                        prefix,
                        sensitivity_desc,
                    )
                ),
                "outlier_mean_n": (
                    output_dir
                    / (
                        "{0}_desc-{1}_"
                        "stat-nOutlierPairMean."
                        "nii.gz"
                    ).format(
                        prefix,
                        sensitivity_desc,
                    )
                ),
                "outlier_diff_n": (
                    output_dir
                    / (
                        "{0}_desc-{1}_"
                        "stat-nOutlierDifference."
                        "nii.gz"
                    ).format(
                        prefix,
                        sensitivity_desc,
                    )
                ),
                "outlier_any_n": (
                    output_dir
                    / (
                        "{0}_desc-{1}_"
                        "stat-nOutlierAny.nii.gz"
                    ).format(
                        prefix,
                        sensitivity_desc,
                    )
                ),
            }
        )

    return paths


def load_existing_result(
    paths,
    reference,
    do_outlier_sensitivity,
):
    result = empty_result(
        int(
            np.prod(reference.shape[:3])
        )
    )

    required = [
        "primary_icc",
        "primary_n",
        "mean_difference",
        "rmse",
    ]

    if do_outlier_sensitivity:
        required.extend(
            [
                "sensitivity_icc",
                "sensitivity_n",
                "outlier_mean_n",
                "outlier_diff_n",
                "outlier_any_n",
            ]
        )

    if not all(
        paths[key].exists()
        for key in required
    ):
        return None

    for key in required:
        order = (
            0
            if (
                key.endswith("_n")
                or "outlier" in key
            )
            else 1
        )

        result[key] = load_like(
            paths[key],
            reference,
            order=order,
        ).reshape(-1)

        if order == 0:
            result[key] = np.rint(
                result[key]
            ).astype(np.int16)
        else:
            result[key] = result[
                key
            ].astype(np.float32)

    return result


def write_result_maps(
    result,
    paths,
    reference,
    do_outlier_sensitivity,
):
    write_nifti(
        result["primary_icc"],
        reference,
        paths["primary_icc"],
        np.float32,
        (
            "Voxelwise ICC(2,1), primary "
            "complete-pair analysis"
        ),
    )

    write_nifti(
        result["primary_n"],
        reference,
        paths["primary_n"],
        np.int16,
        (
            "Number of complete subject pairs "
            "used for primary ICC"
        ),
    )

    write_nifti(
        result["mean_difference"],
        reference,
        paths["mean_difference"],
        np.float32,
        (
            "Mean session B minus session A "
            "difference"
        ),
    )

    write_nifti(
        result["rmse"],
        reference,
        paths["rmse"],
        np.float32,
        (
            "Root mean squared test-retest "
            "difference"
        ),
    )

    if not do_outlier_sensitivity:
        return

    write_nifti(
        result["sensitivity_icc"],
        reference,
        paths["sensitivity_icc"],
        np.float32,
        (
            "Voxelwise ICC(2,1) after "
            "compartment-specific paired "
            "outlier filtering"
        ),
    )

    write_nifti(
        result["sensitivity_n"],
        reference,
        paths["sensitivity_n"],
        np.int16,
        (
            "Complete subject pairs retained "
            "after paired outlier filtering"
        ),
    )

    write_nifti(
        result["outlier_mean_n"],
        reference,
        paths["outlier_mean_n"],
        np.int16,
        (
            "Subject pairs flagged by "
            "across-session mean criterion"
        ),
    )

    write_nifti(
        result["outlier_diff_n"],
        reference,
        paths["outlier_diff_n"],
        np.int16,
        (
            "Subject pairs flagged by "
            "session-difference criterion"
        ),
    )

    write_nifti(
        result["outlier_any_n"],
        reference,
        paths["outlier_any_n"],
        np.int16,
        (
            "Subject pairs removed by either "
            "paired outlier criterion"
        ),
    )


def main():
    args = parse_args()

    require_dependencies()

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    args.work_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    patterns = load_patterns(
        args.patterns_file
    )

    all_metric_specs = build_metric_specs(
        args.patterns_file
    )
    spec_by_label = {
        spec.label: spec
        for spec in all_metric_specs
    }
    hybrid_pairs = gm_noddi_hybrid_pairs(
        all_metric_specs
    )

    qc = load_qc_table(
        args.qc_file
    )

    metric_specs = select_metrics(
        all_metric_specs,
        args.analysis_set,
        args.metric,
        tissues=args.summary_tissues,
    )
    assert_unique_metric_slugs(
        metric_specs
    )

    analysis_sets = selected_analysis_sets(
        args.analysis_set
    )

    display_labels_by_tissue = {
        tissue: {
            analysis_set: metric_display_labels(
                all_metric_specs,
                analysis_set,
                tissue=tissue,
            )
            for analysis_set in analysis_sets
        }
        for tissue in args.summary_tissues
    }

    labels_by_tissue = {
        tissue: {
            analysis_set: set(
                metric_order(
                    all_metric_specs,
                    analysis_set,
                    tissue=tissue,
                )
            )
            for analysis_set in analysis_sets
        }
        for tissue in args.summary_tissues
    }

    subjects = (
        [
            normalize_subject(value)
            for value in args.subject_id
        ]
        if args.subject_id
        else discover_subjects(
            args.derivatives_dir
        )
    )

    print(
        "Building fixed tissue masks from "
        "template probability maps: GM >= {0}, "
        "WM >= {1}".format(
            args.gm_threshold,
            args.wm_threshold,
        ),
        flush=True,
    )

    masks = build_fixed_tissue_masks(
        args.gm_probseg,
        args.wm_probseg,
        args.gm_threshold,
        args.wm_threshold,
        args.gm_erosion_mm,
        args.wm_erosion_mm,
    )
    reference = masks["reference"]

    write_fixed_masks(
        masks,
        reference,
        args.output_dir,
        gm_threshold=args.gm_threshold,
        wm_threshold=args.wm_threshold,
        gm_erosion_mm=args.gm_erosion_mm,
        wm_erosion_mm=args.wm_erosion_mm,
    )

    all_diagnostics = []
    summary_rows = []

    sensitivity_tag = None

    if not args.no_outlier_sensitivity:
        sensitivity_tag = (
            "pairedMAD{0}".format(
                number_token(
                    args.outlier_z
                )
            )
        )

    metadata = {
        "icc_type": "ICC(2,1)",
        "model": "two-way random effects",
        "definition": (
            "absolute agreement, "
            "single measurement"
        ),
        "sessions": [
            args.session_a,
            args.session_b,
        ],
        "space": SPACE,
        "one_map_per_metric": True,
        "analysis_mask": (
            "union of fixed template GM and "
            "WM probability masks"
        ),
        "gm_probseg": str(args.gm_probseg),
        "wm_probseg": str(args.wm_probseg),
        "gm_threshold": args.gm_threshold,
        "wm_threshold": args.wm_threshold,
        "gm_erosion_mm": args.gm_erosion_mm,
        "wm_erosion_mm": args.wm_erosion_mm,
        "gm_eroded_voxels": int(
            np.count_nonzero(masks["gm"])
        ),
        "wm_eroded_voxels": int(
            np.count_nonzero(masks["wm"])
        ),
        "analysis_mask_voxels": int(
            np.count_nonzero(masks["gmwm"])
        ),
        "summary_tissues": list(
            args.summary_tissues
        ),
        "gmwm_hybrid_noddi_pairs": hybrid_pairs,
        "min_subjects": args.min_subjects,
        "zero_values_excluded": (
            not args.allow_zero
        ),
        "outlier_sensitivity_enabled": (
            not args.no_outlier_sensitivity
        ),
        "outlier_compartment_handling": (
            "GM and WM are processed separately "
            "using fixed eroded masks. Outlier "
            "statistics are voxelwise across "
            "subjects and never pool GM and WM "
            "values."
        ),
        "outlier_definition": (
            "Remove the complete subject pair "
            "at a voxel when the absolute "
            "modified MAD z-score of the "
            "subject-pair mean or session "
            "difference exceeds the threshold. "
            "No outlier is flagged for a "
            "criterion when its voxelwise MAD "
            "is zero."
        ),
        "outlier_z_threshold": (
            args.outlier_z
        ),
        "min_retained_fraction": (
            args.min_retained_fraction
        ),
        "subjects_discovered": len(subjects),
        "metrics_requested": [
            spec.label
            for spec in metric_specs
        ],
        "analysis_sets": analysis_sets,
    }

    with (
        args.output_dir
        / "voxelwise_icc_metadata.json"
    ).open("w") as fobj:
        json.dump(
            metadata,
            fobj,
            indent=2,
        )

    for (
        metric_index,
        metric_spec,
    ) in enumerate(metric_specs):
        print(
            "Metric {0}/{1}: {2}".format(
                metric_index + 1,
                len(metric_specs),
                metric_spec.label,
            ),
            flush=True,
        )

        (
            pairs,
            diagnostics,
        ) = collect_subject_pairs(
            args.derivatives_dir,
            patterns,
            qc,
            subjects,
            args.session_a,
            args.session_b,
            metric_spec,
        )

        all_diagnostics.extend(
            diagnostics
        )

        needs_gmwm_hybrid = (
            "gmwm" in args.summary_tissues
            and metric_spec.label in hybrid_pairs
        )

        if needs_gmwm_hybrid:
            gm_counterpart = spec_by_label[
                hybrid_pairs[metric_spec.label]
            ]
            (
                gm_pairs,
                gm_diagnostics,
            ) = collect_subject_pairs(
                args.derivatives_dir,
                patterns,
                qc,
                subjects,
                args.session_a,
                args.session_b,
                gm_counterpart,
            )
            all_diagnostics.extend(
                gm_diagnostics
            )
            pairs = pair_gmwm_hybrid_subject_pairs(
                pairs,
                gm_pairs,
            )

        if len(pairs) < args.min_subjects:
            print(
                "  Skipping {0}: only {1} "
                "complete subject pairs".format(
                    metric_spec.label,
                    len(pairs),
                ),
                flush=True,
            )
            continue

        paths = map_paths(
            args.output_dir,
            metric_spec,
            sensitivity_tag,
            hybrid=needs_gmwm_hybrid,
        )

        result = None

        if not args.force:
            result = load_existing_result(
                paths,
                reference,
                do_outlier_sensitivity=(
                    not args.no_outlier_sensitivity
                ),
            )

            if result is not None:
                print(
                    "  Reusing existing combined "
                    "maps and regenerating "
                    "summaries",
                    flush=True,
                )
                result = restrict_result_to_metric_tissues(
                    result,
                    metric_spec,
                    masks,
                    hybrid=needs_gmwm_hybrid,
                )
                write_result_maps(
                    result,
                    paths,
                    reference,
                    do_outlier_sensitivity=(
                        not args.no_outlier_sensitivity
                    ),
                )
        values = None
        values_path = None

        metric_work_dir = (
            args.work_dir
            / metric_slug(metric_spec)
        )

        metric_work_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        try:
            if result is None:
                (
                    values,
                    values_path,
                ) = build_metric_memmap(
                    pairs,
                    reference,
                    metric_work_dir,
                    metric_slug(metric_spec),
                    gm_mask=(
                        masks["gm"]
                        if needs_gmwm_hybrid
                        else None
                    ),
                )

                if (
                    "gm" in metric_spec.tissues
                    or needs_gmwm_hybrid
                ):
                    print(
                        "  Processing eroded GM "
                        "compartment",
                        flush=True,
                    )

                    gm_result = process_compartment(
                        values,
                        masks["gm"],
                        "GM",
                        min_subjects=(
                            args.min_subjects
                        ),
                        outlier_z=args.outlier_z,
                        min_retained_fraction=(
                            args.min_retained_fraction
                        ),
                        remove_zeros=(
                            not args.allow_zero
                        ),
                        chunk_size=args.chunk_size,
                        do_outlier_sensitivity=(
                            not args.no_outlier_sensitivity
                        ),
                    )
                else:
                    print(
                        "  Skipping GM compartment "
                        "for tissue-specific metric",
                        flush=True,
                    )
                    gm_result = empty_result(
                        len(masks["gm"])
                    )

                if "wm" in metric_spec.tissues:
                    print(
                        "  Processing eroded WM "
                        "compartment",
                        flush=True,
                    )

                    wm_result = process_compartment(
                        values,
                        masks["wm"],
                        "WM",
                        min_subjects=(
                            args.min_subjects
                        ),
                        outlier_z=args.outlier_z,
                        min_retained_fraction=(
                            args.min_retained_fraction
                        ),
                        remove_zeros=(
                            not args.allow_zero
                        ),
                        chunk_size=args.chunk_size,
                        do_outlier_sensitivity=(
                            not args.no_outlier_sensitivity
                        ),
                    )
                else:
                    print(
                        "  Skipping WM compartment "
                        "for tissue-specific metric",
                        flush=True,
                    )
                    wm_result = empty_result(
                        len(masks["wm"])
                    )

                result = (
                    combine_compartment_results(
                        gm_result,
                        wm_result,
                        masks["gm"],
                        masks["wm"],
                    )
                )
                result = restrict_result_to_metric_tissues(
                    result,
                    metric_spec,
                    masks,
                    hybrid=needs_gmwm_hybrid,
                )

                write_result_maps(
                    result,
                    paths,
                    reference,
                    do_outlier_sensitivity=(
                        not args.no_outlier_sensitivity
                    ),
                )

            for analysis_set in analysis_sets:
                for tissue in args.summary_tissues:
                    if (
                        tissue not in metric_spec.tissues
                        or metric_spec.label
                        not in labels_by_tissue[tissue][
                            analysis_set
                        ]
                    ):
                        continue

                    display_label = display_labels_by_tissue[
                        tissue
                    ][analysis_set].get(
                        metric_spec.label,
                        metric_spec.label,
                    )

                    summary_rows.append(
                        summarize_icc_map(
                            metric_spec,
                            display_label,
                            analysis_set,
                            tissue,
                            "primary",
                            result["primary_icc"],
                            result["primary_n"],
                            masks[tissue],
                            len(pairs),
                        )
                    )

                    if not args.no_outlier_sensitivity:
                        summary_rows.append(
                            summarize_icc_map(
                                metric_spec,
                                display_label,
                                analysis_set,
                                tissue,
                                sensitivity_tag,
                                result[
                                    "sensitivity_icc"
                                ],
                                result[
                                    "sensitivity_n"
                                ],
                                masks[tissue],
                                len(pairs),
                            )
                        )

        finally:
            if values is not None:
                values.flush()
                del values

            if not args.keep_work_files:
                if (
                    values_path is not None
                    and Path(values_path).exists()
                ):
                    Path(values_path).unlink()

                try:
                    metric_work_dir.rmdir()
                except OSError:
                    pass

        # Write incremental tables after each metric so completed results
        # survive if a later metric fails.
        pd.DataFrame(
            all_diagnostics
        ).to_csv(
            args.output_dir
            / (
                "voxelwise_icc_subject_"
                "diagnostics.tsv"
            ),
            sep="\t",
            index=False,
        )

        incremental_summary = add_rank_column(
            pd.DataFrame(summary_rows)
        )

        incremental_summary.to_csv(
            args.output_dir
            / "voxelwise_icc_summary.tsv",
            sep="\t",
            index=False,
        )

        incremental_summary.to_csv(
            args.output_dir
            / (
                "voxelwise_icc_ranked_"
                "summary.tsv"
            ),
            sep="\t",
            index=False,
        )

    diagnostics_df = pd.DataFrame(
        all_diagnostics
    )

    diagnostics_df.to_csv(
        args.output_dir
        / (
            "voxelwise_icc_subject_"
            "diagnostics.tsv"
        ),
        sep="\t",
        index=False,
    )

    summary = add_rank_column(
        pd.DataFrame(summary_rows)
    )

    summary.to_csv(
        args.output_dir
        / "voxelwise_icc_summary.tsv",
        sep="\t",
        index=False,
    )

    summary.to_csv(
        args.output_dir
        / "voxelwise_icc_ranked_summary.tsv",
        sep="\t",
        index=False,
    )

    plot_ranked_summaries(
        summary,
        args.output_dir,
    )

    print(
        "Wrote outputs to {0}".format(
            args.output_dir
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
