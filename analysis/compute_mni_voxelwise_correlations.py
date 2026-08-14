#!/usr/bin/env python3
"""Compute voxelwise correlation matrices in MNI space.

For each subject/session and tissue mask, this script loads configured
space-MNI152NLin2009cAsym scalar maps, computes pairwise-valid voxelwise
correlations, Fisher-z transforms them, and averages first within subject and
then across subjects. By default, tissue masks come from each subject/session's
smriprep MNI-space deterministic dseg, with GM=1 and WM=2. Full supplementary
matrices are computed first; primary-analysis matrices are then written as
subsets of those full matrices.
"""

from __future__ import annotations

import argparse
import json
import re
import warnings
from pathlib import Path
from typing import Iterable

try:
    import nibabel as nib
    import numpy as np
    import pandas as pd
    from nibabel.processing import resample_from_to
    from scipy.stats import rankdata
except ImportError:  # pragma: no cover - checked after argparse handles --help
    nib = None
    np = None
    pd = None
    resample_from_to = None
    rankdata = None

from metric_registry import MetricSpec
from metric_registry import (
    build_metric_specs,
    metric_display_labels,
    metric_order,
    primary_metric_specs,
)


TISSUES = ('gm', 'wm')
SPACE = 'MNI152NLin2009cAsym'


def require_dependencies() -> None:
    missing = [
        name
        for name, module in (
            ('nibabel', nib),
            ('numpy', np),
            ('pandas', pd),
            ('scipy.stats', rankdata),
        )
        if module is None
    ]
    if missing:
        raise RuntimeError(
            'Missing required Python packages: '
            f'{", ".join(missing)}. Activate the NIBS processing environment first.'
        )


def normalize_subject(value: str) -> str:
    token = str(value).strip()
    return token if token.startswith('sub-') else f'sub-{token}'


def normalize_session(value: str) -> str:
    token = str(value).strip()
    return token if token.startswith('ses-') else f'ses-{token}'


def subject_for_qc(subject: object) -> str:
    return re.sub(r'^sub-', '', str(subject).strip())


def is_pilot_subject(subject: object) -> bool:
    return subject_for_qc(subject).upper().startswith('PILOT')


def session_label(session: object) -> str:
    match = re.search(r'(\d+)', str(session))
    if match is None:
        raise ValueError(f'Could not parse session number from {session}')
    return f'Session {int(match.group(1)):02d}'


def load_patterns(path: Path) -> dict[str, str]:
    with path.open() as fobj:
        nested = json.load(fobj)
    return {key: value for group in nested.values() for key, value in group.items()}


def first_glob(patterns: Iterable[Path]) -> Path | None:
    matches: list[Path] = []
    for pattern in patterns:
        matches.extend(sorted(pattern.parent.glob(pattern.name)))
    return sorted(set(matches))[0] if matches else None


def pattern_path(
    derivatives: Path,
    rel_pattern: str,
    subject: str,
    session: str,
    space: str,
) -> Path:
    rel_pattern = rel_pattern.replace('_space-MNI152NLin2009cAsym_', '_space-{space}_')
    return derivatives / rel_pattern.format(subject=subject, session=session, space=space)


def discover_subjects(derivatives: Path) -> list[str]:
    roots = (
        derivatives / 'smriprep',
        derivatives / 'qsirecon' / 'derivatives' / 'qsirecon-DIPYDKI',
        derivatives / 'pymp2rage',
    )
    return sorted(
        {
            path.name
            for root in roots
            if root.is_dir()
            for path in root.glob('sub-*')
            if path.is_dir() and not is_pilot_subject(path.name)
        }
    )


def discover_sessions(derivatives: Path, subject: str) -> list[str]:
    roots = (
        derivatives / 'qsirecon' / 'derivatives' / 'qsirecon-DIPYDKI' / subject,
        derivatives / 'pymp2rage' / subject,
        derivatives / 'ihmt' / subject,
    )
    return sorted(
        {
            path.name
            for root in roots
            if root.is_dir()
            for path in root.glob('ses-*')
            if path.is_dir()
        }
    )


def load_qc_table(path: Path | None) -> pd.DataFrame | None:
    if path is None:
        return None
    qc = pd.read_csv(path, sep='\t')
    qc['participant_id'] = qc['participant_id'].map(subject_for_qc)
    qc = qc.loc[~qc['participant_id'].map(is_pilot_subject)].copy()
    return qc.set_index('participant_id', drop=False)


def qc_passes(qc: pd.DataFrame | None, subject: str, session: str, spec: MetricSpec) -> bool:
    if qc is None:
        return True
    if not spec.qc_modalities:
        warnings.warn(f'No QC modality mapping for {spec.label}; applying no modality QC')
        return True
    subject_id = subject_for_qc(subject)
    if subject_id not in qc.index:
        return False
    row = qc.loc[subject_id]
    prefix = session_label(session)
    for modality in spec.qc_modalities:
        column = f'{prefix}--{modality}'
        if column not in qc.columns:
            raise RuntimeError(f'QC file is missing required column: {column}')
        value = row[column]
        if pd.isna(value) or int(value) != 1:
            return False
    return True


def load_like(path: Path, reference: nib.spatialimages.SpatialImage, order: int) -> np.ndarray:
    image = nib.load(str(path))
    if image.shape[:3] != reference.shape[:3] or not np.allclose(
        image.affine, reference.affine, atol=1e-4
    ):
        image = resample_from_to(image, reference, order=order)
    return np.asarray(image.get_fdata(), dtype=np.float32)


def erode_mask_mm(
    mask: np.ndarray,
    reference: nib.spatialimages.SpatialImage,
    erosion_mm: float,
) -> np.ndarray:
    mask_3d = np.asarray(mask, dtype=bool).reshape(reference.shape[:3])
    if erosion_mm <= 0:
        return mask_3d.reshape(-1)
    from scipy.ndimage import distance_transform_edt

    voxel_sizes = tuple(float(value) for value in nib.affines.voxel_sizes(reference.affine))
    distance = distance_transform_edt(mask_3d, sampling=voxel_sizes)
    return (distance > float(erosion_mm)).reshape(-1)


def build_template_tissue_masks(
    gm_probseg: Path,
    wm_probseg: Path,
    gm_threshold: float,
    wm_threshold: float,
    gm_erosion_mm: float,
    wm_erosion_mm: float,
) -> tuple[nib.spatialimages.SpatialImage, dict[str, np.ndarray]]:
    reference = nib.load(str(gm_probseg))
    gm_probability = load_like(gm_probseg, reference, order=1).reshape(-1)
    wm_probability = load_like(wm_probseg, reference, order=1).reshape(-1)
    gm = erode_mask_mm(gm_probability >= gm_threshold, reference, gm_erosion_mm)
    wm = erode_mask_mm(wm_probability >= wm_threshold, reference, wm_erosion_mm)
    overlap = gm & wm
    if np.any(overlap):
        overlap_indices = np.flatnonzero(overlap)
        gm_wins = gm_probability[overlap] >= wm_probability[overlap]
        gm[overlap_indices[~gm_wins]] = False
        wm[overlap_indices[gm_wins]] = False
    if not np.any(gm):
        raise RuntimeError('Template GM mask is empty after thresholding/erosion.')
    if not np.any(wm):
        raise RuntimeError('Template WM mask is empty after thresholding/erosion.')
    return reference, {'gm': gm, 'wm': wm}


def smriprep_dseg_candidates(
    derivatives: Path,
    subject: str,
    session: str,
    space: str,
) -> tuple[Path, ...]:
    base_name = f'{subject}_acq-*_rec-refaced_run-01_space-{space}_dseg.nii.gz'
    session_name = f'{subject}_{session}_acq-*_rec-refaced_run-01_space-{space}_dseg.nii.gz'
    return (
        derivatives / 'smriprep' / subject / 'anat' / base_name,
        derivatives / 'smriprep' / subject / session / 'anat' / session_name,
    )


def find_smriprep_dseg(
    derivatives: Path,
    subject: str,
    session: str,
    space: str,
) -> Path | None:
    return first_glob(smriprep_dseg_candidates(derivatives, subject, session, space))


def build_subject_tissue_masks(
    dseg: Path,
    gm_erosion_mm: float,
    wm_erosion_mm: float,
) -> tuple[nib.spatialimages.SpatialImage, dict[str, np.ndarray]]:
    reference = nib.load(str(dseg))
    segmentation = load_like(dseg, reference, order=0).reshape(-1)
    gm = erode_mask_mm(np.rint(segmentation).astype(np.int16) == 1, reference, gm_erosion_mm)
    wm = erode_mask_mm(np.rint(segmentation).astype(np.int16) == 2, reference, wm_erosion_mm)
    if not np.any(gm):
        raise RuntimeError(f'Subject GM mask is empty after erosion: {dseg}')
    if not np.any(wm):
        raise RuntimeError(f'Subject WM mask is empty after erosion: {dseg}')
    return reference, {'gm': gm, 'wm': wm}


def robust_outlier_mask(values: np.ndarray, z_threshold: float) -> np.ndarray:
    finite = np.isfinite(values)
    out = finite.copy()
    if not finite.any():
        return out
    clean = values[finite]
    median = float(np.median(clean))
    mad = float(np.median(np.abs(clean - median)))
    if np.isfinite(mad) and mad > 0:
        robust_z = 0.67448975 * (values - median) / mad
        return finite & (np.abs(robust_z) <= z_threshold)
    q_low, q_high = np.percentile(clean, [0.1, 99.9])
    return finite & (values >= q_low) & (values <= q_high)


def metric_paths_for_session(
    derivatives: Path,
    patterns: dict[str, str],
    specs: list[MetricSpec],
    qc: pd.DataFrame | None,
    subject: str,
    session: str,
    space: str,
) -> dict[str, Path]:
    paths: dict[str, Path] = {}
    for spec in specs:
        if not qc_passes(qc, subject, session, spec):
            continue
        rel_pattern = patterns.get(spec.pattern_key)
        if rel_pattern is None:
            warnings.warn(f'No {space} pattern found for {spec.label}: {spec.pattern_key}')
            continue
        path = first_glob((pattern_path(derivatives, rel_pattern, subject, session, space),))
        if path is not None:
            paths[spec.label] = path
    return paths


def compute_profile_correlations(
    data: pd.DataFrame,
    tissue_mask: np.ndarray,
    tissue: str,
    method: str,
    outlier_z: float,
    min_voxels: int,
    outlier_masks: tuple[np.ndarray, ...] | None = None,
) -> tuple[pd.DataFrame | None, pd.DataFrame | None, pd.DataFrame | None, dict[str, object]]:
    labels = list(data.columns)
    matrix = data.to_numpy(dtype=np.float32)
    n_metrics = len(labels)
    n_tissue_voxels = int(np.count_nonzero(tissue_mask))
    valid_masks: list[np.ndarray] = []
    compartment_masks = (
        tuple(np.asarray(mask, dtype=bool) for mask in outlier_masks)
        if outlier_masks is not None
        else (np.asarray(tissue_mask, dtype=bool),)
    )

    for col_idx in range(n_metrics):
        valid_mask = np.zeros(tissue_mask.shape, dtype=bool)
        for compartment_mask in compartment_masks:
            finite_nonzero = (
                compartment_mask
                & tissue_mask
                & np.isfinite(matrix[:, col_idx])
                & (matrix[:, col_idx] != 0)
            )
            if np.any(finite_nonzero):
                outlier_mask = robust_outlier_mask(matrix[finite_nonzero, col_idx], outlier_z)
                valid_mask[np.flatnonzero(finite_nonzero)] = outlier_mask
        valid_masks.append(valid_mask)

    corr = np.full((n_metrics, n_metrics), np.nan, dtype=float)
    voxel_counts = np.zeros((n_metrics, n_metrics), dtype=np.int64)
    voxel_proportions = np.full((n_metrics, n_metrics), np.nan, dtype=float)

    for idx in range(n_metrics):
        corr[idx, idx] = 1.0
        n_valid = int(np.count_nonzero(valid_masks[idx]))
        voxel_counts[idx, idx] = n_valid
        voxel_proportions[idx, idx] = n_valid / n_tissue_voxels if n_tissue_voxels else np.nan

    for idx_a in range(n_metrics):
        for idx_b in range(idx_a + 1, n_metrics):
            pair_mask = valid_masks[idx_a] & valid_masks[idx_b]
            n_voxels = int(np.count_nonzero(pair_mask))
            voxel_counts[idx_a, idx_b] = n_voxels
            voxel_counts[idx_b, idx_a] = n_voxels
            pair_proportion = n_voxels / n_tissue_voxels if n_tissue_voxels else np.nan
            voxel_proportions[idx_a, idx_b] = pair_proportion
            voxel_proportions[idx_b, idx_a] = pair_proportion
            if n_voxels < min_voxels:
                continue
            values_a = matrix[pair_mask, idx_a]
            values_b = matrix[pair_mask, idx_b]
            if method == 'spearman':
                values_a = rankdata(values_a).astype(np.float32)
                values_b = rankdata(values_b).astype(np.float32)
            elif method != 'pearson':
                raise ValueError(f'Unsupported correlation method: {method}')
            if np.std(values_a) == 0 or np.std(values_b) == 0:
                continue
            corr_value = float(np.corrcoef(values_a, values_b)[0, 1])
            corr[idx_a, idx_b] = corr_value
            corr[idx_b, idx_a] = corr_value

    max_pair_voxels = int(np.max(voxel_counts)) if voxel_counts.size else 0
    upper_triangle = np.triu(np.ones((n_metrics, n_metrics), dtype=bool), k=1)
    valid_pairs = upper_triangle & np.isfinite(corr)
    pair_counts = voxel_counts[upper_triangle]
    pair_proportions = voxel_proportions[upper_triangle]
    finite_pair_proportions = pair_proportions[np.isfinite(pair_proportions)]
    diagnostics = {
        'tissue': tissue,
        'n_metrics': len(labels),
        'n_tissue_voxels': n_tissue_voxels,
        'n_valid_metric_pairs': int(np.count_nonzero(valid_pairs)),
        'n_total_metric_pairs': int(n_metrics * (n_metrics - 1) / 2),
        'min_pairwise_voxels': int(np.min(pair_counts)) if pair_counts.size else 0,
        'median_pairwise_voxels': float(np.median(pair_counts)) if pair_counts.size else np.nan,
        'max_pairwise_voxels': max_pair_voxels,
        'min_pairwise_proportion': (
            float(np.min(finite_pair_proportions)) if finite_pair_proportions.size else np.nan
        ),
        'median_pairwise_proportion': (
            float(np.median(finite_pair_proportions)) if finite_pair_proportions.size else np.nan
        ),
        'max_pairwise_proportion': (
            float(np.max(finite_pair_proportions)) if finite_pair_proportions.size else np.nan
        ),
    }
    if not np.any(valid_pairs):
        count_df = pd.DataFrame(voxel_counts, index=labels, columns=labels)
        proportion_df = pd.DataFrame(voxel_proportions, index=labels, columns=labels)
        return None, count_df, proportion_df, diagnostics
    corr = np.clip(corr, -0.999999, 0.999999)
    corr_df = pd.DataFrame(corr, index=labels, columns=labels)
    count_df = pd.DataFrame(voxel_counts, index=labels, columns=labels)
    proportion_df = pd.DataFrame(voxel_proportions, index=labels, columns=labels)
    return corr_df, count_df, proportion_df, diagnostics


def pairwise_coverage_rows(
    count_df: pd.DataFrame,
    proportion_df: pd.DataFrame,
    subject: str,
    session: str,
    tissue: str,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    labels = list(count_df.index)
    for idx_a, metric_a in enumerate(labels):
        for idx_b in range(idx_a + 1, len(labels)):
            metric_b = labels[idx_b]
            rows.append(
                {
                    'subject': subject,
                    'session': session,
                    'tissue': tissue,
                    'metric_a': metric_a,
                    'metric_b': metric_b,
                    'n_voxels': int(count_df.loc[metric_a, metric_b]),
                    'proportion_tissue_voxels': float(proportion_df.loc[metric_a, metric_b]),
                }
            )
    return rows


def write_metric_inclusion(
    out_file: Path,
    tissue: str,
    analysis_set: str,
    expected_labels: list[str],
    observed_labels: set[str],
    included_labels: list[str],
    display: dict[str, str],
) -> None:
    included = set(included_labels)
    rows = []
    for label in expected_labels:
        observed = label in observed_labels
        in_matrix = label in included
        rows.append(
            {
                'analysis_set': analysis_set,
                'tissue': tissue,
                'metric_key': label,
                'metric': display.get(label, label),
                'expected': True,
                'observed_after_qc': observed,
                'included': in_matrix,
                'reason_if_not_included': (
                    ''
                    if in_matrix
                    else (
                        'not_observed_after_qc'
                        if not observed
                        else 'fewer_than_min_metrics_or_no_valid_correlations'
                    )
                ),
            }
        )
    pd.DataFrame(rows).to_csv(out_file, sep='\t', index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--project-root',
        type=Path,
        default=Path('/cbica/projects/nibs'),
        help='Project root containing derivatives, code, and data.',
    )
    parser.add_argument('--derivatives-dir', type=Path, default=None)
    parser.add_argument('--patterns-file', type=Path, default=None)
    parser.add_argument('--qc-file', type=Path, default=None)
    parser.add_argument('--subject-id', action='append', help='Subject(s), with or without sub-.')
    parser.add_argument('--session-id', action='append', help='Session(s), with or without ses-.')
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=None,
        help='Defaults to <project-root>/derivatives/mni_voxelwise_correlations.',
    )
    parser.add_argument('--outlier-z', type=float, default=6.0)
    parser.add_argument('--min-voxels', type=int, default=2)
    parser.add_argument(
        '--correlation',
        choices=('pearson', 'spearman'),
        default='pearson',
        help='Voxelwise correlation method. Spearman ranks voxels within each metric first.',
    )
    parser.add_argument(
        '--min-metrics',
        type=int,
        default=2,
        help='Minimum number of selected metrics required for a subject/session profile.',
    )
    parser.add_argument(
        '--tissue-mask-source',
        choices=('subject-dseg', 'template-probseg'),
        default='subject-dseg',
        help='Use subject/session smriprep dseg masks, or the same template probability masks for every profile.',
    )
    parser.add_argument(
        '--gm-probseg',
        type=Path,
        default=None,
        help='Template-space GM probability map used with --tissue-mask-source template-probseg.',
    )
    parser.add_argument(
        '--wm-probseg',
        type=Path,
        default=None,
        help='Template-space WM probability map used with --tissue-mask-source template-probseg.',
    )
    parser.add_argument('--gm-threshold', type=float, default=0.50)
    parser.add_argument('--wm-threshold', type=float, default=0.50)
    parser.add_argument('--gm-erosion-mm', type=float, default=0.0)
    parser.add_argument('--wm-erosion-mm', type=float, default=0.0)
    parser.add_argument(
        '--no-qc',
        action='store_true',
        help='Do not apply manual modality QC even if the default QC file exists.',
    )
    args = parser.parse_args()
    args.project_root = args.project_root.expanduser().resolve()
    args.derivatives_dir = (
        args.derivatives_dir.expanduser().resolve()
        if args.derivatives_dir
        else args.project_root / 'derivatives'
    )
    args.patterns_file = (
        args.patterns_file.expanduser().resolve()
        if args.patterns_file
        else args.project_root / 'code' / 'configuration' / 'patterns.json'
    )
    if not args.patterns_file.exists():
        fallback = Path(__file__).resolve().parents[1] / 'configuration' / 'patterns.json'
        args.patterns_file = fallback
    if args.qc_file is None and not args.no_qc:
        candidates = (
            args.project_root / 'code' / 'data' / 'manual_qc_modality.tsv',
            Path(__file__).resolve().parents[1] / 'data' / 'manual_qc_modality.tsv',
        )
        args.qc_file = next((path for path in candidates if path.exists()), None)
    elif args.qc_file is not None:
        args.qc_file = args.qc_file.expanduser().resolve()
    args.output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir
        else args.project_root / 'derivatives' / 'mni_voxelwise_correlations'
    )
    data_candidates = (
        args.project_root / 'code' / 'data',
        Path(__file__).resolve().parents[1] / 'data',
    )
    if args.gm_probseg is None:
        args.gm_probseg = next(
            (
                directory / f'tpl-{SPACE}_res-01_label-GM_probseg.nii.gz'
                for directory in data_candidates
                if (directory / f'tpl-{SPACE}_res-01_label-GM_probseg.nii.gz').exists()
            ),
            data_candidates[0] / f'tpl-{SPACE}_res-01_label-GM_probseg.nii.gz',
        )
    else:
        args.gm_probseg = args.gm_probseg.expanduser().resolve()
    if args.wm_probseg is None:
        args.wm_probseg = next(
            (
                directory / f'tpl-{SPACE}_res-01_label-WM_probseg.nii.gz'
                for directory in data_candidates
                if (directory / f'tpl-{SPACE}_res-01_label-WM_probseg.nii.gz').exists()
            ),
            data_candidates[0] / f'tpl-{SPACE}_res-01_label-WM_probseg.nii.gz',
        )
    else:
        args.wm_probseg = args.wm_probseg.expanduser().resolve()
    if args.tissue_mask_source == 'template-probseg':
        if not args.gm_probseg.exists():
            raise FileNotFoundError(f'GM probability map not found: {args.gm_probseg}')
        if not args.wm_probseg.exists():
            raise FileNotFoundError(f'WM probability map not found: {args.wm_probseg}')
        if not (0.0 < args.gm_threshold <= 1.0):
            parser.error('--gm-threshold must be in (0, 1].')
        if not (0.0 < args.wm_threshold <= 1.0):
            parser.error('--wm-threshold must be in (0, 1].')
    if args.gm_erosion_mm < 0 or args.wm_erosion_mm < 0:
        parser.error('Tissue mask erosion distances must be nonnegative.')
    return args


def main() -> None:
    args = parse_args()
    require_dependencies()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    patterns = load_patterns(args.patterns_file)
    specs = build_metric_specs(args.patterns_file)
    primary_specs = primary_metric_specs(specs)
    if len(primary_specs) != len(metric_order(specs, 'primary')):
        raise RuntimeError('Primary metric registry contains duplicate labels.')
    qc = load_qc_table(args.qc_file)

    subjects = (
        [normalize_subject(subject) for subject in args.subject_id]
        if args.subject_id
        else discover_subjects(args.derivatives_dir)
    )
    diagnostics: list[dict[str, object]] = []
    pairwise_coverage: list[dict[str, object]] = []
    template_reference = None
    template_tissue_masks = None
    if args.tissue_mask_source == 'template-probseg':
        template_reference, template_tissue_masks = build_template_tissue_masks(
            args.gm_probseg,
            args.wm_probseg,
            args.gm_threshold,
            args.wm_threshold,
            args.gm_erosion_mm,
            args.wm_erosion_mm,
        )
    z_mats: dict[str, list[pd.DataFrame]] = {tissue: [] for tissue in TISSUES}
    count_mats: dict[str, list[pd.DataFrame]] = {tissue: [] for tissue in TISSUES}
    proportion_mats: dict[str, list[pd.DataFrame]] = {tissue: [] for tissue in TISSUES}

    for subject in subjects:
        sessions = (
            [normalize_session(session) for session in args.session_id]
            if args.session_id
            else discover_sessions(args.derivatives_dir, subject)
        )
        subject_z_mats: dict[str, list[pd.DataFrame]] = {tissue: [] for tissue in TISSUES}
        subject_count_mats: dict[str, list[pd.DataFrame]] = {
            tissue: [] for tissue in TISSUES
        }
        subject_proportion_mats: dict[str, list[pd.DataFrame]] = {
            tissue: [] for tissue in TISSUES
        }
        for session in sessions:
            metric_paths = metric_paths_for_session(
                args.derivatives_dir,
                patterns,
                specs,
                qc,
                subject,
                session,
                SPACE,
            )
            if len(metric_paths) < args.min_metrics:
                print(f'Skipping {subject} {session}: only {len(metric_paths)} metrics')
                continue

            if args.tissue_mask_source == 'subject-dseg':
                dseg_path = find_smriprep_dseg(args.derivatives_dir, subject, session, SPACE)
                if dseg_path is None:
                    diagnostics.append(
                        {
                            'subject': subject,
                            'session': session,
                            'tissue': 'all',
                            'n_metrics': len(metric_paths),
                            'n_tissue_voxels': 0,
                            'n_valid_metric_pairs': 0,
                            'n_total_metric_pairs': 0,
                            'reason': 'missing_subject_dseg',
                            'tissue_mask_source': args.tissue_mask_source,
                            'tissue_mask_file': '',
                            'metrics': ','.join(metric_paths),
                        }
                    )
                    print(f'Skipping {subject} {session}: missing smriprep MNI dseg')
                    continue
                try:
                    reference, tissue_masks = build_subject_tissue_masks(
                        dseg_path,
                        args.gm_erosion_mm,
                        args.wm_erosion_mm,
                    )
                except RuntimeError as exc:
                    diagnostics.append(
                        {
                            'subject': subject,
                            'session': session,
                            'tissue': 'all',
                            'n_metrics': len(metric_paths),
                            'n_tissue_voxels': 0,
                            'n_valid_metric_pairs': 0,
                            'n_total_metric_pairs': 0,
                            'reason': str(exc),
                            'tissue_mask_source': args.tissue_mask_source,
                            'tissue_mask_file': str(dseg_path),
                            'metrics': ','.join(metric_paths),
                        }
                    )
                    print(f'Skipping {subject} {session}: {exc}')
                    continue
                tissue_mask_file = str(dseg_path)
            else:
                reference = template_reference
                tissue_masks = template_tissue_masks
                if reference is None or tissue_masks is None:
                    raise RuntimeError('Template tissue masks were not initialized.')
                tissue_mask_file = f'{args.gm_probseg},{args.wm_probseg}'

            metric_data = {
                label: load_like(path, reference, order=1).reshape(-1)
                for label, path in metric_paths.items()
            }
            data = pd.DataFrame(metric_data)
            for tissue in TISSUES:
                analysis_data = data
                tissue_labels = [
                    label
                    for label in metric_order(specs, 'full', tissue=tissue)
                    if label in analysis_data.columns
                ]
                tissue_data = analysis_data.loc[:, tissue_labels]
                if len(tissue_labels) < args.min_metrics:
                    diagnostics.append(
                        {
                            'subject': subject,
                            'session': session,
                            'tissue': tissue,
                            'n_metrics': len(tissue_labels),
                            'n_tissue_voxels': int(np.count_nonzero(tissue_masks[tissue])),
                            'n_valid_metric_pairs': 0,
                            'n_total_metric_pairs': 0,
                            'reason': 'too_few_tissue_eligible_metrics',
                            'tissue_mask_source': args.tissue_mask_source,
                            'tissue_mask_file': tissue_mask_file,
                            'metrics': ','.join(tissue_labels),
                        }
                    )
                    continue
                corr, counts, proportions, diag = compute_profile_correlations(
                    tissue_data,
                    tissue_masks[tissue],
                    tissue,
                    method=args.correlation,
                    outlier_z=args.outlier_z,
                    min_voxels=args.min_voxels,
                    outlier_masks=None,
                )
                diag.update(
                    {
                        'subject': subject,
                        'session': session,
                        'tissue_mask_source': args.tissue_mask_source,
                        'tissue_mask_file': tissue_mask_file,
                        'metrics': ','.join(tissue_data.columns),
                    }
                )
                diagnostics.append(diag)
                if counts is None or proportions is None:
                    continue
                pairwise_coverage.extend(
                    pairwise_coverage_rows(counts, proportions, subject, session, tissue)
                )
                if corr is None:
                    continue
                z = np.arctanh(np.clip(corr, -0.999999, 0.999999))
                np.fill_diagonal(z.values, 0.0)
                subject_z_mats[tissue].append(z)
                subject_count_mats[tissue].append(counts)
                subject_proportion_mats[tissue].append(proportions)

        for tissue, mats in subject_z_mats.items():
            if not mats:
                continue
            all_labels = sorted({label for mat in mats for label in mat.index})
            stack = np.stack(
                [mat.reindex(index=all_labels, columns=all_labels).to_numpy(dtype=float) for mat in mats]
            )
            mean_z = pd.DataFrame(
                np.nanmean(stack, axis=0),
                index=all_labels,
                columns=all_labels,
            )
            z_mats[tissue].append(mean_z)
            count_stack = np.stack(
                [
                    mat.reindex(index=all_labels, columns=all_labels).to_numpy(dtype=float)
                    for mat in subject_count_mats[tissue]
                ]
            )
            proportion_stack = np.stack(
                [
                    mat.reindex(index=all_labels, columns=all_labels).to_numpy(dtype=float)
                    for mat in subject_proportion_mats[tissue]
                ]
            )
            count_mats[tissue].append(
                pd.DataFrame(np.nanmean(count_stack, axis=0), index=all_labels, columns=all_labels)
            )
            proportion_mats[tissue].append(
                pd.DataFrame(
                    np.nanmean(proportion_stack, axis=0),
                    index=all_labels,
                    columns=all_labels,
                )
            )

    pd.DataFrame(diagnostics).to_csv(
        args.output_dir / 'mni_voxelwise_correlation_diagnostics.tsv',
        sep='\t',
        index=False,
    )
    pd.DataFrame(pairwise_coverage).to_csv(
        args.output_dir / 'mni_voxelwise_pairwise_correlation_coverage.tsv',
        sep='\t',
        index=False,
    )

    for tissue, mats in z_mats.items():
        if not mats:
            continue
        orders = {
            'full': metric_order(specs, 'full', tissue=tissue),
            'primary': metric_order(specs, 'primary', tissue=tissue),
        }
        display_labels = {
            analysis_set: metric_display_labels(specs, analysis_set, tissue=tissue)
            for analysis_set in orders
        }
        full_labels = [label for label in orders['full'] if any(label in mat.index for mat in mats)]
        observed_labels = set(full_labels)
        stack = np.stack(
            [
                mat.reindex(index=full_labels, columns=full_labels).to_numpy(dtype=float)
                for mat in mats
            ]
        )
        full_mean_z = pd.DataFrame(np.nanmean(stack, axis=0), index=full_labels, columns=full_labels)
        full_n_subjects = pd.DataFrame(
            np.sum(np.isfinite(stack), axis=0),
            index=full_labels,
            columns=full_labels,
        )
        count_stack = np.stack(
            [
                mat.reindex(index=full_labels, columns=full_labels).to_numpy(dtype=float)
                for mat in count_mats[tissue]
            ]
        )
        proportion_stack = np.stack(
            [
                mat.reindex(index=full_labels, columns=full_labels).to_numpy(dtype=float)
                for mat in proportion_mats[tissue]
            ]
        )
        full_mean_counts = pd.DataFrame(
            np.nanmean(count_stack, axis=0),
            index=full_labels,
            columns=full_labels,
        )
        full_mean_proportions = pd.DataFrame(
            np.nanmean(proportion_stack, axis=0),
            index=full_labels,
            columns=full_labels,
        )

        for analysis_set, ordered_labels in orders.items():
            labels = [label for label in ordered_labels if label in full_mean_z.index]
            label_map = display_labels[analysis_set]
            stem = f'mni_voxelwise_{analysis_set}_{tissue}_{args.correlation}'
            write_metric_inclusion(
                args.output_dir / f'{stem}_metric_inclusion.tsv',
                tissue,
                analysis_set,
                ordered_labels,
                observed_labels,
                labels,
                label_map,
            )
            if len(labels) < 2:
                continue
            mean_z = full_mean_z.reindex(index=labels, columns=labels)
            mean_r = pd.DataFrame(np.tanh(mean_z), index=labels, columns=labels)
            np.fill_diagonal(mean_r.values, 1.0)
            mean_counts = full_mean_counts.reindex(index=labels, columns=labels)
            mean_proportions = full_mean_proportions.reindex(index=labels, columns=labels)
            n_subjects = full_n_subjects.reindex(index=labels, columns=labels)
            mean_z = mean_z.rename(index=label_map, columns=label_map)
            mean_r = mean_r.rename(index=label_map, columns=label_map)
            mean_counts = mean_counts.rename(index=label_map, columns=label_map)
            mean_proportions = mean_proportions.rename(index=label_map, columns=label_map)
            n_subjects = n_subjects.rename(index=label_map, columns=label_map)
            mean_r.to_csv(
                args.output_dir / f'{stem}_r.tsv',
                sep='\t',
                index_label='metric',
            )
            mean_z.to_csv(
                args.output_dir / f'{stem}_fisherz.tsv',
                sep='\t',
                index_label='metric',
            )
            mean_counts.to_csv(
                args.output_dir / f'{stem}_mean_pairwise_nvoxels.tsv',
                sep='\t',
                index_label='metric',
            )
            mean_proportions.to_csv(
                args.output_dir / f'{stem}_mean_pairwise_proportion.tsv',
                sep='\t',
                index_label='metric',
            )
            n_subjects.to_csv(
                args.output_dir / f'{stem}_nsubjects.tsv',
                sep='\t',
                index_label='metric',
            )


if __name__ == '__main__':
    main()
