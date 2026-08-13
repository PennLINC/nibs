#!/usr/bin/env python3
"""Compute test-retest discriminability for WM bundle and DKT parcel profiles."""

from __future__ import annotations

import argparse
import re
import sys
from collections import Counter
from glob import glob
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from metric_registry import build_metric_specs, metric_display_labels, metric_order
from parcel_metric_utils import add_metric_metadata, canonical_metric_from_row
from path_utils import CODE_ROOT, DERIVATIVES_ROOT


DEFAULT_QC_FILE = CODE_ROOT / 'data' / 'manual_qc_modality.tsv'
QC_MODES = ('metricqc',)
ANALYSIS_SETS = ('primary', 'full')
RESULT_COLUMNS = [
    'profile_type',
    'profile_group',
    'stat',
    'distance_metric',
    'discriminability',
    'nearest_neighbor_accuracy',
    'mean_genuine_distance',
    'mean_impostor_distance',
    'mean_rank_percentile',
    'n_subjects',
    'n_sessions',
    'n_profiles',
    'n_features',
]
DEFAULT_WM_GLOBS = [
    str(DERIVATIVES_ROOT / 'qsirecon/derivatives/qsirecon-*/sub-*/ses-*/dwi/sub-*_ses-*_*_scalarstats.tsv'),
    str(DERIVATIVES_ROOT / 'bundle_myelin_stats/sub-*/ses-*/dwi/sub-*_ses-*_acq-HBCD75_run-01_space-T1w_model-*_scalarstats.tsv'),
]
DEFAULT_DKT_GLOBS = [
    str(
        DERIVATIVES_ROOT
        / 'DKTatlas_myelin_stats'
        / 'sub-*/sub-*_ses-*_run-*_desc-DKTatlas_scalarstats.csv'
    )
]
EXCLUDED_DKT_METRICS = {'G-ihMTsat', 'G-ihMTR'}
FILE_RE = re.compile(r'sub-(?P<sub>[^_]+)_(?P<ses>ses-[^_]+)_(?P<run>run-[^_]+)_')
PATH_RE = re.compile(r'sub-(?P<sub>[^_/]+).*(ses-(?P<ses>[^_/]+))')
REQUIRED_BUNDLE_COLUMNS = {'bundle', 'variable_name', 'masked_mean', 'masked_median'}
EXCLUDED_BUNDLE_PATTERNS = (
    'AnteriorCommissure',
    'DentatorubrothalamicTract-lr',
    'DentatorubrothalamicTract-rl',
    'DentatorubrothalamicTractlr',
    'DentatorubrothalamicTractrl',
)


def _normalize_subject(value: object) -> str:
    return re.sub(r'^sub-', '', str(value).strip())


def _is_pilot_subject(value: object) -> bool:
    return _normalize_subject(value).upper().startswith('PILOT')


def _session_label(value: object) -> str:
    match = re.search(r'(\d+)', str(value))
    if match is None:
        raise ValueError(f'Could not parse session number from: {value}')
    return f'Session {int(match.group(1)):02d}'


def load_qc_table(qc_file: Path) -> pd.DataFrame:
    qc_df = pd.read_csv(qc_file, sep='\t')
    if 'participant_id' not in qc_df.columns:
        raise RuntimeError(f'{qc_file} is missing participant_id')
    qc_df = qc_df.copy()
    qc_df['participant_id'] = qc_df['participant_id'].map(_normalize_subject)
    qc_df = qc_df.loc[~qc_df['participant_id'].map(_is_pilot_subject)].copy()
    return qc_df.set_index('participant_id', drop=False)


def metric_required_modalities(metric: str, patterns_file: Path) -> tuple[str, ...]:
    for spec in build_metric_specs(patterns_file):
        if metric in {spec.label, spec.primary_label}:
            return spec.qc_modalities
    raise ValueError(f'No QC modality mapping defined for metric: {metric}')


def _qc_passes(
    qc_df: pd.DataFrame,
    subject: object,
    session: object,
    modalities: tuple[str, ...],
) -> bool:
    subject_id = _normalize_subject(subject)
    if subject_id not in qc_df.index:
        return False
    row = qc_df.loc[subject_id]
    session_prefix = _session_label(session)
    for modality in modalities:
        column = f'{session_prefix}--{modality}'
        if column not in qc_df.columns:
            raise RuntimeError(f'QC file is missing required column: {column}')
        value = row[column]
        if pd.isna(value) or int(value) != 1:
            return False
    return True


def apply_metric_qc(
    df: pd.DataFrame,
    qc_df: pd.DataFrame,
    patterns_file: Path,
    subject_col: str = 'subject',
    session_col: str = 'session',
) -> pd.DataFrame:
    keep = [
        _qc_passes(
            qc_df,
            row[subject_col],
            row[session_col],
            metric_required_modalities(str(row['metric']), patterns_file),
        )
        for _, row in df.iterrows()
    ]
    return df.loc[keep].copy()


def parse_dkt_filename(path: Path) -> tuple[str, str, str]:
    match = FILE_RE.search(path.name)
    if match is None:
        raise ValueError(f'Could not parse subject/session/run from: {path}')
    return match.group('sub'), match.group('ses'), match.group('run')


def collect_rows(input_glob: str) -> pd.DataFrame:
    records = []
    for file_str in sorted(glob(input_glob)):
        path = Path(file_str)
        subject, session, run = parse_dkt_filename(path)
        if _is_pilot_subject(subject):
            continue
        df = pd.read_csv(path)
        df['subject'] = subject
        df['session'] = session
        df['run'] = run
        records.append(df)
    return pd.concat(records, ignore_index=True) if records else pd.DataFrame()


def build_value_table(
    df: pd.DataFrame,
    stat: str,
    patterns_file: Path,
) -> pd.DataFrame:
    base_cols = [
        'subject',
        'session',
        'run',
        'parcel_intensity',
        'parcel_name',
        'parcel_hemi',
        'parcel_count_t1w',
        'parcel_count_acpc',
    ]
    metric_cols = [col for col in df.columns if col.endswith(f'_{stat}')]
    if not metric_cols:
        raise RuntimeError(f'No columns found ending with _{stat}')
    value_df = df[base_cols + metric_cols].melt(
        id_vars=base_cols,
        value_vars=metric_cols,
        var_name='metric_stat',
        value_name='value',
    )
    value_df['metric'] = value_df['metric_stat'].str[: -(len(stat) + 1)]
    value_df = add_metric_metadata(value_df, 'metric', patterns_file)
    value_df['parcel'] = (
        value_df['parcel_hemi'].astype(str) + '_' + value_df['parcel_name'].astype(str)
    )
    return value_df.drop(columns=['metric_stat'])


def _parse_bundle_path(path: str) -> tuple[str | None, str | None]:
    match = PATH_RE.search(path)
    if match is None:
        return None, None
    return match.group('sub'), f'ses-{match.group("ses")}'


def collect_scalarstats(input_globs: list[str], patterns_file: Path) -> pd.DataFrame:
    rows = []
    all_files = set()
    dropped_counter: Counter[tuple[str, str]] = Counter()
    for input_glob in input_globs:
        for file_path in glob(input_glob):
            all_files.add(file_path)
    for file_path in sorted(all_files):
        df = pd.read_csv(file_path, sep='\t')
        missing = REQUIRED_BUNDLE_COLUMNS.difference(df.columns)
        if missing:
            raise RuntimeError(f'Missing required columns {missing} in {file_path}')
        if 'subject_id' not in df.columns or df['subject_id'].isna().all():
            parsed_sub, _ = _parse_bundle_path(file_path)
            if parsed_sub is None:
                raise RuntimeError(f'Could not infer subject_id from {file_path}')
            df['subject_id'] = parsed_sub
        if 'session_id' not in df.columns or df['session_id'].isna().all():
            _, parsed_ses = _parse_bundle_path(file_path)
            if parsed_ses is None:
                raise RuntimeError(f'Could not infer session_id from {file_path}')
            df['session_id'] = parsed_ses
        df['source_tsv'] = file_path
        df['metric'] = df.apply(
            lambda row: canonical_metric_from_row(row, patterns_file=patterns_file),
            axis=1,
        )
        for _, drow in df[df['metric'].isna()].iterrows():
            dropped_counter[
                (str(drow.get('variable_name', '')), str(drow.get('qsirecon_suffix', '')))
            ] += 1
        df = df.dropna(subset=['metric']).copy()
        if not df.empty:
            rows.append(df)
    if not rows:
        return pd.DataFrame()
    out = pd.concat(rows, ignore_index=True)
    n_rows_before_subject_filters = len(out)
    out['subject_id'] = out['subject_id'].astype(str).str.replace('^sub-', '', regex=True)
    out = out.loc[~out['subject_id'].map(_is_pilot_subject)].copy()
    out['session_id'] = out['session_id'].astype(str)
    out['bundle'] = out['bundle'].astype(str)
    excluded = out['bundle'].str.contains('|'.join(EXCLUDED_BUNDLE_PATTERNS), regex=True, na=False)
    if excluded.any():
        out = out.loc[~excluded].copy()
    metric_counts = out['metric'].value_counts().sort_index()
    print(
        '[INFO] Loaded WM scalarstats: '
        f'{len(all_files)} files, {n_rows_before_subject_filters} mapped rows before subject/bundle filters, '
        f'{len(out)} rows after filters, {len(metric_counts)} canonical metrics.',
        flush=True,
    )
    print(
        '[INFO] WM canonical metrics after input loading: '
        + ', '.join(f'{metric}={count}' for metric, count in metric_counts.items()),
        flush=True,
    )
    if dropped_counter:
        print('[WARN] Dropped rows with unmapped registry metrics (top 20):', flush=True)
        for (var_name, suffix), count in dropped_counter.most_common(20):
            print(f'  variable_name={var_name} qsirecon_suffix={suffix} n={count}', flush=True)
    return out


def apply_qc_mode(
    df: pd.DataFrame,
    qc_df: pd.DataFrame,
    qc_mode: str,
    profile_type: str,
    patterns_file: Path,
) -> pd.DataFrame:
    if profile_type in {'wm', 'dkt'}:
        if qc_mode == 'metricqc':
            return apply_metric_qc(df, qc_df, patterns_file)
    raise ValueError(f'Unsupported QC mode/profile_type: {qc_mode}/{profile_type}')


def _value_column(df: pd.DataFrame, stat: str, prefer_masked: bool) -> pd.Series:
    masked_col = f'masked_{stat}'
    if prefer_masked and masked_col in df.columns:
        masked = pd.to_numeric(df[masked_col], errors='coerce')
        raw = pd.to_numeric(df[stat], errors='coerce')
        return masked.where(np.isfinite(masked.to_numpy(dtype=float)), raw)
    return pd.to_numeric(df[stat], errors='coerce')


def _zscore_columns(matrix: pd.DataFrame) -> pd.DataFrame:
    means = matrix.mean(axis=0)
    stds = matrix.std(axis=0, ddof=0)
    keep = np.isfinite(stds.to_numpy(dtype=float)) & (stds.to_numpy(dtype=float) > 0)
    matrix = matrix.loc[:, keep]
    stds = stds.loc[matrix.columns]
    means = means.loc[matrix.columns]
    return (matrix - means) / stds


def _pairwise_distances(values: np.ndarray, metric: str) -> np.ndarray:
    if metric == 'euclidean':
        diffs = values[:, None, :] - values[None, :, :]
        return np.sqrt(np.sum(diffs * diffs, axis=2))

    if metric != 'correlation':
        raise ValueError(f'Unsupported distance metric: {metric}')

    centered = values - values.mean(axis=1, keepdims=True)
    norms = np.linalg.norm(centered, axis=1, keepdims=True)
    scaled = np.divide(centered, norms, out=np.zeros_like(centered), where=norms > 0)
    corr = np.clip(scaled @ scaled.T, -1.0, 1.0)
    return 1.0 - corr


def _score_profile_matrix(
    matrix: pd.DataFrame,
    group_name: str,
    profile_type: str,
    stat: str,
    distance_metric: str,
) -> dict[str, object] | None:
    if matrix.empty:
        return None

    matrix = keep_paired_profile_rows(matrix).sort_index()
    if matrix.empty:
        return None
    subjects = matrix.index.get_level_values('subject').astype(str).to_numpy()
    sessions = matrix.index.get_level_values('session').astype(str).to_numpy()
    values = matrix.to_numpy(dtype=float)

    if len(np.unique(subjects)) < 2 or len(np.unique(sessions)) < 2 or values.shape[1] < 2:
        return None

    distances = _pairwise_distances(values, distance_metric)
    scores: list[float] = []
    nearest_correct: list[float] = []
    genuine_distances: list[float] = []
    impostor_distances: list[float] = []
    rank_percentiles: list[float] = []

    for i, subject in enumerate(subjects):
        valid = np.arange(len(subjects)) != i
        same = (subjects == subject) & valid
        other = (subjects != subject) & valid
        if not same.any() or not other.any():
            continue

        genuine = distances[i, same]
        impostor = distances[i, other]
        genuine_min = float(np.min(genuine))

        scores.append(float(np.mean(impostor > genuine_min)))
        nearest_idx = np.argmin(np.where(valid, distances[i, :], np.inf))
        nearest_correct.append(float(subjects[nearest_idx] == subject))
        genuine_distances.append(genuine_min)
        impostor_distances.append(float(np.mean(impostor)))

        all_valid_distances = distances[i, valid]
        rank = 1 + int(np.sum(all_valid_distances < genuine_min))
        denom = max(len(all_valid_distances) - 1, 1)
        rank_percentiles.append(float(1.0 - (rank - 1) / denom))

    if not scores:
        return None

    return {
        'profile_type': profile_type,
        'profile_group': group_name,
        'stat': stat,
        'distance_metric': distance_metric,
        'discriminability': float(np.mean(scores)),
        'nearest_neighbor_accuracy': float(np.mean(nearest_correct)),
        'mean_genuine_distance': float(np.mean(genuine_distances)),
        'mean_impostor_distance': float(np.mean(impostor_distances)),
        'mean_rank_percentile': float(np.mean(rank_percentiles)),
        'n_subjects': int(len(np.unique(subjects))),
        'n_sessions': int(len(np.unique(sessions))),
        'n_profiles': int(len(subjects)),
        'n_features': int(values.shape[1]),
    }


def keep_paired_profile_rows(
    matrix: pd.DataFrame,
    min_sessions: int = 2,
) -> pd.DataFrame:
    if matrix.empty:
        return matrix
    subjects = matrix.index.get_level_values('subject').astype(str).to_numpy()
    sessions = matrix.index.get_level_values('session').astype(str).to_numpy()
    session_counts = pd.Series(sessions, index=subjects).groupby(level=0).nunique()
    paired_subjects = set(session_counts[session_counts >= min_sessions].index)
    keep_rows = [subject in paired_subjects for subject in subjects]
    return matrix.iloc[keep_rows, :]


def _build_profile_matrix(
    df: pd.DataFrame,
    feature_col: str,
    value_col: str,
    min_feature_coverage: float,
    min_profile_coverage: float,
    zscore_features: bool,
) -> pd.DataFrame:
    grouped = (
        df[['subject', 'session', feature_col, value_col]]
        .dropna(subset=[value_col])
        .groupby(['subject', 'session', feature_col], as_index=False)[value_col]
        .mean()
    )
    matrix = grouped.pivot_table(
        index=['subject', 'session'],
        columns=feature_col,
        values=value_col,
        aggfunc='mean',
    )
    if matrix.empty:
        return matrix

    matrix = keep_paired_profile_rows(matrix)
    if matrix.empty:
        return matrix

    feature_coverage = matrix.notna().mean(axis=0)
    matrix = matrix.loc[:, feature_coverage >= min_feature_coverage]
    profile_coverage = matrix.notna().mean(axis=1)
    matrix = matrix.loc[profile_coverage >= min_profile_coverage, :]
    if matrix.empty:
        return matrix

    matrix = keep_paired_profile_rows(matrix)
    if matrix.empty:
        return matrix

    matrix = matrix.dropna(axis=1, how='any')

    if matrix.empty:
        return matrix

    matrix = keep_paired_profile_rows(matrix)
    if matrix.empty:
        return matrix

    if zscore_features:
        matrix = _zscore_columns(matrix)
    return matrix


def _compute_discriminability(
    long_df: pd.DataFrame,
    profile_type: str,
    stat: str,
    min_feature_coverage: float,
    min_profile_coverage: float,
    distance_metric: str,
    zscore_features: bool,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    value_col = 'value'

    all_df = long_df.copy()
    all_df['metric_feature'] = all_df['metric'].astype(str) + '|' + all_df['feature'].astype(str)
    all_matrix = _build_profile_matrix(
        all_df,
        feature_col='metric_feature',
        value_col=value_col,
        min_feature_coverage=min_feature_coverage,
        min_profile_coverage=min_profile_coverage,
        zscore_features=zscore_features,
    )
    all_score = _score_profile_matrix(
        all_matrix,
        group_name='ALL_METRICS',
        profile_type=profile_type,
        stat=stat,
        distance_metric=distance_metric,
    )
    if all_score is not None:
        rows.append(all_score)

    for metric_name, metric_df in long_df.groupby('metric', sort=True):
        matrix = _build_profile_matrix(
            metric_df,
            feature_col='feature',
            value_col=value_col,
            min_feature_coverage=min_feature_coverage,
            min_profile_coverage=min_profile_coverage,
            zscore_features=zscore_features,
        )
        score = _score_profile_matrix(
            matrix,
            group_name=str(metric_name),
            profile_type=profile_type,
            stat=stat,
            distance_metric=distance_metric,
        )
        if score is not None:
            rows.append(score)

    if not rows:
        return pd.DataFrame(columns=RESULT_COLUMNS)
    return pd.DataFrame(rows, columns=RESULT_COLUMNS).sort_values(['profile_type', 'profile_group']).reset_index(drop=True)


def filter_analysis_set(
    df: pd.DataFrame,
    specs,
    analysis_set: str,
    tissue: str,
) -> pd.DataFrame:
    allowed = set(metric_order(specs, analysis_set, tissue=tissue))
    return df.loc[df['metric'].isin(allowed)].copy()


def write_metric_inclusion(
    out_file: Path,
    profile_type: str,
    analysis_set: str,
    qc_mode: str,
    tissue: str,
    expected_labels: list[str],
    observed_labels: set[str],
    scored_labels: set[str],
    display: dict[str, str],
) -> None:
    rows = []
    for label in expected_labels:
        observed = label in observed_labels
        scored = label in scored_labels
        rows.append(
            {
                'profile_type': profile_type,
                'analysis_set': analysis_set,
                'qc_mode': qc_mode,
                'tissue': tissue,
                'metric_key': label,
                'metric': display.get(label, label),
                'expected': True,
                'observed_after_qc': observed,
                'scored': scored,
                'reason_if_not_scored': (
                    ''
                    if scored
                    else (
                        'not_observed_after_qc'
                        if not observed
                        else 'insufficient_paired_profiles_or_feature_coverage'
                    )
                ),
            }
        )
    pd.DataFrame(rows).to_csv(out_file, sep='\t', index=False)


def load_wm_long_df(
    input_globs: list[str],
    stat: str,
    prefer_masked: bool,
    patterns_file: Path,
) -> pd.DataFrame:
    df = collect_scalarstats(input_globs, patterns_file=patterns_file)
    if df.empty:
        raise RuntimeError(f'No WM bundle scalarstats found for globs: {input_globs}')
    df = df.copy()
    df['subject'] = df['subject_id'].astype(str)
    df['session'] = df['session_id'].astype(str)
    df['feature'] = df['bundle'].astype(str)
    df['value'] = _value_column(df, stat=stat, prefer_masked=prefer_masked)
    return df[['subject', 'session', 'metric', 'feature', 'value']]


def load_dkt_long_df(
    input_globs: list[str],
    stat: str,
    patterns_file: Path,
) -> pd.DataFrame:
    row_tables = [collect_rows(input_glob) for input_glob in input_globs]
    row_tables = [table for table in row_tables if not table.empty]
    if not row_tables:
        raise RuntimeError(f'No DKT parcel stats found for glob(s): {input_globs}')
    rows = pd.concat(row_tables, ignore_index=True).drop_duplicates()
    value_df = build_value_table(rows, stat=stat, patterns_file=patterns_file)
    value_df = value_df.copy()
    value_df['feature'] = value_df['parcel'].astype(str)
    value_df = value_df.rename(columns={'subject': 'subject', 'session': 'session'})
    value_df = value_df[~value_df['metric'].isin(EXCLUDED_DKT_METRICS)].copy()
    value_df['value'] = pd.to_numeric(value_df['value'], errors='coerce')
    return value_df[['subject', 'session', 'metric', 'feature', 'value']]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--analysis',
        choices=('wm', 'dkt', 'both'),
        default='both',
        help='Which profile discriminability analysis to run.',
    )
    parser.add_argument(
        '--stat',
        choices=('mean', 'median'),
        default='median',
        help='Scalar statistic to use.',
    )
    parser.add_argument(
        '--prefer-masked',
        action='store_true',
        help='For WM bundle TSVs, prefer masked_mean/masked_median when available.',
    )
    parser.add_argument(
        '--distance-metric',
        choices=('correlation', 'euclidean'),
        default='correlation',
        help='Distance metric between subject-session profiles.',
    )
    parser.add_argument(
        '--no-zscore',
        action='store_true',
        help='Do not z-score features before computing distances.',
    )
    parser.add_argument(
        '--min-feature-coverage',
        type=float,
        default=0.0,
        help='Minimum fraction of profiles with finite data required for a feature.',
    )
    parser.add_argument(
        '--min-profile-coverage',
        type=float,
        default=0.0,
        help='Minimum fraction of retained features required for a subject-session profile.',
    )
    parser.add_argument(
        '--wm-input-globs',
        nargs='+',
        default=DEFAULT_WM_GLOBS,
        help='Input globs for WM bundle scalarstats TSVs.',
    )
    parser.add_argument(
        '--dkt-input-glob',
        nargs='+',
        default=DEFAULT_DKT_GLOBS,
        help=(
            'Input glob(s) or expanded file path(s) for DKT parcel stats CSVs. '
            'Quote shell globs to avoid expansion, or pass multiple files.'
        ),
    )
    parser.add_argument(
        '--qc-file',
        type=Path,
        default=DEFAULT_QC_FILE,
        help='Manual modality QC TSV.',
    )
    parser.add_argument(
        '--qc-mode',
        nargs='+',
        choices=QC_MODES,
        default=list(QC_MODES),
        help='QC-filtered discriminability versions to write.',
    )
    parser.add_argument(
        '--outdir',
        default=str(DERIVATIVES_ROOT / 'parcel_bundle_discriminability'),
        help='Output directory.',
    )
    parser.add_argument(
        '--patterns-file',
        type=Path,
        default=Path(__file__).resolve().parents[1] / 'configuration' / 'patterns.json',
        help='Metric pattern registry.',
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    zscore_features = not args.no_zscore
    qc_df = load_qc_table(args.qc_file)
    specs = build_metric_specs(args.patterns_file)
    source_by_metric = {spec.label: spec.source_image for spec in specs}

    if args.analysis in {'wm', 'both'}:
        wm_df = load_wm_long_df(
            input_globs=args.wm_input_globs,
            stat=args.stat,
            prefer_masked=args.prefer_masked,
            patterns_file=args.patterns_file,
        )
        wm_df = add_metric_metadata(wm_df, 'metric', args.patterns_file)
        suffix = f'{args.stat}_{args.distance_metric}'
        if args.prefer_masked:
            suffix = f'masked_preferred_{suffix}'
        for qc_mode in args.qc_mode:
            filtered_wm = apply_qc_mode(
                wm_df,
                qc_df,
                qc_mode,
                profile_type='wm',
                patterns_file=args.patterns_file,
            )
            for analysis_set in ANALYSIS_SETS:
                expected_labels = metric_order(specs, analysis_set, tissue='wm')
                observed_labels = set(filtered_wm['metric'])
                display = metric_display_labels(specs, analysis_set, tissue='wm')
                set_df = filter_analysis_set(
                    filtered_wm,
                    specs,
                    analysis_set,
                    tissue='wm',
                )
                wm_out = _compute_discriminability(
                    set_df,
                    profile_type='wm_bundles',
                    stat=args.stat,
                    min_feature_coverage=args.min_feature_coverage,
                    min_profile_coverage=args.min_profile_coverage,
                    distance_metric=args.distance_metric,
                    zscore_features=zscore_features,
                )
                if wm_out.empty:
                    write_metric_inclusion(
                        outdir
                        / f'discriminability_wm_bundles_{analysis_set}_{suffix}_metric_inclusion.tsv',
                        'wm_bundles',
                        analysis_set,
                        qc_mode,
                        'wm',
                        expected_labels,
                        observed_labels,
                        set(),
                        display,
                    )
                    continue
                wm_out.insert(0, 'analysis_set', analysis_set)
                wm_out.insert(1, 'qc_mode', qc_mode)
                wm_out.insert(2, 'profile_group_key', wm_out['profile_group'])
                wm_out['profile_group'] = (
                    wm_out['profile_group_key']
                    .map(display)
                    .fillna(wm_out['profile_group_key'])
                )
                wm_out['source_image'] = wm_out['profile_group_key'].map(source_by_metric).fillna('Other')
                out_csv = outdir / f'discriminability_wm_bundles_{analysis_set}_{suffix}.csv'
                wm_out.to_csv(out_csv, index=False)
                write_metric_inclusion(
                    outdir
                    / f'discriminability_wm_bundles_{analysis_set}_{suffix}_metric_inclusion.tsv',
                    'wm_bundles',
                    analysis_set,
                    qc_mode,
                    'wm',
                    expected_labels,
                    observed_labels,
                    set(wm_out['profile_group_key']),
                    display,
                )
                print(f'Wrote: {out_csv}', flush=True)

    if args.analysis in {'dkt', 'both'}:
        dkt_df = load_dkt_long_df(
            args.dkt_input_glob,
            stat=args.stat,
            patterns_file=args.patterns_file,
        )
        dkt_df = add_metric_metadata(dkt_df, 'metric', args.patterns_file)
        for qc_mode in args.qc_mode:
            filtered_dkt = apply_qc_mode(
                dkt_df,
                qc_df,
                qc_mode,
                profile_type='dkt',
                patterns_file=args.patterns_file,
            )
            for analysis_set in ANALYSIS_SETS:
                expected_labels = metric_order(specs, analysis_set, tissue='gm')
                observed_labels = set(filtered_dkt['metric'])
                display = metric_display_labels(specs, analysis_set, tissue='gm')
                set_df = filter_analysis_set(
                    filtered_dkt,
                    specs,
                    analysis_set,
                    tissue='gm',
                )
                dkt_out = _compute_discriminability(
                    set_df,
                    profile_type='DKTatlas_parcels',
                    stat=args.stat,
                    min_feature_coverage=args.min_feature_coverage,
                    min_profile_coverage=args.min_profile_coverage,
                    distance_metric=args.distance_metric,
                    zscore_features=zscore_features,
                )
                if dkt_out.empty:
                    write_metric_inclusion(
                        outdir
                        / (
                            f'discriminability_DKTatlas_{analysis_set}_{args.stat}_'
                            f'{args.distance_metric}_metric_inclusion.tsv'
                        ),
                        'DKTatlas_parcels',
                        analysis_set,
                        qc_mode,
                        'gm',
                        expected_labels,
                        observed_labels,
                        set(),
                        display,
                    )
                    continue
                dkt_out.insert(0, 'analysis_set', analysis_set)
                dkt_out.insert(1, 'qc_mode', qc_mode)
                dkt_out.insert(2, 'profile_group_key', dkt_out['profile_group'])
                dkt_out['profile_group'] = (
                    dkt_out['profile_group_key']
                    .map(display)
                    .fillna(dkt_out['profile_group_key'])
                )
                dkt_out['source_image'] = dkt_out['profile_group_key'].map(source_by_metric).fillna('Other')
                out_csv = (
                    outdir
                    / (
                        f'discriminability_DKTatlas_{analysis_set}_{args.stat}_'
                        f'{args.distance_metric}.csv'
                    )
                )
                dkt_out.to_csv(out_csv, index=False)
                write_metric_inclusion(
                    outdir
                    / (
                        f'discriminability_DKTatlas_{analysis_set}_{args.stat}_'
                        f'{args.distance_metric}_metric_inclusion.tsv'
                    ),
                    'DKTatlas_parcels',
                    analysis_set,
                    qc_mode,
                    'gm',
                    expected_labels,
                    observed_labels,
                    set(dkt_out['profile_group_key']),
                    display,
                )
                print(f'Wrote: {out_csv}', flush=True)


if __name__ == '__main__':
    main()
