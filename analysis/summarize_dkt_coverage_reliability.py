#!/usr/bin/env python3
"""Summarize DKT parcel coverage and reliability diagnostics."""

from __future__ import annotations

import argparse
import re
import sys
from glob import glob
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from metric_registry import build_metric_specs, metric_order
from path_utils import DERIVATIVES_ROOT


FILE_RE = re.compile(r'sub-(?P<subject>[^_]+)_(?P<session>ses-[^_]+)_(?P<run>run-[^_]+)_')
BASE_PARCEL_COLS = ('parcel_intensity', 'parcel_name', 'parcel_hemi')
DEFAULT_ROOT = DERIVATIVES_ROOT / 'DKTatlas_myelin_stats'


def normalize_subject(value: object) -> str:
    return re.sub(r'^sub-', '', str(value).strip())


def parse_filename(path: Path) -> tuple[str, str, str]:
    match = FILE_RE.search(path.name)
    if not match:
        raise ValueError(f'Could not parse subject/session/run from filename: {path}')
    return (
        normalize_subject(match.group('subject')),
        match.group('session'),
        match.group('run'),
    )


def collect_files(input_globs: list[str]) -> list[Path]:
    files: list[Path] = []
    for input_glob in input_globs:
        files.extend(Path(path) for path in glob(input_glob))
    return sorted(set(files))


def selected_metric_labels(args: argparse.Namespace) -> set[str] | None:
    if args.metric_regex:
        return None
    specs = build_metric_specs(args.patterns_file)
    return set(metric_order(specs, args.metric_set, tissue='gm'))


def metric_is_selected(
    value: object,
    metric_re: re.Pattern[str] | None,
    metric_labels: set[str] | None,
) -> bool:
    metric = str(value)
    if metric_re is not None:
        return bool(metric_re.search(metric))
    return metric in metric_labels if metric_labels is not None else True


def load_coverage(
    files: list[Path],
    metric_re: re.Pattern[str] | None,
    metric_labels: set[str] | None,
) -> pd.DataFrame:
    records = []
    for path in files:
        df = pd.read_csv(path)
        required = {
            'subject',
            'session',
            'run',
            'metric',
            'space',
            'parcel_intensity',
            'parcel_name',
            'parcel_hemi',
            'parcel_count',
            'valid_count',
            'coverage',
        }
        missing = required - set(df.columns)
        if missing:
            raise RuntimeError(f'{path} is missing coverage columns: {sorted(missing)}')
        df = df.loc[
            df['metric'].map(lambda value: metric_is_selected(value, metric_re, metric_labels))
        ].copy()
        if df.empty:
            continue
        df['subject'] = df['subject'].map(normalize_subject)
        df['source_coverage_file'] = str(path)
        records.append(df)
    if not records:
        return pd.DataFrame()
    out = pd.concat(records, ignore_index=True)
    out['parcel_intensity'] = pd.to_numeric(out['parcel_intensity'], errors='coerce').astype('Int64')
    for col in ('parcel_count', 'valid_count', 'coverage'):
        out[col] = pd.to_numeric(out[col], errors='coerce')
    out['parcel'] = out['parcel_hemi'].astype(str) + '_' + out['parcel_name'].astype(str)
    return out


def load_scalarstats(
    files: list[Path],
    metric_re: re.Pattern[str] | None,
    metric_labels: set[str] | None,
    stat: str,
) -> pd.DataFrame:
    records = []
    suffix = f'_{stat}'
    for path in files:
        subject, session, run = parse_filename(path)
        df = pd.read_csv(path)
        missing = set(BASE_PARCEL_COLS) - set(df.columns)
        if missing:
            raise RuntimeError(f'{path} is missing scalarstats columns: {sorted(missing)}')
        metric_cols = [
            col
            for col in df.columns
            if col.endswith(suffix)
            and metric_is_selected(col[: -len(suffix)], metric_re, metric_labels)
        ]
        if not metric_cols:
            continue
        value_df = df[list(BASE_PARCEL_COLS) + metric_cols].melt(
            id_vars=list(BASE_PARCEL_COLS),
            value_vars=metric_cols,
            var_name='metric_stat',
            value_name='value',
        )
        value_df['metric'] = value_df['metric_stat'].str[: -len(suffix)]
        value_df['subject'] = subject
        value_df['session'] = session
        value_df['run'] = run
        value_df['source_scalarstats_file'] = str(path)
        records.append(value_df.drop(columns=['metric_stat']))
    if not records:
        return pd.DataFrame()
    out = pd.concat(records, ignore_index=True)
    out['parcel_intensity'] = pd.to_numeric(out['parcel_intensity'], errors='coerce').astype('Int64')
    out['value'] = pd.to_numeric(out['value'], errors='coerce')
    out['parcel'] = out['parcel_hemi'].astype(str) + '_' + out['parcel_name'].astype(str)
    return out


def summarize_coverage(coverage: pd.DataFrame) -> pd.DataFrame:
    group_cols = ['metric', 'space', 'parcel_intensity', 'parcel_name', 'parcel_hemi', 'parcel']
    rows = []
    for key, dfg in coverage.groupby(group_cols, dropna=False, sort=True):
        values = dfg['coverage'].dropna().to_numpy(dtype=float)
        if values.size == 0:
            continue
        row = dict(zip(group_cols, key, strict=True))
        row.update(
            {
                'n_observations': int(len(dfg)),
                'n_subjects': int(dfg['subject'].nunique()),
                'n_sessions': int(dfg['session'].nunique()),
                'mean_parcel_count': float(dfg['parcel_count'].mean()),
                'mean_valid_count': float(dfg['valid_count'].mean()),
                'mean_coverage': float(np.mean(values)),
                'median_coverage': float(np.median(values)),
                'min_coverage': float(np.min(values)),
                'p05_coverage': float(np.quantile(values, 0.05)),
                'p25_coverage': float(np.quantile(values, 0.25)),
                'p75_coverage': float(np.quantile(values, 0.75)),
                'p95_coverage': float(np.quantile(values, 0.95)),
                'prop_coverage_eq_1': float(np.mean(values >= 0.999999)),
                'prop_coverage_lt_0_90': float(np.mean(values < 0.90)),
                'prop_coverage_lt_0_50': float(np.mean(values < 0.50)),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows).sort_values(['metric', 'space', 'parcel_intensity']).reset_index(drop=True)


def complete_case_matrix(dfg: pd.DataFrame) -> np.ndarray:
    matrix_df = dfg.pivot_table(index='subject', columns='session', values='value', aggfunc='mean')
    matrix = matrix_df.to_numpy(dtype=float)
    return matrix[~np.any(~np.isfinite(matrix), axis=1)]


def anova_mean_squares(matrix: np.ndarray) -> tuple[float, float, float]:
    if matrix.shape[0] < 2 or matrix.shape[1] < 2:
        return np.nan, np.nan, np.nan
    n_sub, n_ses = matrix.shape
    grand = matrix.mean()
    row_means = matrix.mean(axis=1)
    col_means = matrix.mean(axis=0)
    ssr = n_ses * np.sum((row_means - grand) ** 2)
    ssc = n_sub * np.sum((col_means - grand) ** 2)
    sse = np.sum((matrix - grand) ** 2) - ssr - ssc
    return (
        float(ssr / (n_sub - 1)),
        float(ssc / (n_ses - 1)),
        float(sse / ((n_sub - 1) * (n_ses - 1))),
    )


def icc2_1(matrix: np.ndarray) -> float:
    if matrix.shape[0] < 2 or matrix.shape[1] < 2:
        return np.nan
    n_sub, n_ses = matrix.shape
    msr, msc, mse = anova_mean_squares(matrix)
    denom = msr + (n_ses - 1) * mse + n_ses * (msc - mse) / n_sub
    if denom == 0:
        return np.nan
    return float((msr - mse) / denom)


def icc3_1(matrix: np.ndarray) -> float:
    if matrix.shape[0] < 2 or matrix.shape[1] < 2:
        return np.nan
    n_ses = matrix.shape[1]
    msr, _, mse = anova_mean_squares(matrix)
    denom = msr + (n_ses - 1) * mse
    if denom == 0:
        return np.nan
    return float((msr - mse) / denom)


def pairwise_pearson_r(matrix: np.ndarray) -> float:
    if matrix.shape[0] < 3 or matrix.shape[1] < 2:
        return np.nan
    values = []
    for i in range(matrix.shape[1]):
        for j in range(i + 1, matrix.shape[1]):
            x = matrix[:, i]
            y = matrix[:, j]
            if np.std(x) == 0 or np.std(y) == 0:
                continue
            values.append(float(np.corrcoef(x, y)[0, 1]))
    if not values:
        return np.nan
    z_values = np.arctanh(np.clip(values, -0.999999, 0.999999))
    return float(np.tanh(np.mean(z_values)))


def reliability_row(dfg: pd.DataFrame) -> dict[str, float | int]:
    matrix = complete_case_matrix(dfg)
    if matrix.shape[0] < 2 or matrix.shape[1] < 2:
        return {
            'ICC2_1': np.nan,
            'ICC3_1': np.nan,
            'pearson_r': np.nan,
            'between_subject_sd': np.nan,
            'within_subject_sd': np.nan,
            'within_between_sd_ratio': np.nan,
            'mean_abs_session_diff': np.nan,
            'n_complete_subjects': int(matrix.shape[0]),
            'n_complete_sessions': int(matrix.shape[1]),
        }
    _, _, mse = anova_mean_squares(matrix)
    between_subject_sd = float(np.std(matrix.mean(axis=1), ddof=1))
    within_subject_sd = float(np.sqrt(max(mse, 0.0))) if np.isfinite(mse) else np.nan
    abs_diffs = [
        np.abs(matrix[:, i] - matrix[:, j])
        for i in range(matrix.shape[1])
        for j in range(i + 1, matrix.shape[1])
    ]
    return {
        'ICC2_1': icc2_1(matrix),
        'ICC3_1': icc3_1(matrix),
        'pearson_r': pairwise_pearson_r(matrix),
        'between_subject_sd': between_subject_sd,
        'within_subject_sd': within_subject_sd,
        'within_between_sd_ratio': (
            float(within_subject_sd / between_subject_sd) if between_subject_sd > 0 else np.nan
        ),
        'mean_abs_session_diff': float(np.mean(np.concatenate(abs_diffs))),
        'n_complete_subjects': int(matrix.shape[0]),
        'n_complete_sessions': int(matrix.shape[1]),
    }


def summarize_reliability(observations: pd.DataFrame, thresholds: list[float]) -> pd.DataFrame:
    rows = []
    group_cols = ['metric', 'parcel_intensity', 'parcel_name', 'parcel_hemi', 'parcel']
    for threshold in thresholds:
        thresholded = observations.loc[
            (observations['coverage'] >= threshold) & np.isfinite(observations['value'])
        ].copy()
        for key, dfg in thresholded.groupby(group_cols, dropna=False, sort=True):
            row = dict(zip(group_cols, key, strict=True))
            row['coverage_threshold'] = float(threshold)
            row['n_observations'] = int(len(dfg))
            row['n_subjects_with_any_value'] = int(dfg['subject'].nunique())
            row.update(reliability_row(dfg))
            rows.append(row)
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(
        ['coverage_threshold', 'metric', 'parcel_intensity']
    ).reset_index(drop=True)


def summarize_metric_level(reliability: pd.DataFrame) -> pd.DataFrame:
    if reliability.empty:
        return pd.DataFrame()
    value_cols = [
        'ICC2_1',
        'ICC3_1',
        'pearson_r',
        'between_subject_sd',
        'within_subject_sd',
        'within_between_sd_ratio',
        'mean_abs_session_diff',
    ]
    rows = []
    for (threshold, metric), dfg in reliability.groupby(['coverage_threshold', 'metric'], sort=True):
        finite_icc = pd.to_numeric(dfg['ICC2_1'], errors='coerce').notna()
        row: dict[str, object] = {
            'coverage_threshold': threshold,
            'metric': metric,
            'n_parcels': int(len(dfg)),
            'n_parcels_with_icc': int(finite_icc.sum()),
            'median_complete_subjects': float(dfg['n_complete_subjects'].median()),
        }
        for col in value_cols:
            values = pd.to_numeric(dfg[col], errors='coerce')
            row[f'mean_{col}'] = float(values.mean()) if values.notna().any() else np.nan
            row[f'median_{col}'] = float(values.median()) if values.notna().any() else np.nan
        rows.append(row)
    return pd.DataFrame(rows).sort_values(['coverage_threshold', 'metric']).reset_index(drop=True)


def summarize_metric_rankings(metric_summary: pd.DataFrame) -> pd.DataFrame:
    if metric_summary.empty:
        return pd.DataFrame()
    rank_df = metric_summary.copy()
    ranking_cols = {
        'rank_highest_median_within_between_sd_ratio': ('median_within_between_sd_ratio', False),
        'rank_lowest_median_ICC2_1': ('median_ICC2_1', True),
        'rank_lowest_median_pearson_r': ('median_pearson_r', True),
        'rank_lowest_median_between_subject_sd': ('median_between_subject_sd', True),
        'rank_highest_median_within_subject_sd': ('median_within_subject_sd', False),
    }
    for rank_col, (value_col, ascending) in ranking_cols.items():
        rank_df[rank_col] = (
            rank_df.groupby('coverage_threshold')[value_col]
            .rank(method='min', ascending=ascending, na_option='bottom')
            .astype('Int64')
        )
    rank_df['likely_low_between_subject_contrast'] = (
        rank_df['median_within_between_sd_ratio'] >= 1.0
    )
    sort_cols = ['coverage_threshold', 'rank_highest_median_within_between_sd_ratio', 'metric']
    return rank_df.sort_values(sort_cols).reset_index(drop=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--coverage-glob',
        nargs='+',
        default=[str(DEFAULT_ROOT / 'sub-*' / 'sub-*_ses-*_run-*_desc-DKTatlas_coverage.csv')],
        help='One or more globs matching DKT coverage CSV files.',
    )
    parser.add_argument(
        '--scalarstats-glob',
        nargs='+',
        default=[str(DEFAULT_ROOT / 'sub-*' / 'sub-*_ses-*_run-*_desc-DKTatlas_scalarstats.csv')],
        help='One or more globs matching DKT scalarstats CSV files.',
    )
    parser.add_argument(
        '--metric-regex',
        default=None,
        help='Case-insensitive regex selecting metric names. Overrides --metric-set when supplied.',
    )
    parser.add_argument(
        '--metric-set',
        choices=('primary', 'full'),
        default='primary',
        help='Registry metric set to summarize when --metric-regex is not supplied.',
    )
    parser.add_argument(
        '--patterns-file',
        type=Path,
        default=Path(__file__).resolve().parents[1] / 'configuration' / 'patterns.json',
        help='Metric pattern registry used for --metric-set selection.',
    )
    parser.add_argument(
        '--output-prefix',
        default=None,
        help='Prefix for output filenames. Defaults to dkt_<metric-set> or dkt_regex_<metric-regex>.',
    )
    parser.add_argument('--stat', choices=('mean', 'median'), default='median')
    parser.add_argument(
        '--coverage-thresholds',
        nargs='+',
        type=float,
        default=[0.0, 0.5, 0.75, 0.9, 0.99],
        help='Coverage thresholds used for reliability sensitivity summaries.',
    )
    parser.add_argument(
        '--outdir',
        type=Path,
        default=DERIVATIVES_ROOT / 'DKTatlas_myelin_stats' / 'dkt_coverage_reliability_summary',
        help='Output directory for summary CSVs.',
    )
    return parser


def output_prefix(args: argparse.Namespace) -> str:
    if args.output_prefix:
        return args.output_prefix
    if args.metric_regex:
        clean = re.sub(r'[^A-Za-z0-9]+', '_', args.metric_regex).strip('_').lower()
        return f'dkt_regex_{clean}'
    return f'dkt_{args.metric_set}'


def main() -> None:
    args = build_parser().parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    metric_re = re.compile(args.metric_regex, flags=re.IGNORECASE) if args.metric_regex else None
    metric_labels = selected_metric_labels(args)
    prefix = output_prefix(args)

    coverage_files = collect_files(args.coverage_glob)
    scalarstats_files = collect_files(args.scalarstats_glob)
    if not coverage_files:
        raise FileNotFoundError(f'No coverage files matched: {args.coverage_glob}')
    if not scalarstats_files:
        raise FileNotFoundError(f'No scalarstats files matched: {args.scalarstats_glob}')

    coverage = load_coverage(coverage_files, metric_re, metric_labels)
    scalarstats = load_scalarstats(scalarstats_files, metric_re, metric_labels, args.stat)
    if coverage.empty:
        raise RuntimeError('No coverage rows matched the requested metrics.')
    if scalarstats.empty:
        raise RuntimeError(f'No scalarstats columns matched the requested metrics/stat: {args.stat}')

    merge_cols = ['subject', 'session', 'run', 'metric', 'parcel_intensity', 'parcel_name', 'parcel_hemi']
    observations = scalarstats.merge(
        coverage.drop(columns=['parcel'], errors='ignore'),
        on=merge_cols,
        how='left',
        validate='many_to_one',
    )
    observations['has_coverage_row'] = observations['coverage'].notna()
    observations['parcel'] = observations['parcel_hemi'].astype(str) + '_' + observations['parcel_name'].astype(str)

    coverage_summary = summarize_coverage(coverage)
    reliability_summary = summarize_reliability(observations, sorted(set(args.coverage_thresholds)))
    metric_summary = summarize_metric_level(reliability_summary)
    metric_rankings = summarize_metric_rankings(metric_summary)

    observations_file = args.outdir / f'{prefix}_{args.stat}_parcel_observations.csv'
    coverage_file = args.outdir / f'{prefix}_coverage_by_parcel.csv'
    reliability_file = args.outdir / f'{prefix}_{args.stat}_reliability_by_parcel.csv'
    metric_file = args.outdir / f'{prefix}_{args.stat}_metric_summary.csv'
    rankings_file = args.outdir / f'{prefix}_{args.stat}_metric_rankings.csv'

    observations.to_csv(observations_file, index=False)
    coverage_summary.to_csv(coverage_file, index=False)
    reliability_summary.to_csv(reliability_file, index=False)
    metric_summary.to_csv(metric_file, index=False)
    metric_rankings.to_csv(rankings_file, index=False)

    print(f'Coverage files read: {len(coverage_files)}', flush=True)
    print(f'Scalarstats files read: {len(scalarstats_files)}', flush=True)
    print(f'Metric selector: {args.metric_regex if args.metric_regex else args.metric_set}', flush=True)
    print(f'Matched metrics: {", ".join(sorted(observations["metric"].dropna().unique()))}', flush=True)
    print(f'Wrote: {observations_file}', flush=True)
    print(f'Wrote: {coverage_file}', flush=True)
    print(f'Wrote: {reliability_file}', flush=True)
    print(f'Wrote: {metric_file}', flush=True)
    print(f'Wrote: {rankings_file}', flush=True)


if __name__ == '__main__':
    main()
