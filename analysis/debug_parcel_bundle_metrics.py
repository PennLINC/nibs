#!/usr/bin/env python3
"""Debug where parcel/bundle metrics drop out of downstream analyses."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from compute_parcel_bundle_discriminability import (
    DEFAULT_DKT_GLOBS,
    DEFAULT_QC_FILE,
    DEFAULT_WM_GLOBS,
    _qc_passes,
    load_dkt_long_df,
    load_qc_table,
    load_wm_long_df,
    metric_required_modalities,
)
from metric_registry import build_metric_specs, metric_display_labels, metric_order
from path_utils import DERIVATIVES_ROOT


DEFAULT_METRICS = ('ihMTR', 'R2', 'R2*-E5', 'R2*')


def canonical_debug_metrics(metrics: list[str], patterns_file: Path, tissue: str) -> list[str]:
    specs = build_metric_specs(patterns_file)
    display = metric_display_labels(specs, 'primary', tissue=tissue)
    reverse_display = {str(value): key for key, value in display.items()}
    by_primary = {spec.primary_label: spec.label for spec in specs}
    by_label = {spec.label: spec.label for spec in specs}
    out = []
    for metric in metrics:
        candidates = [
            by_label.get(metric),
            by_primary.get(metric),
            reverse_display.get(metric),
            metric,
        ]
        for candidate in candidates:
            if candidate and candidate not in out:
                out.append(candidate)
                break
    return out


def summarize_subset(df: pd.DataFrame, metric: str) -> dict[str, object]:
    subset = df.loc[df['metric'] == metric].copy()
    if subset.empty:
        return {
            'rows': 0,
            'finite_values': 0,
            'subjects': 0,
            'sessions': 0,
            'subject_sessions': 0,
            'features': 0,
        }
    finite = pd.to_numeric(subset['value'], errors='coerce')
    return {
        'rows': int(len(subset)),
        'finite_values': int(np.isfinite(finite.to_numpy(float)).sum()),
        'subjects': int(subset['subject'].astype(str).nunique()),
        'sessions': int(subset['session'].astype(str).nunique()),
        'subject_sessions': int(subset[['subject', 'session']].drop_duplicates().shape[0]),
        'features': int(subset['feature'].astype(str).nunique()),
    }


def qc_subset(df: pd.DataFrame, qc_df: pd.DataFrame, metric: str, patterns_file: Path) -> pd.DataFrame:
    subset = df.loc[df['metric'] == metric].copy()
    if subset.empty:
        return subset
    modalities = metric_required_modalities(metric, patterns_file)
    keep = [
        _qc_passes(qc_df, row['subject'], row['session'], modalities)
        for _, row in subset.iterrows()
    ]
    return subset.loc[keep].copy()


def read_matrix_labels(path: Path) -> set[str]:
    if not path.exists():
        return set()
    table = pd.read_csv(path, sep='\t', index_col=0, nrows=0)
    return set(map(str, table.columns))


def read_icc_metric_keys(path: Path) -> set[str]:
    if not path.exists():
        return set()
    table = pd.read_csv(path, usecols=['metric_key'])
    return set(table['metric_key'].astype(str))


def inclusion_rows(path: Path, metrics: list[str]) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    table = pd.read_csv(path, sep='\t')
    return table.loc[table['metric_key'].astype(str).isin(metrics)].copy()


def debug_profile(
    profile: str,
    df: pd.DataFrame,
    qc_df: pd.DataFrame,
    metrics: list[str],
    patterns_file: Path,
    stat: str,
) -> None:
    tissue = 'wm' if profile == 'wm_bundles' else 'gm'
    expected = set(metric_order(build_metric_specs(patterns_file), 'primary', tissue=tissue))
    display = metric_display_labels(build_metric_specs(patterns_file), 'primary', tissue=tissue)
    metrics = canonical_debug_metrics(metrics, patterns_file, tissue)
    corr_dir = DERIVATIVES_ROOT / 'parcel_bundle_correlations'
    icc_dir = DERIVATIVES_ROOT / 'parcel_bundle_icc'
    corr_matrix = corr_dir / f'{profile}_primary_spearman_{stat}_r.tsv'
    corr_inclusion = corr_dir / f'{profile}_primary_spearman_{stat}_metric_inclusion.tsv'
    icc_csv = icc_dir / f'icc_{profile}_primary_{stat}.csv'
    icc_inclusion = icc_dir / f'icc_{profile}_primary_{stat}_metric_inclusion.tsv'
    corr_labels = read_matrix_labels(corr_matrix)
    icc_labels = read_icc_metric_keys(icc_csv)

    print(f'\n## {profile} ({tissue})')
    print(f'Input rows: {len(df)}')
    print(f'Correlation matrix: {corr_matrix}')
    print(f'ICC table: {icc_csv}')
    for metric in metrics:
        before = summarize_subset(df, metric)
        after_df = qc_subset(df, qc_df, metric, patterns_file)
        after = summarize_subset(after_df, metric)
        try:
            modalities = ','.join(metric_required_modalities(metric, patterns_file))
        except ValueError as exc:
            modalities = f'ERROR: {exc}'
        print(f'\nmetric_key={metric}')
        print(f'  display_label={display.get(metric, metric)}')
        print(f'  expected_primary={metric in expected}')
        print(f'  required_qc_modalities={modalities}')
        print(
            '  before_qc: '
            + ', '.join(f'{key}={value}' for key, value in before.items())
        )
        print(
            '  after_metricqc: '
            + ', '.join(f'{key}={value}' for key, value in after.items())
        )
        print(f'  in_corr_matrix={display.get(metric, metric) in corr_labels or metric in corr_labels}')
        print(f'  in_icc_table={metric in icc_labels}')

    for label, path in (('corr_inclusion', corr_inclusion), ('icc_inclusion', icc_inclusion)):
        rows = inclusion_rows(path, metrics)
        print(f'\n{label}: {path}')
        if rows.empty:
            print('  no matching rows or file missing')
        else:
            print(rows.to_string(index=False))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--analysis', choices=('wm', 'gm', 'both'), default='both')
    parser.add_argument('--metrics', nargs='+', default=list(DEFAULT_METRICS))
    parser.add_argument('--stat', choices=('mean', 'median'), default='median')
    parser.add_argument('--prefer-masked', action='store_true')
    parser.add_argument('--patterns-file', type=Path, default=Path(__file__).resolve().parents[1] / 'configuration' / 'patterns.json')
    parser.add_argument('--qc-file', type=Path, default=DEFAULT_QC_FILE)
    parser.add_argument('--wm-input-globs', nargs='+', default=DEFAULT_WM_GLOBS)
    parser.add_argument('--dkt-input-glob', nargs='+', default=DEFAULT_DKT_GLOBS)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    qc_df = load_qc_table(args.qc_file)
    if args.analysis in {'wm', 'both'}:
        wm_df = load_wm_long_df(
            args.wm_input_globs,
            stat=args.stat,
            prefer_masked=args.prefer_masked,
            patterns_file=args.patterns_file,
        )
        debug_profile('wm_bundles', wm_df, qc_df, args.metrics, args.patterns_file, args.stat)
    if args.analysis in {'gm', 'both'}:
        gm_df = load_dkt_long_df(
            args.dkt_input_glob,
            stat=args.stat,
            patterns_file=args.patterns_file,
        )
        debug_profile('gm_parcels', gm_df, qc_df, args.metrics, args.patterns_file, args.stat)


if __name__ == '__main__':
    main()
