#!/usr/bin/env python3
"""Compute ICC for GM parcels and WM bundles with registry metric labels."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sys.path.insert(0, str(Path(__file__).resolve().parent))

from compute_parcel_bundle_discriminability import (
    DEFAULT_DKT_GLOBS,
    DEFAULT_QC_FILE,
    DEFAULT_WM_GLOBS,
    QC_MODES,
    apply_qc_mode,
    load_dkt_long_df,
    load_qc_table,
    load_wm_long_df,
)
from metric_registry import SOURCE_IMAGE_COLORS, build_metric_specs, metric_display_labels, metric_order
from parcel_metric_utils import add_metric_metadata


ANALYSIS_SETS = ('primary', 'full')


def compute_icc2(values: np.ndarray, subjects: np.ndarray, sessions: np.ndarray) -> float:
    sub_unique, sub_idx = np.unique(subjects, return_inverse=True)
    ses_unique, ses_idx = np.unique(sessions, return_inverse=True)
    matrix = np.full((len(sub_unique), len(ses_unique)), np.nan, dtype=float)
    for value, i_sub, i_ses in zip(values, sub_idx, ses_idx):
        matrix[i_sub, i_ses] = value
    matrix = matrix[~np.any(np.isnan(matrix), axis=1)]
    if matrix.shape[0] < 2 or matrix.shape[1] < 2:
        return np.nan
    n_sub, n_ses = matrix.shape
    grand = matrix.mean()
    row_means = matrix.mean(axis=1)
    col_means = matrix.mean(axis=0)
    ssr = n_ses * np.sum((row_means - grand) ** 2)
    ssc = n_sub * np.sum((col_means - grand) ** 2)
    sse = np.sum((matrix - grand) ** 2) - ssr - ssc
    msr = ssr / (n_sub - 1)
    msc = ssc / (n_ses - 1)
    mse = sse / ((n_sub - 1) * (n_ses - 1))
    denom = msr + (n_ses - 1) * mse + n_ses * (msc - mse) / n_sub
    if denom == 0:
        return np.nan
    return float((msr - mse) / denom)


def compute_icc_table(long_df: pd.DataFrame, profile_type: str, stat: str) -> pd.DataFrame:
    collapsed = (
        long_df[['subject', 'session', 'metric', 'feature', 'value']]
        .dropna(subset=['value'])
        .groupby(['subject', 'session', 'metric', 'feature'], as_index=False)['value']
        .mean()
    )
    rows = []
    for (metric, feature), dfg in collapsed.groupby(['metric', 'feature'], sort=True):
        finite = dfg[np.isfinite(dfg['value'].to_numpy(float))].copy()
        paired_counts = finite.groupby('subject')['session'].nunique()
        paired_subjects = paired_counts[paired_counts >= 2].index
        paired = finite.loc[finite['subject'].isin(paired_subjects)].copy()
        if paired['subject'].nunique() < 2 or paired['session'].nunique() < 2:
            continue
        rows.append(
            {
                'profile_type': profile_type,
                'metric': metric,
                'feature': feature,
                'stat': stat,
                'ICC2_1': compute_icc2(
                    paired['value'].to_numpy(float),
                    paired['subject'].astype(str).to_numpy(),
                    paired['session'].astype(str).to_numpy(),
                ),
                'n_subjects': int(paired['subject'].nunique()),
                'n_sessions': int(paired['session'].nunique()),
                'n_observations': int(len(paired)),
            }
        )
    return pd.DataFrame(rows).sort_values(['profile_type', 'metric', 'feature']).reset_index(drop=True)


def plot_icc_violins(
    icc_df: pd.DataFrame,
    out_stem: Path,
    title: str,
    source_by_metric: dict[str, str],
) -> None:
    plot_df = icc_df[np.isfinite(icc_df['ICC2_1'].to_numpy(float))].copy()
    if plot_df.empty:
        return
    summary = (
        plot_df.groupby('metric')['ICC2_1']
        .agg(median='median', q25=lambda x: np.percentile(x, 25), q75=lambda x: np.percentile(x, 75))
        .sort_values('median', ascending=False)
    )
    order = list(summary.index)
    palette = {
        metric: SOURCE_IMAGE_COLORS.get(source_by_metric.get(metric, 'Other'), SOURCE_IMAGE_COLORS['Other'])
        for metric in order
    }
    height = max(5.5, 0.34 * len(order))
    fig, ax = plt.subplots(figsize=(10, height))
    sns.violinplot(
        data=plot_df,
        y='metric',
        x='ICC2_1',
        order=order,
        palette=palette,
        inner=None,
        cut=0,
        linewidth=0.8,
        ax=ax,
    )
    sns.stripplot(
        data=plot_df,
        y='metric',
        x='ICC2_1',
        order=order,
        color='black',
        size=1.8,
        alpha=0.35,
        ax=ax,
    )
    for y_pos, metric in enumerate(order):
        row = summary.loc[metric]
        ax.scatter(row['median'], y_pos, s=22, color='white', edgecolor='black', zorder=5)
        ax.text(
            1.02,
            y_pos,
            f"{row['median']:.2f} [{row['q25']:.2f}, {row['q75']:.2f}]",
            va='center',
            ha='left',
            fontsize=7,
            transform=ax.get_yaxis_transform(),
        )
    ax.set_xlim(-1, 1)
    ax.set_xlabel('ICC(2,1)')
    ax.set_ylabel('')
    ax.set_title(title)
    ax.axvline(0, color='black', linewidth=0.7, alpha=0.5)
    fig.subplots_adjust(right=0.78)
    for ext in ('png', 'pdf'):
        fig.savefig(out_stem.with_suffix(f'.{ext}'), dpi=300, bbox_inches='tight')
    plt.close(fig)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--analysis', choices=('wm', 'gm', 'both'), default='both')
    parser.add_argument('--stat', choices=('mean', 'median'), default='median')
    parser.add_argument('--prefer-masked', action='store_true')
    parser.add_argument('--qc-mode', nargs='+', choices=QC_MODES, default=list(QC_MODES))
    parser.add_argument('--wm-input-globs', nargs='+', default=DEFAULT_WM_GLOBS)
    parser.add_argument('--dkt-input-glob', nargs='+', default=DEFAULT_DKT_GLOBS)
    parser.add_argument('--qc-file', type=Path, default=DEFAULT_QC_FILE)
    parser.add_argument('--patterns-file', type=Path, default=Path(__file__).resolve().parents[1] / 'configuration' / 'patterns.json')
    parser.add_argument('--outdir', type=Path, default=Path('/cbica/projects/nibs/derivatives/parcel_bundle_icc'))
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    specs = build_metric_specs(args.patterns_file)
    source_by_metric = {spec.label: spec.source_image for spec in specs}
    qc_df = load_qc_table(args.qc_file)

    inputs = []
    if args.analysis in {'wm', 'both'}:
        wm_df = load_wm_long_df(
            args.wm_input_globs,
            stat=args.stat,
            prefer_masked=args.prefer_masked,
            patterns_file=args.patterns_file,
        )
        wm_df = add_metric_metadata(wm_df, 'metric', args.patterns_file)
        inputs.append(('wm_bundles', 'wm', wm_df))
    if args.analysis in {'gm', 'both'}:
        gm_df = load_dkt_long_df(
            args.dkt_input_glob,
            stat=args.stat,
            patterns_file=args.patterns_file,
        )
        gm_df = add_metric_metadata(gm_df, 'metric', args.patterns_file)
        inputs.append(('gm_parcels', 'dkt', gm_df))

    for profile_name, qc_profile, input_df in inputs:
        for qc_mode in args.qc_mode:
            filtered = apply_qc_mode(
                input_df,
                qc_df,
                qc_mode,
                profile_type=qc_profile,
                patterns_file=args.patterns_file,
            )
            for analysis_set in ANALYSIS_SETS:
                tissue = 'wm' if profile_name == 'wm_bundles' else 'gm'
                labels = set(
                    metric_order(
                        specs,
                        analysis_set,
                        tissue=tissue,
                    )
                )
                set_df = filtered.loc[filtered['metric'].isin(labels)].copy()
                if set_df.empty:
                    continue
                icc_df = compute_icc_table(set_df, profile_name, args.stat)
                if icc_df.empty:
                    continue
                display = metric_display_labels(specs, analysis_set, tissue=tissue)
                icc_df.insert(0, 'analysis_set', analysis_set)
                icc_df.insert(1, 'qc_mode', qc_mode)
                icc_df.insert(2, 'metric_key', icc_df['metric'])
                icc_df['metric'] = icc_df['metric_key'].map(display).fillna(icc_df['metric_key'])
                icc_df['source_image'] = icc_df['metric_key'].map(source_by_metric).fillna('Other')
                source_by_display = {
                    display.get(label, label): source_by_metric.get(label, 'Other')
                    for label in labels
                }
                stem = args.outdir / f'icc_{profile_name}_{analysis_set}_{args.stat}_{qc_mode}'
                csv_path = stem.with_name(stem.name + '.csv')
                icc_df.to_csv(csv_path, index=False)
                plot_icc_violins(
                    icc_df,
                    stem.with_name(stem.name + '_violins'),
                    f'{analysis_set.title()} {profile_name.replace("_", " ").title()} ICC',
                    source_by_display,
                )
                print(f'Wrote: {csv_path}', flush=True)


if __name__ == '__main__':
    main()
