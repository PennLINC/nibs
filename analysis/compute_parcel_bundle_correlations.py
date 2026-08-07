#!/usr/bin/env python3
"""Compute metric correlation matrices from WM bundle and GM parcel profiles."""

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
from matplotlib.patches import Patch, Rectangle
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import squareform
from scipy.stats import rankdata

sys.path.insert(0, str(Path(__file__).resolve().parent))

from compute_parcel_bundle_discriminability import load_dkt_long_df, load_qc_table, load_wm_long_df
from compute_parcel_bundle_discriminability import apply_qc_mode
from metric_registry import SOURCE_IMAGE_COLORS, build_metric_specs, metric_display_labels, metric_order
from parcel_metric_utils import add_metric_metadata, safe_label


DEFAULT_WM_GLOBS = [
    '/cbica/projects/nibs/derivatives/qsirecon/derivatives/qsirecon-*/sub-*/ses-*/dwi/sub-*_ses-*_*_scalarstats.tsv',
    '/cbica/projects/nibs/derivatives/bundle_myelin_stats/sub-*/ses-*/dwi/sub-*_ses-*_acq-HBCD75_run-01_space-T1w_model-*_scalarstats.tsv',
]
DEFAULT_DKT_GLOBS = [
    '/cbica/projects/nibs/derivatives/DKTatlas_myelin_stats/sub-*/sub-*_ses-*_run-*_desc-DKTatlas_scalarstats.csv'
]
QC_MODES = ('metricqc', 'completeqc')
ANALYSIS_SETS = ('primary', 'full')
PROFILE_TYPES = ('wm_bundles', 'gm_parcels')


def pairwise_profile_correlation(profile: pd.DataFrame, min_features: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    metrics = list(profile.columns)
    corr = np.full((len(metrics), len(metrics)), np.nan, dtype=float)
    counts = np.zeros((len(metrics), len(metrics)), dtype=np.int64)
    values = profile.to_numpy(dtype=float)
    for i in range(len(metrics)):
        corr[i, i] = 1.0
        counts[i, i] = int(np.count_nonzero(np.isfinite(values[:, i])))
        for j in range(i + 1, len(metrics)):
            valid = np.isfinite(values[:, i]) & np.isfinite(values[:, j])
            n_valid = int(np.count_nonzero(valid))
            counts[i, j] = n_valid
            counts[j, i] = n_valid
            if n_valid < min_features:
                continue
            x = rankdata(values[valid, i])
            y = rankdata(values[valid, j])
            if np.std(x) == 0 or np.std(y) == 0:
                continue
            value = float(np.corrcoef(x, y)[0, 1])
            corr[i, j] = value
            corr[j, i] = value
    return (
        pd.DataFrame(corr, index=metrics, columns=metrics),
        pd.DataFrame(counts, index=metrics, columns=metrics),
    )


def mean_correlation_matrix(long_df: pd.DataFrame, labels: list[str], min_features: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    counts = []
    for _, dfg in long_df[long_df['metric'].isin(labels)].groupby(['subject', 'session'], sort=True):
        profile = dfg.pivot_table(index='feature', columns='metric', values='value', aggfunc='mean')
        profile = profile.reindex(columns=labels)
        profile = profile.dropna(axis=1, how='all')
        if profile.shape[1] < 2:
            continue
        corr, count = pairwise_profile_correlation(profile, min_features)
        z = np.arctanh(np.clip(corr, -0.999999, 0.999999))
        np.fill_diagonal(z.values, 0.0)
        rows.append(z.reindex(index=labels, columns=labels))
        counts.append(count.reindex(index=labels, columns=labels))
    if not rows:
        raise RuntimeError('No valid subject/session correlation profiles were available.')
    mean_z = pd.DataFrame(np.nanmean(np.stack([row.to_numpy(float) for row in rows]), axis=0), index=labels, columns=labels)
    mean_r = np.tanh(mean_z)
    np.fill_diagonal(mean_r.values, 1.0)
    mean_counts = pd.DataFrame(np.nanmean(np.stack([row.to_numpy(float) for row in counts]), axis=0), index=labels, columns=labels)
    return mean_r, mean_counts


def correlation_linkage(corr: pd.DataFrame):
    if len(corr) < 2:
        return None
    safe = corr.fillna(0.0).to_numpy(float)
    distance = np.clip(1.0 - np.abs(safe), 0.0, 1.0)
    distance = (distance + distance.T) / 2.0
    np.fill_diagonal(distance, 0.0)
    return linkage(squareform(distance, checks=False), method='average', optimal_ordering=True)


def plot_matrix(corr: pd.DataFrame, out_stem: Path, title: str, source_by_metric: dict[str, str]) -> None:
    plot_data = corr.copy()
    np.fill_diagonal(plot_data.values, np.nan)
    z_matrix = correlation_linkage(corr)
    row_colors = pd.Series(
        {label: SOURCE_IMAGE_COLORS[source_by_metric.get(label, 'Other')] for label in corr.index},
        name='Source image',
    )
    cmap = sns.diverging_palette(220, 20, as_cmap=True)
    cmap.set_bad('#e6e6e6')
    grid = sns.clustermap(
        plot_data,
        row_linkage=z_matrix,
        col_linkage=z_matrix,
        row_cluster=z_matrix is not None,
        col_cluster=z_matrix is not None,
        row_colors=row_colors,
        cmap=cmap,
        vmin=-1,
        vmax=1,
        center=0,
        figsize=(12, 11.5),
        dendrogram_ratio=(0.12, 0.025),
        colors_ratio=0.025,
        cbar_pos=(0.27, 0.055, 0.46, 0.022),
        cbar_kws={'orientation': 'horizontal', 'label': 'Mean Spearman rho'},
    )
    grid.ax_col_dendrogram.set_visible(False)
    grid.ax_heatmap.set_aspect('equal', adjustable='box')
    grid.ax_heatmap.tick_params(axis='both', length=0)
    plt.setp(grid.ax_heatmap.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor', fontsize=8)
    plt.setp(grid.ax_heatmap.get_yticklabels(), rotation=0, fontsize=8)
    row_order = list(grid.data2d.index)
    col_position = {label: idx for idx, label in enumerate(grid.data2d.columns)}
    for row_idx, label in enumerate(row_order):
        grid.ax_heatmap.add_patch(Rectangle((col_position[label], row_idx), 1, 1, facecolor='black', edgecolor='black', linewidth=0, zorder=5))
    observed = {source_by_metric.get(label, 'Other') for label in corr.index}
    handles = [
        Patch(facecolor=color, edgecolor='none', label=source)
        for source, color in SOURCE_IMAGE_COLORS.items()
        if source in observed
    ]
    grid.ax_heatmap.legend(handles=handles, title='Source image', loc='upper left', bbox_to_anchor=(1.18, 0.55), frameon=False, fontsize=8, title_fontsize=9)
    grid.fig.suptitle(title, fontsize=18, y=0.97)
    grid.fig.subplots_adjust(left=0.08, right=0.82, top=0.92, bottom=0.16)
    grid.cax.set_position([0.27, 0.055, 0.46, 0.022])
    for ext in ('png', 'pdf'):
        grid.fig.savefig(out_stem.with_suffix(f'.{ext}'), bbox_inches='tight', dpi=300)
    plt.close(grid.fig)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--patterns-file', type=Path, default=Path(__file__).resolve().parents[1] / 'configuration' / 'patterns.json')
    parser.add_argument('--qc-file', type=Path, default=Path(__file__).resolve().parents[1] / 'data' / 'manual_qc_modality.tsv')
    parser.add_argument('--outdir', type=Path, default=Path('/cbica/projects/nibs/derivatives/parcel_bundle_correlations'))
    parser.add_argument('--analysis', choices=('wm', 'gm', 'both'), default='both')
    parser.add_argument('--qc-mode', choices=QC_MODES, default='metricqc')
    parser.add_argument('--stat', choices=('mean', 'median'), default='median')
    parser.add_argument('--prefer-masked', action='store_true')
    parser.add_argument('--min-features', type=int, default=10)
    parser.add_argument('--wm-input-globs', nargs='+', default=DEFAULT_WM_GLOBS)
    parser.add_argument('--dkt-input-glob', nargs='+', default=DEFAULT_DKT_GLOBS)
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
        wm_df = apply_qc_mode(
            wm_df,
            qc_df,
            args.qc_mode,
            profile_type='wm',
            patterns_file=args.patterns_file,
        )
        inputs.append(('wm_bundles', wm_df))
    if args.analysis in {'gm', 'both'}:
        gm_df = load_dkt_long_df(
            args.dkt_input_glob,
            stat=args.stat,
            patterns_file=args.patterns_file,
        )
        gm_df = add_metric_metadata(gm_df, 'metric', args.patterns_file)
        gm_df = apply_qc_mode(
            gm_df,
            qc_df,
            args.qc_mode,
            profile_type='dkt',
            patterns_file=args.patterns_file,
        )
        inputs.append(('gm_parcels', gm_df))

    for profile_type, long_df in inputs:
        for analysis_set in ANALYSIS_SETS:
            tissue = 'wm' if profile_type == 'wm_bundles' else 'gm'
            labels = [
                label
                for label in metric_order(
                    specs,
                    analysis_set,
                    tissue=tissue,
                )
                if label in set(long_df['metric'])
            ]
            if len(labels) < 2:
                continue
            corr, counts = mean_correlation_matrix(long_df, labels, args.min_features)
            display = metric_display_labels(
                specs,
                analysis_set,
                tissue=tissue,
            )
            corr = corr.rename(index=display, columns=display)
            counts = counts.rename(index=display, columns=display)
            source_display = {display.get(label, label): source_by_metric.get(label, 'Other') for label in labels}
            stem = args.outdir / f'mean_{profile_type}_{analysis_set}_spearman_{args.stat}_{args.qc_mode}'
            corr.to_csv(stem.with_name(stem.name + '_r.tsv'), sep='\t')
            counts.to_csv(stem.with_name(stem.name + '_mean_pairwise_nfeatures.tsv'), sep='\t')
            plot_matrix(
                corr,
                stem,
                f'{analysis_set.title()} {profile_type.replace("_", " ").title()} Spearman Correlations',
                source_display,
            )
            print(f'Wrote: {stem}_r.tsv', flush=True)


if __name__ == '__main__':
    main()
