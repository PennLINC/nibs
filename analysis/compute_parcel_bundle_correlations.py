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


def write_metric_inclusion(
    out_file: Path,
    profile_type: str,
    analysis_set: str,
    tissue: str,
    expected_labels: list[str],
    observed_labels: set[str],
    plotted_labels: list[str],
    display: dict[str, str],
) -> None:
    plotted = set(plotted_labels)
    rows = [
        {
            'profile_type': profile_type,
            'analysis_set': analysis_set,
            'tissue': tissue,
            'metric_key': label,
            'metric': display.get(label, label),
            'expected': True,
            'observed_in_input_after_qc': label in observed_labels,
            'plotted': label in plotted,
            'reason_if_not_plotted': (
                ''
                if label in plotted
                else (
                    'not_observed_in_input_after_qc'
                    if label not in observed_labels
                    else 'fewer_than_two_plottable_metrics_or_no_valid_correlations'
                )
            ),
        }
        for label in expected_labels
    ]
    pd.DataFrame(rows).to_csv(out_file, sep='\t', index=False)


def source_display_for_labels(
    specs,
    analysis_set: str,
    tissue: str,
    matrix_labels,
) -> dict[str, str]:
    display = metric_display_labels(specs, analysis_set, tissue=tissue)
    source_by_display = {
        display.get(spec.label, spec.label): spec.source_image
        for spec in specs
    }
    for spec in specs:
        if spec.pattern_key == 'DKI Micro AWF':
            source_by_display['DKI Micro AWF'] = spec.source_image
    return {
        str(label): source_by_display.get(str(label), 'Other')
        for label in matrix_labels
    }


def read_matrix(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, sep='\t', index_col=0)


def plot_existing_matrix(
    stem: Path,
    profile_type: str,
    analysis_set: str,
    tissue: str,
    specs,
) -> bool:
    matrix_path = stem.with_name(stem.name + '_r.tsv')
    if not matrix_path.exists():
        print(f'[WARN] Missing existing matrix for plot-only mode: {matrix_path}', flush=True)
        return False
    corr = read_matrix(matrix_path)
    source_display = source_display_for_labels(
        specs,
        analysis_set,
        tissue,
        corr.index,
    )
    plot_matrix(
        corr,
        stem,
        f'{analysis_set.title()} {profile_type.replace("_", " ").title()} Spearman Correlations',
        source_display,
    )
    print(f'Replotted: {stem}.png/.pdf', flush=True)
    return True


def selected_profile_types(analysis: str) -> list[str]:
    if analysis == 'wm':
        return ['wm_bundles']
    if analysis == 'gm':
        return ['gm_parcels']
    return list(PROFILE_TYPES)


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


def mean_correlation_matrix(
    long_df: pd.DataFrame,
    labels: list[str],
    min_features: int,
    min_group_subjects: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    subject_z_mats = []
    subject_count_mats = []
    for subject, subject_df in long_df[long_df['metric'].isin(labels)].groupby('subject', sort=True):
        session_z_mats = []
        session_count_mats = []
        for _, dfg in subject_df.groupby('session', sort=True):
            profile = dfg.pivot_table(index='feature', columns='metric', values='value', aggfunc='mean')
            profile = profile.reindex(columns=labels)
            profile = profile.dropna(axis=1, how='all')
            if profile.shape[1] < 2:
                continue
            corr, count = pairwise_profile_correlation(profile, min_features)
            z = np.arctanh(np.clip(corr, -0.999999, 0.999999))
            np.fill_diagonal(z.values, 0.0)
            session_z_mats.append(z.reindex(index=labels, columns=labels))
            session_count_mats.append(count.reindex(index=labels, columns=labels))
        if not session_z_mats:
            continue
        z_stack = np.stack([mat.to_numpy(float) for mat in session_z_mats])
        count_stack = np.stack([mat.to_numpy(float) for mat in session_count_mats])
        subject_z_mats.append(
            pd.DataFrame(np.nanmean(z_stack, axis=0), index=labels, columns=labels)
        )
        subject_count_mats.append(
            pd.DataFrame(np.nanmean(count_stack, axis=0), index=labels, columns=labels)
        )
    if not subject_z_mats:
        empty = pd.DataFrame(np.nan, index=labels, columns=labels)
        return empty, empty.copy(), pd.DataFrame(0, index=labels, columns=labels)
    group_stack = np.stack([mat.to_numpy(float) for mat in subject_z_mats])
    nsubjects = pd.DataFrame(np.sum(np.isfinite(group_stack), axis=0), index=labels, columns=labels)
    mean_z = pd.DataFrame(np.nanmean(group_stack, axis=0), index=labels, columns=labels)
    mean_r = pd.DataFrame(np.tanh(mean_z), index=labels, columns=labels)
    low_n = nsubjects < int(min_group_subjects)
    mean_r = mean_r.mask(low_n)
    np.fill_diagonal(mean_r.values, 1.0)
    count_stack = np.stack([mat.to_numpy(float) for mat in subject_count_mats])
    mean_counts = pd.DataFrame(np.nanmean(count_stack, axis=0), index=labels, columns=labels)
    mean_counts = mean_counts.mask(low_n)
    return mean_r, mean_counts, nsubjects


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
    grid.fig.canvas.draw()
    heatmap_position = grid.ax_heatmap.get_position()
    color_position = grid.ax_row_colors.get_position()
    grid.ax_row_colors.set_position(
        [
            color_position.x0,
            heatmap_position.y0,
            color_position.width,
            heatmap_position.height,
        ]
    )
    grid.ax_row_colors.set_ylim(grid.ax_heatmap.get_ylim())
    dendrogram_position = grid.ax_row_dendrogram.get_position()
    grid.ax_row_dendrogram.set_position(
        [
            dendrogram_position.x0,
            heatmap_position.y0,
            dendrogram_position.width,
            heatmap_position.height,
        ]
    )
    grid.ax_row_dendrogram.set_ylim(grid.ax_heatmap.get_ylim())
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
    parser.add_argument('--min-group-subjects', type=int, default=1)
    parser.add_argument(
        '--plot-only',
        action='store_true',
        help='Regenerate PNG/PDF figures from existing *_r.tsv matrices without recomputing correlations.',
    )
    parser.add_argument(
        '--plot-stem',
        action='append',
        type=Path,
        default=None,
        help=(
            'Existing output stem to replot, without _r.tsv. '
            'May be repeated. If omitted with --plot-only, expected stems are derived from --analysis/--stat/--qc-mode.'
        ),
    )
    parser.add_argument('--wm-input-globs', nargs='+', default=DEFAULT_WM_GLOBS)
    parser.add_argument('--dkt-input-glob', nargs='+', default=DEFAULT_DKT_GLOBS)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    specs = build_metric_specs(args.patterns_file)
    source_by_metric = {spec.label: spec.source_image for spec in specs}

    if args.plot_only:
        plotted_any = False
        if args.plot_stem:
            for stem in args.plot_stem:
                stem = stem.expanduser()
                stem_name = stem.name
                profile_type = 'wm_bundles' if 'wm_bundles' in stem_name else 'gm_parcels'
                tissue = 'wm' if profile_type == 'wm_bundles' else 'gm'
                analysis_set = 'primary' if '_primary_' in stem_name else 'full'
                plotted_any = (
                    plot_existing_matrix(
                        stem,
                        profile_type,
                        analysis_set,
                        tissue,
                        specs,
                    )
                    or plotted_any
                )
        else:
            for profile_type in selected_profile_types(args.analysis):
                tissue = 'wm' if profile_type == 'wm_bundles' else 'gm'
                for analysis_set in ANALYSIS_SETS:
                    stem = args.outdir / f'mean_{profile_type}_{analysis_set}_spearman_{args.stat}_{args.qc_mode}'
                    plotted_any = (
                        plot_existing_matrix(
                            stem,
                            profile_type,
                            analysis_set,
                            tissue,
                            specs,
                        )
                        or plotted_any
                    )
        if not plotted_any:
            raise RuntimeError('No existing matrices were found to plot.')
        return

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
            expected_labels = metric_order(
                specs,
                analysis_set,
                tissue=tissue,
            )
            observed_labels = set(long_df['metric'])
            labels = [
                label
                for label in expected_labels
                if label in observed_labels
            ]
            display = metric_display_labels(
                specs,
                analysis_set,
                tissue=tissue,
            )
            stem = args.outdir / f'mean_{profile_type}_{analysis_set}_spearman_{args.stat}_{args.qc_mode}'
            write_metric_inclusion(
                stem.with_name(stem.name + '_metric_inclusion.tsv'),
                profile_type,
                analysis_set,
                tissue,
                expected_labels,
                observed_labels,
                labels,
                display,
            )
            if len(labels) < 2:
                continue
            corr, counts, nsubjects = mean_correlation_matrix(
                long_df,
                labels,
                args.min_features,
                args.min_group_subjects,
            )
            corr = corr.rename(index=display, columns=display)
            counts = counts.rename(index=display, columns=display)
            nsubjects = nsubjects.rename(index=display, columns=display)
            source_display = {display.get(label, label): source_by_metric.get(label, 'Other') for label in labels}
            corr.to_csv(stem.with_name(stem.name + '_r.tsv'), sep='\t')
            counts.to_csv(stem.with_name(stem.name + '_mean_pairwise_nfeatures.tsv'), sep='\t')
            nsubjects.to_csv(stem.with_name(stem.name + '_nsubjects.tsv'), sep='\t')
            plot_matrix(
                corr,
                stem,
                f'{analysis_set.title()} {profile_type.replace("_", " ").title()} Spearman Correlations',
                source_display,
            )
            print(f'Wrote: {stem}_r.tsv', flush=True)


if __name__ == '__main__':
    main()
