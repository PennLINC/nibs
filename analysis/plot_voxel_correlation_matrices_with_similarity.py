#!/usr/bin/env python3
"""Plot WM/GM voxelwise correlation matrices plus GM-vs-WM profile similarity."""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

try:
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import seaborn as sns
    from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
    from matplotlib.patches import Patch
except ImportError:  # pragma: no cover - checked after argparse handles --help
    mpl = None
    plt = None
    np = None
    pd = None
    sns = None
    GridSpec = None
    GridSpecFromSubplotSpec = None
    Patch = None

sys.path.insert(0, str(Path(__file__).resolve().parent))

from metric_registry import METRIC_FAMILY_LEGEND_TITLE, SOURCE_IMAGE_COLORS
from path_utils import CODE_ROOT, PROJECT_ROOT
from plot_combined_correlation_matrices import (
    PanelSpec,
    align_side_axes_to_heatmaps,
    draw_heatmap,
    draw_left_dendrogram,
    draw_source_bar,
    label_fontsize,
    observed_sources,
    ordered_matrix,
    require_dependencies,
    source_lookup,
    title_fontsize,
)
from plot_correlation_matrices import (
    load_correlation_matrix,
    mni_input_path,
    selected_values,
    source_display_label,
)


ANALYSIS_SETS = ('primary', 'full')
MNI_CORRELATIONS = ('pearson', 'spearman')
PROFILE_SIMILARITY_METHODS = ('pearson', 'spearman')


@dataclass(frozen=True)
class VoxelPanelData:
    panel: PanelSpec
    matrix: pd.DataFrame
    linkage: object | None


def default_mni_dir() -> Path:
    return PROJECT_ROOT / 'derivatives' / 'mni_voxelwise_correlations'


def figure_size(n_metrics: int) -> tuple[float, float]:
    panel_side = max(5.5, min(8.2, 1.35 + 0.24 * n_metrics))
    width = 2.0 * panel_side + 4.15
    height = panel_side + max(4.8, min(7.0, 1.8 + 0.17 * n_metrics))
    return width, height


def cbar_label(correlation: str) -> str:
    method = r'Pearson $\bf{\it r}$' if correlation == 'pearson' else 'Spearman ρ'
    return f'Mean voxelwise correlation ({method})'


def common_metric_labels(wm_corr: pd.DataFrame, gm_corr: pd.DataFrame) -> list[str]:
    return [
        label
        for label in wm_corr.index
        if label in wm_corr.columns and label in gm_corr.index and label in gm_corr.columns
    ]


def rank_values(values: np.ndarray) -> np.ndarray:
    series = pd.Series(values)
    return series.rank(method='average').to_numpy(dtype=float)


def correlation_value(x: np.ndarray, y: np.ndarray, method: str) -> float:
    valid = np.isfinite(x) & np.isfinite(y)
    if np.count_nonzero(valid) < 3:
        return np.nan
    x_valid = x[valid]
    y_valid = y[valid]
    if method == 'spearman':
        x_valid = rank_values(x_valid)
        y_valid = rank_values(y_valid)
    elif method != 'pearson':
        raise ValueError(f'Unsupported profile similarity method: {method}')
    if np.std(x_valid) == 0 or np.std(y_valid) == 0:
        return np.nan
    return float(np.corrcoef(x_valid, y_valid)[0, 1])


def profile_similarity_table(
    wm_corr: pd.DataFrame,
    gm_corr: pd.DataFrame,
    labels: list[str],
    method: str,
    source_by_label: dict[str, str],
) -> pd.DataFrame:
    rows = []
    for label in labels:
        others = [other for other in labels if other != label]
        wm_profile = wm_corr.loc[label, others].to_numpy(dtype=float)
        gm_profile = gm_corr.loc[label, others].to_numpy(dtype=float)
        valid = np.isfinite(wm_profile) & np.isfinite(gm_profile)
        rows.append(
            {
                'metric': label,
                'source_image': source_by_label.get(label, 'Other'),
                'profile_similarity': correlation_value(wm_profile, gm_profile, method),
                'n_profile_metrics': int(np.count_nonzero(valid)),
            }
        )
    return pd.DataFrame(rows)


def load_ordered_data(
    analysis_set: str,
    correlation: str,
    patterns_file: Path,
    mni_dir: Path,
    profile_similarity_method: str,
) -> tuple[list[VoxelPanelData], pd.DataFrame]:
    wm_panel = PanelSpec(
        tissue='wm',
        title='White Matter Voxels',
        path=mni_input_path(mni_dir, analysis_set, 'wm', correlation),
        source_by_label=source_lookup(patterns_file, analysis_set, 'wm'),
    )
    gm_panel = PanelSpec(
        tissue='gm',
        title='Cortical Gray Matter Voxels',
        path=mni_input_path(mni_dir, analysis_set, 'gm', correlation),
        source_by_label=source_lookup(patterns_file, analysis_set, 'gm'),
    )
    wm_corr = load_correlation_matrix(wm_panel.path)
    gm_corr = load_correlation_matrix(gm_panel.path)
    labels = common_metric_labels(wm_corr, gm_corr)
    if len(labels) < 3:
        raise RuntimeError('Need at least 3 shared WM/GM metrics for profile similarity.')
    wm_common = wm_corr.loc[labels, labels].copy()
    gm_common = gm_corr.loc[labels, labels].copy()
    wm_ordered, wm_linkage = ordered_matrix(wm_common)
    gm_ordered, gm_linkage = ordered_matrix(gm_common)
    similarity = profile_similarity_table(
        wm_common,
        gm_common,
        labels,
        profile_similarity_method,
        wm_panel.source_by_label,
    )
    return (
        [
            VoxelPanelData(wm_panel, wm_ordered, wm_linkage),
            VoxelPanelData(gm_panel, gm_ordered, gm_linkage),
        ],
        similarity,
    )


def draw_similarity_panel(
    ax,
    similarity: pd.DataFrame,
    label_size: float,
    method: str,
) -> None:
    plot_df = similarity.sort_values(
        ['profile_similarity', 'metric'],
        ascending=[False, True],
        na_position='first',
    ).copy()
    y = np.arange(len(plot_df))
    colors = [
        SOURCE_IMAGE_COLORS.get(source, SOURCE_IMAGE_COLORS['Other'])
        for source in plot_df['source_image']
    ]
    values = plot_df['profile_similarity'].to_numpy(dtype=float)
    ax.axvline(0.0, color='#bbbbbb', lw=0.9, zorder=0)
    for benchmark in (-0.5, 0.5):
        ax.axvline(benchmark, color='#e4e4e4', lw=0.8, zorder=0)
    ax.barh(y, values, height=0.66, color=colors, edgecolor='#2b2b2b', linewidth=0.45, alpha=0.90)
    ax.set_yticks(y)
    ax.set_yticklabels(plot_df['metric'].tolist(), fontsize=label_size)
    ax.tick_params(axis='y', length=0, pad=4)
    ax.tick_params(axis='x', labelsize=max(9.5, label_size - 0.4), length=3)
    ax.set_xlim(-1.0, 1.0)
    ax.set_xlabel(f'GM-WM correlation-profile similarity ({method})', fontweight='bold')
    ax.set_title(
        'Metric-Specific GM vs WM Structure',
        loc='left',
        fontsize=title_fontsize(len(plot_df)),
        fontweight='bold',
        pad=7,
    )
    ax.text(
        -0.075,
        1.03,
        'C',
        transform=ax.transAxes,
        ha='left',
        va='bottom',
        fontsize=title_fontsize(len(plot_df)) + 1.5,
        fontweight='bold',
        clip_on=False,
    )
    ax.set_ylim(-0.7, len(plot_df) - 0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


def position_guides(
    fig: plt.Figure,
    cbar_ax: plt.Axes,
    legend_ax: plt.Axes,
    heatmap_axes: list[plt.Axes],
) -> None:
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    tick_boxes = [
        tick.get_window_extent(renderer)
        for ax in heatmap_axes
        for tick in ax.get_xticklabels()
        if tick.get_visible() and tick.get_text()
    ]
    if tick_boxes:
        label_bottom_display = min(box.y0 for box in tick_boxes)
        label_bottom = fig.transFigure.inverted().transform((0, label_bottom_display))[1]
    else:
        label_bottom = min(ax.get_position().y0 for ax in heatmap_axes)
    cbar_y0 = max(0.075, label_bottom - 0.036)
    cbar_ax.set_position([0.24, cbar_y0, 0.28, 0.014])
    legend_ax.set_position([0.57, cbar_y0 - 0.018, 0.36, 0.055])


def draw_figure(
    panels: list[VoxelPanelData],
    similarity: pd.DataFrame,
    analysis_set: str,
    correlation: str,
    profile_similarity_method: str,
    out_stem: Path,
) -> None:
    matrices = [panel.matrix for panel in panels]
    max_metrics = max(matrix.shape[0] for matrix in matrices)
    fs = max(9.4, label_fontsize(max_metrics) - 0.7)

    cmap = sns.diverging_palette(220, 20, as_cmap=True)
    cmap.set_bad('#eeeeee')

    fig_width, fig_height = figure_size(max_metrics)
    fig = plt.figure(figsize=(fig_width, fig_height), constrained_layout=False)
    outer = GridSpec(
        3,
        2,
        figure=fig,
        height_ratios=[1.0, 0.07, 0.86],
        width_ratios=[1.0, 1.0],
        hspace=0.55,
        wspace=0.12,
    )

    image = None
    panel_axes: list[tuple[plt.Axes, plt.Axes, plt.Axes]] = []
    for panel_index, panel_data in enumerate(panels):
        inner = GridSpecFromSubplotSpec(
            1,
            3,
            subplot_spec=outer[0, panel_index],
            width_ratios=[0.085, 0.021, 0.894],
            wspace=0.004,
        )
        dendro_ax = fig.add_subplot(inner[0, 0])
        source_ax = fig.add_subplot(inner[0, 1])
        heatmap_ax = fig.add_subplot(inner[0, 2])

        draw_left_dendrogram(dendro_ax, panel_data.linkage, panel_data.matrix.shape[0])
        draw_source_bar(source_ax, list(panel_data.matrix.index), panel_data.panel.source_by_label, max(8.5, fs - 0.6))
        draw_heatmap(heatmap_ax, panel_data.matrix, cmap, fs)
        image = heatmap_ax.images[0]
        panel_axes.append((dendro_ax, source_ax, heatmap_ax))

        heatmap_ax.set_title(
            panel_data.panel.title,
            loc='left',
            fontsize=title_fontsize(max_metrics),
            fontweight='bold',
            pad=6,
        )
        heatmap_ax.text(
            -0.16,
            1.02,
            chr(ord('A') + panel_index),
            transform=heatmap_ax.transAxes,
            ha='left',
            va='bottom',
            fontsize=title_fontsize(max_metrics) + 1.5,
            fontweight='bold',
            clip_on=False,
        )

    similarity_ax = fig.add_subplot(outer[2, :])
    draw_similarity_panel(similarity_ax, similarity, max(8.8, fs - 0.4), profile_similarity_method)

    fig.subplots_adjust(left=0.055, right=0.965, top=0.955, bottom=0.060)
    align_side_axes_to_heatmaps(fig, panel_axes)

    if image is None:
        raise RuntimeError('No matrices were plotted.')
    cbar_ax = fig.add_subplot(outer[1, 0])
    cbar = fig.colorbar(image, cax=cbar_ax, orientation='horizontal')
    cbar.set_ticks([-1, -0.5, 0, 0.5, 1])
    cbar.ax.tick_params(labelsize=max(9.8, fs - 0.6), length=3)
    cbar.set_label(cbar_label(correlation), fontsize=max(10.5, fs - 0.1), labelpad=5)
    cbar.ax.xaxis.label.set_fontweight('bold')

    legend_ax = fig.add_subplot(outer[1, 1])
    legend_ax.axis('off')
    sources = observed_sources(matrices, [panel.panel for panel in panels])
    handles = [
        Patch(
            facecolor=SOURCE_IMAGE_COLORS[source],
            edgecolor='none',
            label=source_display_label(source),
        )
        for source in sources
    ]
    legend = legend_ax.legend(
        handles=handles,
        title=METRIC_FAMILY_LEGEND_TITLE,
        loc='center',
        ncol=max(1, len(handles)),
        frameon=False,
        fontsize=max(9.8, fs - 0.5),
        title_fontsize=max(10.8, fs),
        handlelength=1.4,
        columnspacing=1.15,
    )
    legend.get_title().set_fontweight('bold')
    position_guides(fig, cbar_ax, legend_ax, [axes[2] for axes in panel_axes])

    out_stem.parent.mkdir(parents=True, exist_ok=True)
    similarity_path = out_stem.with_name(out_stem.name + '_profile_similarity.tsv')
    similarity.to_csv(similarity_path, sep='\t', index=False)
    print(f'Wrote: {similarity_path}', flush=True)
    for extension in ('pdf', 'png'):
        out_path = out_stem.with_suffix(f'.{extension}')
        fig.savefig(out_path, bbox_inches='tight', dpi=300)
        print(f'Wrote: {out_path}', flush=True)
    plt.close(fig)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--patterns-file',
        type=Path,
        default=CODE_ROOT / 'configuration' / 'patterns.json',
        help='Metric pattern registry used to assign metric-family colors.',
    )
    parser.add_argument(
        '--mni-dir',
        type=Path,
        default=default_mni_dir(),
        help='Directory containing compute_mni_voxelwise_correlations.py TSV outputs.',
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=PROJECT_ROOT / 'figures' / 'voxel_correlation_matrices_with_similarity',
        help='Directory for voxel correlation figures.',
    )
    parser.add_argument(
        '--analysis-set',
        nargs='+',
        choices=('primary', 'full', 'both'),
        default=['primary'],
        help='Metric set(s) to plot.',
    )
    parser.add_argument(
        '--correlation',
        nargs='+',
        choices=('pearson', 'spearman', 'both'),
        default=['pearson'],
        help='Voxelwise correlation method(s) to plot.',
    )
    parser.add_argument(
        '--profile-similarity-method',
        choices=PROFILE_SIMILARITY_METHODS,
        default='pearson',
        help='Method used to correlate each metric correlation profile between WM and GM.',
    )
    parser.add_argument(
        '--strict',
        action='store_true',
        help='Fail instead of warning when an expected input matrix is missing.',
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    require_dependencies()
    mpl.rcParams['font.family'] = 'Arial'
    mpl.rcParams['pdf.fonttype'] = 42
    mpl.rcParams['ps.fonttype'] = 42
    sns.set_theme(style='white', font='Arial')

    analysis_sets = selected_values(args.analysis_set, ANALYSIS_SETS)
    correlations = selected_values(args.correlation, MNI_CORRELATIONS)

    for analysis_set in analysis_sets:
        for correlation in correlations:
            expected_paths = [
                mni_input_path(args.mni_dir, analysis_set, 'wm', correlation),
                mni_input_path(args.mni_dir, analysis_set, 'gm', correlation),
            ]
            missing = [path for path in expected_paths if not path.exists()]
            if missing:
                message = 'Missing input(s), skipping voxel correlation figure: ' + ', '.join(map(str, missing))
                if args.strict:
                    raise FileNotFoundError(message)
                print(f'[WARN] {message}', flush=True)
                continue
            panels, similarity = load_ordered_data(
                analysis_set,
                correlation,
                args.patterns_file,
                args.mni_dir,
                args.profile_similarity_method,
            )
            out_stem = (
                args.output_dir
                / f'voxel_correlations_{analysis_set}_{correlation}_gm-wm-profile-similarity'
            )
            draw_figure(
                panels,
                similarity,
                analysis_set,
                correlation,
                args.profile_similarity_method,
                out_stem,
            )


if __name__ == '__main__':
    main()
