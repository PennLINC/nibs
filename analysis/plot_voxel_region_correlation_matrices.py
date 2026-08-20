#!/usr/bin/env python3
"""Plot voxelwise and parcel/bundle correlation matrices in one combined figure."""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

try:
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    import seaborn as sns
    from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
    from matplotlib.patches import Patch
except ImportError:  # pragma: no cover - checked after argparse handles --help
    mpl = None
    plt = None
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
    PARCEL_CORRELATIONS,
    PARCEL_STAT,
    load_correlation_matrix,
    mni_input_path,
    parcel_input_path,
    selected_values,
    source_display_label,
)


ANALYSIS_SETS = ('primary', 'full')
MNI_CORRELATIONS = ('pearson', 'spearman')


@dataclass(frozen=True)
class MixedPanelSpec:
    panel: PanelSpec
    kind: str
    correlation: str


def default_mni_dir() -> Path:
    return PROJECT_ROOT / 'derivatives' / 'mni_voxelwise_correlations'


def default_parcel_dir() -> Path:
    return PROJECT_ROOT / 'derivatives' / 'parcel_bundle_correlations'


def mixed_figure_size(max_metrics: int) -> tuple[float, float]:
    panel_side = max(5.2, min(7.6, 1.25 + 0.23 * max_metrics))
    width = 2.0 * panel_side + 5.5
    height = 2.0 * panel_side + 4.0
    return width, height


def shared_cbar_label(mni_correlation: str, parcel_correlation: str) -> str:
    mni_label = r'Pearson $r$' if mni_correlation == 'pearson' else 'Spearman ρ'
    parcel_label = r'Pearson $r$' if parcel_correlation == 'pearson' else 'Spearman ρ'
    return f'Mean correlation (voxels: {mni_label}; regions: {parcel_label})'


def position_bottom_guides(
    fig: plt.Figure,
    cbar_ax: plt.Axes,
    legend_ax: plt.Axes,
    bottom_heatmap_axes: list[plt.Axes],
) -> None:
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    tick_boxes = [
        tick.get_window_extent(renderer)
        for ax in bottom_heatmap_axes
        for tick in ax.get_xticklabels()
        if tick.get_visible() and tick.get_text()
    ]
    if tick_boxes:
        label_bottom_display = min(box.y0 for box in tick_boxes)
        label_bottom = fig.transFigure.inverted().transform((0, label_bottom_display))[1]
    else:
        label_bottom = min(ax.get_position().y0 for ax in bottom_heatmap_axes)

    guide_y0 = max(0.045, label_bottom - 0.038)
    cbar_ax.set_position([0.20, guide_y0 + 0.009, 0.26, 0.014])
    legend_ax.set_position([0.49, guide_y0 - 0.003, 0.47, 0.050])


def draw_mixed_figure(
    panel_specs: list[MixedPanelSpec],
    analysis_set: str,
    mni_correlation: str,
    parcel_correlation: str,
    out_stem: Path,
) -> None:
    loaded = [load_correlation_matrix(spec.panel.path) for spec in panel_specs]
    ordered = [ordered_matrix(matrix) for matrix in loaded]
    matrices = [item[0] for item in ordered]
    linkages = [item[1] for item in ordered]
    max_metrics = max(matrix.shape[0] for matrix in matrices)
    fs = max(9.6, label_fontsize(max_metrics) - 0.7)

    cmap = sns.diverging_palette(220, 20, as_cmap=True)
    cmap.set_bad('#eeeeee')

    fig_width, fig_height = mixed_figure_size(max_metrics)
    fig = plt.figure(figsize=(fig_width, fig_height), constrained_layout=False)
    outer = GridSpec(
        3,
        2,
        figure=fig,
        height_ratios=[1, 1, 0.055],
        width_ratios=[1, 1],
        hspace=0.58,
        wspace=0.38,
    )

    image = None
    panel_axes: list[tuple[plt.Axes, plt.Axes, plt.Axes]] = []
    heatmap_axes: list[plt.Axes] = []
    for panel_index, (spec, matrix, z_matrix) in enumerate(zip(panel_specs, matrices, linkages, strict=True)):
        row = 0 if panel_index < 2 else 1
        col = panel_index % 2
        inner = GridSpecFromSubplotSpec(
            1,
            3,
            subplot_spec=outer[row, col],
            width_ratios=[0.085, 0.021, 0.894],
            wspace=0.004,
        )
        dendro_ax = fig.add_subplot(inner[0, 0])
        source_ax = fig.add_subplot(inner[0, 1])
        heatmap_ax = fig.add_subplot(inner[0, 2])

        draw_left_dendrogram(dendro_ax, z_matrix, matrix.shape[0])
        draw_source_bar(source_ax, list(matrix.index), spec.panel.source_by_label, max(8.5, fs - 0.6))
        draw_heatmap(heatmap_ax, matrix, cmap, fs)
        image = heatmap_ax.images[0]
        panel_axes.append((dendro_ax, source_ax, heatmap_ax))
        heatmap_axes.append(heatmap_ax)

        heatmap_ax.set_title(
            spec.panel.title,
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

    fig.subplots_adjust(left=0.030, right=0.965, top=0.960, bottom=0.045)
    align_side_axes_to_heatmaps(fig, panel_axes)

    if image is None:
        raise RuntimeError('No matrices were plotted.')
    cbar_ax = fig.add_subplot(outer[2, 0])
    cbar = fig.colorbar(image, cax=cbar_ax, orientation='horizontal')
    cbar.set_ticks([-1, -0.5, 0, 0.5, 1])
    cbar.ax.tick_params(labelsize=max(9.8, fs - 0.6), length=3)
    cbar.set_label(
        shared_cbar_label(mni_correlation, parcel_correlation),
        fontsize=max(10.5, fs - 0.1),
        labelpad=5,
    )

    legend_ax = fig.add_subplot(outer[2, 1])
    legend_ax.axis('off')
    panel_panels = [spec.panel for spec in panel_specs]
    sources = observed_sources(matrices, panel_panels)
    handles = [
        Patch(
            facecolor=SOURCE_IMAGE_COLORS[source],
            edgecolor='none',
            label=source_display_label(source),
        )
        for source in sources
    ]
    legend_ax.legend(
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
    position_bottom_guides(fig, cbar_ax, legend_ax, heatmap_axes[2:])

    out_stem.parent.mkdir(parents=True, exist_ok=True)
    for extension in ('pdf', 'png'):
        out_path = out_stem.with_suffix(f'.{extension}')
        fig.savefig(out_path, bbox_inches='tight', dpi=300)
        print(f'Wrote: {out_path}', flush=True)
    plt.close(fig)


def build_mixed_panel_specs(
    analysis_set: str,
    mni_correlation: str,
    parcel_correlation: str,
    patterns_file: Path,
    mni_dir: Path,
    parcel_dir: Path,
    parcel_stat: str,
) -> list[MixedPanelSpec]:
    return [
        MixedPanelSpec(
            PanelSpec(
                tissue='wm',
                title='White Matter Voxels',
                path=mni_input_path(mni_dir, analysis_set, 'wm', mni_correlation),
                source_by_label=source_lookup(patterns_file, analysis_set, 'wm'),
            ),
            kind='mni',
            correlation=mni_correlation,
        ),
        MixedPanelSpec(
            PanelSpec(
                tissue='wm',
                title='White Matter Bundles',
                path=parcel_input_path(parcel_dir, analysis_set, 'wm', parcel_correlation, parcel_stat),
                source_by_label=source_lookup(patterns_file, analysis_set, 'wm'),
            ),
            kind='parcel',
            correlation=parcel_correlation,
        ),
        MixedPanelSpec(
            PanelSpec(
                tissue='gm',
                title='Gray Matter Voxels',
                path=mni_input_path(mni_dir, analysis_set, 'gm', mni_correlation),
                source_by_label=source_lookup(patterns_file, analysis_set, 'gm'),
            ),
            kind='mni',
            correlation=mni_correlation,
        ),
        MixedPanelSpec(
            PanelSpec(
                tissue='gm',
                title='Gray Matter Parcels',
                path=parcel_input_path(parcel_dir, analysis_set, 'gm', parcel_correlation, parcel_stat),
                source_by_label=source_lookup(patterns_file, analysis_set, 'gm'),
            ),
            kind='parcel',
            correlation=parcel_correlation,
        ),
    ]


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
        '--parcel-dir',
        type=Path,
        default=default_parcel_dir(),
        help='Directory containing compute_parcel_bundle_correlations.py TSV outputs.',
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=PROJECT_ROOT / 'figures' / 'correlation_matrices_voxel_region',
        help='Directory for mixed voxel/region correlation matrix figures.',
    )
    parser.add_argument(
        '--analysis-set',
        nargs='+',
        choices=('primary', 'full', 'both'),
        default=['primary'],
        help='Metric set(s) to plot.',
    )
    parser.add_argument(
        '--mni-correlation',
        choices=MNI_CORRELATIONS,
        default='pearson',
        help='Voxelwise correlation method to plot.',
    )
    parser.add_argument(
        '--parcel-correlation',
        choices=PARCEL_CORRELATIONS,
        default='spearman',
        help='Parcel/bundle correlation method to plot.',
    )
    parser.add_argument(
        '--parcel-stat',
        choices=('mean', 'median'),
        default=PARCEL_STAT,
        help='Parcel/bundle summary statistic used in parcel correlation filenames.',
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

    for analysis_set in analysis_sets:
        panel_specs = build_mixed_panel_specs(
            analysis_set,
            args.mni_correlation,
            args.parcel_correlation,
            args.patterns_file,
            args.mni_dir,
            args.parcel_dir,
            args.parcel_stat,
        )
        missing = [spec.panel.path for spec in panel_specs if not spec.panel.path.exists()]
        if missing:
            message = 'Missing input(s), skipping mixed voxel/region figure: ' + ', '.join(map(str, missing))
            if args.strict:
                raise FileNotFoundError(message)
            print(f'[WARN] {message}', flush=True)
            continue
        out_stem = (
            args.output_dir
            / f'voxel_region_{analysis_set}_mni-{args.mni_correlation}_'
            f'parcel-{args.parcel_correlation}_{args.parcel_stat}_correlations'
        )
        draw_mixed_figure(
            panel_specs,
            analysis_set,
            args.mni_correlation,
            args.parcel_correlation,
            out_stem,
        )


if __name__ == '__main__':
    main()
