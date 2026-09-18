#!/usr/bin/env python3
"""Plot full-metric regional ICC heatmaps and voxelwise ICC interval figures."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

try:
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    from matplotlib.patches import Patch, Rectangle
except ImportError:  # pragma: no cover
    mpl = None
    plt = None
    np = None
    pd = None
    Patch = None
    Rectangle = None

sys.path.insert(0, str(Path(__file__).resolve().parent))

from metric_registry import (  # noqa: E402
    METRIC_FAMILY_LEGEND_TITLE,
    SOURCE_IMAGE_COLORS,
    build_metric_specs,
    metric_display_labels,
    source_image_display_label,
)
from path_utils import CODE_ROOT, PROJECT_ROOT  # noqa: E402
from plot_icc_figures import (  # noqa: E402
    BENCHMARKS,
    default_mni_icc_dir,
    default_parcel_bundle_icc_dir,
    load_voxelwise_icc,
    require_dependencies,
    summarize_metric_values,
)


REGIONAL_DOMAINS = {
    'wm': ('wm_bundles', 'White Matter Bundles'),
    'gm': ('gm_parcels', 'Cortical Gray Matter Parcels'),
}
VOXEL_TITLES = {
    'wm': 'White Matter Voxelwise ICC',
    'gm': 'Cortical Gray Matter Voxelwise ICC',
}


def icc_display_label(icc_column: str) -> str:
    return {'ICC2_1': 'ICC(2,1)', 'ICC3_1': 'ICC(3,1)'}[icc_column]


def regional_input_path(icc_dir: Path, tissue: str, analysis_set: str, stat: str) -> Path:
    profile, _ = REGIONAL_DOMAINS[tissue]
    return icc_dir / f'icc_{profile}_{analysis_set}_{stat}.csv'


def load_regional_icc(
    path: Path,
    tissue: str,
    patterns_file: Path,
    analysis_set: str,
    icc_column: str,
) -> pd.DataFrame:
    table = pd.read_csv(path)
    required = {'feature', icc_column}
    missing = required - set(table.columns)
    if missing:
        raise RuntimeError(f'{path} is missing required columns: {", ".join(sorted(missing))}')
    if 'metric_key' not in table:
        if 'metric' not in table:
            raise RuntimeError(f'{path} has neither metric_key nor metric.')
        table['metric_key'] = table['metric'].astype(str)

    specs = build_metric_specs(patterns_file)
    displays = metric_display_labels(specs, analysis_set, tissue=tissue)
    sources = {spec.label: spec.source_image for spec in specs}
    table = table.copy()
    table[icc_column] = pd.to_numeric(table[icc_column], errors='coerce')
    table['metric_key'] = table['metric_key'].astype(str)
    table['metric'] = table['metric_key'].map(displays).fillna(table['metric_key'])
    table['source_image'] = table['metric_key'].map(sources).fillna(
        table.get('source_image', 'Other')
    )
    return table.dropna(subset=[icc_column])


def ordered_icc_matrix(
    table: pd.DataFrame,
    icc_column: str,
) -> tuple[pd.DataFrame, pd.Series, pd.Series, dict[str, str], dict[str, str]]:
    matrix = table.pivot_table(
        index='feature',
        columns='metric_key',
        values=icc_column,
        aggfunc='mean',
    )
    row_means = matrix.mean(axis=1, skipna=True).sort_values(ascending=False)
    column_means = matrix.mean(axis=0, skipna=True).sort_values(ascending=False)
    matrix = matrix.loc[row_means.index, column_means.index]
    display = (
        table.drop_duplicates('metric_key').set_index('metric_key')['metric'].astype(str).to_dict()
    )
    source = (
        table.drop_duplicates('metric_key')
        .set_index('metric_key')['source_image']
        .astype(str)
        .to_dict()
    )
    return matrix, row_means, column_means, display, source


def save_order_table(
    output_stem: Path,
    row_means: pd.Series,
    column_means: pd.Series,
    display: dict[str, str],
) -> None:
    records = []
    for rank, (feature, value) in enumerate(row_means.items(), start=1):
        records.append(
            {'axis': 'region', 'rank': rank, 'key': feature, 'label': feature, 'mean_icc': value}
        )
    for rank, (metric, value) in enumerate(column_means.items(), start=1):
        records.append(
            {
                'axis': 'metric',
                'rank': rank,
                'key': metric,
                'label': display.get(metric, metric),
                'mean_icc': value,
            }
        )
    pd.DataFrame(records).to_csv(
        output_stem.with_name(output_stem.name + '_ordering.tsv'),
        sep='\t',
        index=False,
    )


def plot_regional_heatmap(
    table: pd.DataFrame,
    tissue: str,
    icc_column: str,
    icc_label: str,
    output_stem: Path,
) -> None:
    matrix, row_means, column_means, display, source = ordered_icc_matrix(table, icc_column)
    if matrix.empty:
        raise RuntimeError(f'No finite regional ICC values for {tissue}.')

    n_rows, n_columns = matrix.shape
    fig_width = max(14.0, min(34.0, 5.0 + 0.29 * n_columns))
    fig_height = max(8.0, min(28.0, 3.7 + 0.20 * n_rows))
    fig = plt.figure(figsize=(fig_width, fig_height), constrained_layout=False)
    grid = fig.add_gridspec(
        2,
        1,
        height_ratios=[0.018, 1.0],
        left=0.19,
        right=0.975,
        bottom=0.29,
        top=0.91,
        hspace=0.008,
    )
    family_ax = fig.add_subplot(grid[0, 0])
    ax = fig.add_subplot(grid[1, 0], sharex=family_ax)

    cmap = mpl.colormaps['viridis'].copy()
    cmap.set_bad('#eeeeee')
    cmap.set_under('#52245f')
    image = ax.imshow(
        matrix.to_numpy(dtype=float),
        aspect='auto',
        interpolation='nearest',
        cmap=cmap,
        vmin=0.0,
        vmax=1.0,
    )
    column_sources = [source.get(metric, 'Other') for metric in matrix.columns]
    family_colors = np.asarray(
        [mpl.colors.to_rgba(SOURCE_IMAGE_COLORS.get(item, SOURCE_IMAGE_COLORS['Other'])) for item in column_sources]
    ).reshape(1, n_columns, 4)
    family_ax.imshow(family_colors, aspect='auto', interpolation='nearest')
    family_ax.set_axis_off()

    ax.set_xticks(np.arange(n_columns))
    ax.set_xticklabels(
        [display.get(metric, metric) for metric in matrix.columns],
        rotation=55,
        ha='right',
        rotation_mode='anchor',
        fontsize=max(5.2, min(8.2, 460.0 / max(n_columns, 1))),
    )
    ax.set_yticks(np.arange(n_rows))
    ax.set_yticklabels(matrix.index, fontsize=max(4.8, min(8.0, 360.0 / max(n_rows, 1))))
    ax.tick_params(length=0, pad=2)
    ax.set_ylabel('Bundle' if tissue == 'wm' else 'Parcel', fontweight='bold')
    ax.set_title(
        f'{REGIONAL_DOMAINS[tissue][1]} {icc_label}',
        loc='left',
        fontsize=17,
        fontweight='bold',
        pad=15,
    )

    cbar_ax = fig.add_axes([0.36, 0.09, 0.36, 0.014])
    cbar = fig.colorbar(image, cax=cbar_ax, orientation='horizontal')
    cbar.set_label(icc_label, fontweight='bold')
    cbar.ax.xaxis.set_label_position('top')
    cbar.set_ticks([0, 0.25, 0.5, 0.75, 1.0])

    observed_sources = [
        key for key in SOURCE_IMAGE_COLORS if key in set(column_sources)
    ]
    handles = [
        Patch(
            facecolor=SOURCE_IMAGE_COLORS[key],
            edgecolor='none',
            label=source_image_display_label(key),
        )
        for key in observed_sources
    ]
    fig.legend(
        handles=handles,
        title=METRIC_FAMILY_LEGEND_TITLE,
        loc='lower center',
        bbox_to_anchor=(0.55, 0.005),
        ncol=max(1, len(handles)),
        frameon=False,
        fontsize=9,
        title_fontsize=10,
    )

    output_stem.parent.mkdir(parents=True, exist_ok=True)
    save_order_table(output_stem, row_means, column_means, display)
    for extension in ('pdf', 'png'):
        output = output_stem.with_suffix(f'.{extension}')
        fig.savefig(output, dpi=240, bbox_inches='tight')
        print(f'Wrote: {output}')
    plt.close(fig)


def plot_voxel_intervals(
    voxel_data: pd.DataFrame,
    tissue: str,
    output_stem: Path,
) -> None:
    summary = summarize_metric_values(voxel_data)
    summary = summary.loc[summary['tissue'] == tissue].copy()
    summary = summary.sort_values(['median', 'metric'], ascending=[True, True])
    if summary.empty:
        raise RuntimeError(f'No finite voxelwise ICC values for {tissue}.')

    positions = np.arange(len(summary))
    fig_height = max(8.0, 1.8 + 0.265 * len(summary))
    fig, ax = plt.subplots(figsize=(9.0, fig_height), constrained_layout=False)
    for position, (_, row) in zip(positions, summary.iterrows(), strict=True):
        color = SOURCE_IMAGE_COLORS.get(row['source_image'], SOURCE_IMAGE_COLORS['Other'])
        ax.add_patch(
            Rectangle(
                (row['q25'], position - 0.22),
                max(row['q75'] - row['q25'], 0.001),
                0.44,
                facecolor=color,
                edgecolor='#2b2b2b',
                linewidth=0.65,
                alpha=0.88,
                zorder=2,
            )
        )
        ax.plot(
            [row['median'], row['median']],
            [position - 0.25, position + 0.25],
            color='white',
            lw=1.8,
            zorder=3,
        )
        ax.scatter(
            [row['median']],
            [position],
            s=22,
            facecolor='white',
            edgecolor='#2b2b2b',
            linewidth=0.65,
            zorder=4,
        )

    for benchmark in BENCHMARKS:
        ax.axvline(
            benchmark,
            color='#c7c7c7',
            lw=0.9 if benchmark else 1.1,
            ls='-' if benchmark else ':',
            zorder=0,
        )
    ax.set_yticks(positions)
    ax.set_yticklabels(summary['metric'], fontsize=8.8)
    ax.tick_params(axis='y', length=0, pad=4)
    ax.set_ylim(-0.8, len(summary) - 0.2)
    ax.set_xlim(0.0, 1.0)
    ax.set_xlabel(
        'ICC(2,1) across white matter voxels'
        if tissue == 'wm'
        else 'ICC(2,1) across cortical gray matter voxels',
        fontweight='bold',
    )
    ax.set_title(VOXEL_TITLES[tissue], loc='left', fontsize=16, fontweight='bold')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    sources = [
        key for key in SOURCE_IMAGE_COLORS if key in set(summary['source_image'])
    ]
    handles = [
        Patch(
            facecolor=SOURCE_IMAGE_COLORS[key],
            edgecolor='none',
            label=source_image_display_label(key),
        )
        for key in sources
    ]
    fig.legend(
        handles=handles,
        title=METRIC_FAMILY_LEGEND_TITLE,
        loc='lower center',
        bbox_to_anchor=(0.58, -0.035),
        ncol=min(4, len(handles)),
        frameon=False,
        fontsize=9,
        title_fontsize=10,
    )
    fig.subplots_adjust(left=0.33, right=0.97, top=0.96, bottom=0.08)

    output_stem.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output_stem.with_suffix('.summary.tsv'), sep='\t', index=False)
    for extension in ('pdf', 'png'):
        output = output_stem.with_suffix(f'.{extension}')
        fig.savefig(output, dpi=240, bbox_inches='tight')
        print(f'Wrote: {output}')
    plt.close(fig)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--analysis-set', choices=('primary', 'full'), default='full')
    parser.add_argument('--stat', choices=('mean', 'median'), default='median')
    parser.add_argument(
        '--icc-column',
        choices=('ICC2_1', 'ICC3_1'),
        default='ICC2_1',
        help='ICC column for parcel/bundle heatmaps; voxelwise maps are ICC(2,1).',
    )
    parser.add_argument('--mni-icc-dir', type=Path, default=default_mni_icc_dir())
    parser.add_argument(
        '--parcel-bundle-icc-dir',
        type=Path,
        default=default_parcel_bundle_icc_dir(),
    )
    parser.add_argument(
        '--patterns-file',
        type=Path,
        default=CODE_ROOT / 'configuration' / 'patterns.json',
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=PROJECT_ROOT / 'figures' / 'supplemental_icc',
    )
    parser.add_argument('--voxelwise-analysis', default='primary')
    parser.add_argument('--max-voxels-per-metric', type=int, default=100000)
    parser.add_argument('--skip-regional', action='store_true')
    parser.add_argument('--skip-voxelwise', action='store_true')
    parser.add_argument('--strict', action='store_true')
    return parser


def main() -> None:
    args = build_parser().parse_args()
    require_dependencies(need_nibabel=not args.skip_voxelwise)
    mpl.rcParams.update(
        {
            'font.family': 'Arial',
            'pdf.fonttype': 42,
            'ps.fonttype': 42,
        }
    )
    patterns_file = args.patterns_file.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()

    if not args.skip_regional:
        for tissue in ('wm', 'gm'):
            input_path = regional_input_path(
                args.parcel_bundle_icc_dir.expanduser().resolve(),
                tissue,
                args.analysis_set,
                args.stat,
            )
            if not input_path.exists():
                message = f'Missing regional ICC input, skipping: {input_path}'
                if args.strict:
                    raise FileNotFoundError(message)
                print(f'[WARN] {message}', file=sys.stderr)
                continue
            table = load_regional_icc(
                input_path,
                tissue,
                patterns_file,
                args.analysis_set,
                args.icc_column,
            )
            icc_label = icc_display_label(args.icc_column)
            output_stem = output_dir / f'icc_{args.analysis_set}_{tissue}_{REGIONAL_DOMAINS[tissue][0]}_heatmap'
            plot_regional_heatmap(table, tissue, args.icc_column, icc_label, output_stem)

    if not args.skip_voxelwise:
        voxel_data = load_voxelwise_icc(
            args.mni_icc_dir.expanduser().resolve(),
            args.analysis_set,
            args.voxelwise_analysis,
            patterns_file,
            args.max_voxels_per_metric,
        )
        for tissue in ('wm', 'gm'):
            tissue_data = voxel_data.loc[voxel_data['tissue'] == tissue].copy()
            if tissue_data.empty:
                message = f'No voxelwise ICC data for {tissue}; skipping.'
                if args.strict:
                    raise RuntimeError(message)
                print(f'[WARN] {message}', file=sys.stderr)
                continue
            plot_voxel_intervals(
                tissue_data,
                tissue,
                output_dir / f'icc_{args.analysis_set}_{tissue}_voxels',
            )


if __name__ == '__main__':
    main()
