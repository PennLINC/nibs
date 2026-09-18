#!/usr/bin/env python3
"""Plot full-metric WM-bundle and GM-parcel discriminability as faceted heatmaps."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

try:
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
except ImportError:  # pragma: no cover
    mpl = None
    plt = None
    np = None
    pd = None

sys.path.insert(0, str(Path(__file__).resolve().parent))

from metric_registry import build_metric_specs  # noqa: E402
from path_utils import CODE_ROOT, DERIVATIVES_ROOT, PROJECT_ROOT  # noqa: E402
from plot_parcel_bundle_discriminability import (  # noqa: E402
    SCORE_COLUMNS,
    default_gm_input,
    default_wm_input,
    load_discriminability_table,
    require_dependencies,
)


CATEGORY_LABELS = {
    'Tensor': 'TORTOISE Tensor / MAPMRI Tensor',
    'DKI': 'DKI',
    'DKI Micro': 'DKI Microstructure',
    'NODDI': 'NODDI',
    'MAPMRI': 'MAPMRI',
    'GQI': 'DSI Studio GQI',
    'dMRI': 'Other dMRI',
    'T1w/T2w': 'T₁w/T₂w',
    'g-ratio': 'g-ratio',
    'R1': 'MP2RAGE R₁',
    'MESE': 'R₂',
    'MEGRE': 'MEGRE',
    'Q-Ratio': 'Q-ratio',
    'ihMT': 'ihMT',
    'QSM': 'QSM',
}


def metric_categories(patterns_file: Path, level: str) -> tuple[dict[str, str], list[str]]:
    specs = build_metric_specs(patterns_file)
    if level == 'family':
        def supplemental_family(spec) -> str:
            # Keep the dMRI model subdivisions from the registry, while
            # retaining acquisition-derived groups such as Q-ratio as their
            # own publication-facing categories.
            if spec.group == 'Q-Ratio':
                return 'Q-Ratio'
            return spec.family

        lookup = {spec.label: supplemental_family(spec) for spec in specs}
        order = list(dict.fromkeys(supplemental_family(spec) for spec in specs))
    elif level == 'group':
        lookup = {spec.label: spec.group for spec in specs}
        order = list(dict.fromkeys(spec.group for spec in specs))
    elif level == 'source-image':
        lookup = {spec.label: spec.source_image for spec in specs}
        order = list(dict.fromkeys(spec.source_image for spec in specs))
    else:  # pragma: no cover - argparse constrains this
        raise ValueError(level)
    return lookup, order


def add_categories(
    data: pd.DataFrame,
    patterns_file: Path,
    level: str,
) -> tuple[pd.DataFrame, list[str]]:
    lookup, registry_order = metric_categories(patterns_file, level)
    out = data.copy()
    out['category'] = out['metric_key'].map(lookup).fillna('Other')
    observed = set(out['category'])
    order = [category for category in registry_order if category in observed]
    if 'Other' in observed:
        order.append('Other')
    return out, order


def pack_categories(
    categories: list[str],
    counts: dict[str, int],
    max_columns_per_row: int,
) -> list[list[str]]:
    rows: list[list[str]] = []
    current: list[str] = []
    current_columns = 0
    for category in categories:
        # Reserve a little horizontal space for each panel's title and row
        # labels so a run of one- or two-metric families does not collide.
        n_columns = max(1, counts[category]) + 2
        if current and current_columns + n_columns > max_columns_per_row:
            rows.append(current)
            current = []
            current_columns = 0
        current.append(category)
        current_columns += n_columns
    if current:
        rows.append(current)
    return rows


def category_matrix(
    data: pd.DataFrame,
    category: str,
) -> tuple[pd.DataFrame, dict[str, str], pd.Series]:
    category_data = data.loc[data['category'] == category].copy()
    matrix = category_data.pivot_table(
        index='tissue',
        columns='metric_key',
        values='score',
        aggfunc='first',
    ).reindex(index=['wm', 'gm'])
    means = matrix.mean(axis=0, skipna=True).sort_values(ascending=False)
    matrix = matrix.reindex(columns=means.index)
    display = (
        category_data.drop_duplicates('metric_key')
        .set_index('metric_key')['metric']
        .astype(str)
        .to_dict()
    )
    return matrix, display, means


def annotation_color(value: float) -> str:
    return 'white' if value >= 0.56 else '#202124'


def plot_faceted_heatmaps(
    data: pd.DataFrame,
    categories: list[str],
    score_label: str,
    output_stem: Path,
    max_columns_per_row: int,
) -> None:
    if data.empty:
        raise RuntimeError('No finite discriminability values to plot.')
    counts = {
        category: int(data.loc[data['category'] == category, 'metric_key'].nunique())
        for category in categories
    }
    category_rows = pack_categories(categories, counts, max_columns_per_row)
    fig = plt.figure(
        figsize=(20.0, max(6.0, 4.15 * len(category_rows) + 1.3)),
        constrained_layout=False,
    )
    outer = fig.add_gridspec(
        len(category_rows) + 1,
        1,
        height_ratios=[1.0] * len(category_rows) + [0.075],
        left=0.065,
        right=0.985,
        top=0.94,
        bottom=0.06,
        hspace=1.50,
    )
    cmap = mpl.colormaps['viridis'].copy()
    cmap.set_bad('#e5e5e5')
    image = None
    ordering_records: list[dict[str, object]] = []

    for row_index, row_categories in enumerate(category_rows):
        inner = outer[row_index].subgridspec(
            1,
            len(row_categories),
            width_ratios=[max(2.5, counts[category]) for category in row_categories],
            wspace=0.28,
        )
        for panel_index, category in enumerate(row_categories):
            ax = fig.add_subplot(inner[0, panel_index])
            matrix, display, means = category_matrix(data, category)
            image = ax.imshow(
                matrix.to_numpy(dtype=float),
                aspect='auto',
                interpolation='nearest',
                cmap=cmap,
                vmin=0.0,
                vmax=1.0,
            )
            for y_index in range(matrix.shape[0]):
                for x_index in range(matrix.shape[1]):
                    value = matrix.iloc[y_index, x_index]
                    if np.isfinite(value):
                        ax.text(
                            x_index,
                            y_index,
                            f'{value:.2f}',
                            ha='center',
                            va='center',
                            fontsize=max(6.2, min(9.0, 70.0 / max(matrix.shape[1], 1))),
                            color=annotation_color(float(value)),
                            fontweight='bold',
                        )
            ax.set_xticks(np.arange(matrix.shape[1]))
            ax.set_xticklabels(
                [display.get(metric, metric) for metric in matrix.columns],
                rotation=52,
                ha='right',
                rotation_mode='anchor',
                fontsize=max(6.5, min(9.0, 85.0 / max(matrix.shape[1], 1))),
            )
            ax.set_yticks([0, 1])
            ax.set_yticklabels(
                ['WM bundles', 'GM parcels'] if panel_index == 0 else ['', ''],
                fontsize=9.2,
            )
            ax.tick_params(length=0, pad=3)
            ax.set_title(
                CATEGORY_LABELS.get(category, category.replace('_', ' ').title()),
                loc='left',
                fontsize=12.5,
                fontweight='bold',
                pad=8,
            )
            for spine in ax.spines.values():
                spine.set_edgecolor('#555555')
                spine.set_linewidth(0.6)

            for rank, (metric, mean_score) in enumerate(means.items(), start=1):
                ordering_records.append(
                    {
                        'category': category,
                        'rank_within_category': rank,
                        'metric_key': metric,
                        'metric': display.get(metric, metric),
                        'mean_wm_gm_score': mean_score,
                    }
                )

    if image is None:
        raise RuntimeError('No discriminability panels were drawn.')
    cbar_ax = fig.add_subplot(outer[-1, 0])
    cbar = fig.colorbar(image, cax=cbar_ax, orientation='horizontal')
    cbar.set_ticks([0, 0.25, 0.5, 0.75, 1.0])
    cbar.set_label(score_label, fontweight='bold', labelpad=6)
    fig.suptitle(
        f'White Matter Bundle and Gray Matter Parcel {score_label}',
        fontsize=17,
        fontweight='bold',
        y=0.975,
    )

    output_stem.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(output_stem.with_suffix('.summary.tsv'), sep='\t', index=False)
    pd.DataFrame(ordering_records).to_csv(
        output_stem.with_name(output_stem.name + '_ordering.tsv'),
        sep='\t',
        index=False,
    )
    for extension in ('pdf', 'png'):
        output = output_stem.with_suffix(f'.{extension}')
        fig.savefig(output, dpi=260, bbox_inches='tight')
        print(f'Wrote: {output}')
    plt.close(fig)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--analysis-set', choices=('primary', 'full'), default='full')
    parser.add_argument('--stat', choices=('mean', 'median'), default='median')
    parser.add_argument('--distance-metric', choices=('correlation', 'euclidean'), default='correlation')
    parser.add_argument('--score-column', choices=tuple(SCORE_COLUMNS), default='discriminability')
    parser.add_argument(
        '--category-level',
        choices=('family', 'group', 'source-image'),
        default='family',
        help='Registry field used to create separate metric-category heatmaps.',
    )
    parser.add_argument('--max-columns-per-row', type=int, default=28)
    parser.add_argument(
        '--input-dir',
        type=Path,
        default=DERIVATIVES_ROOT / 'parcel_bundle_discriminability',
    )
    parser.add_argument('--wm-input', type=Path, default=None)
    parser.add_argument('--gm-input', type=Path, default=None)
    parser.add_argument(
        '--patterns-file',
        type=Path,
        default=CODE_ROOT / 'configuration' / 'patterns.json',
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=PROJECT_ROOT / 'figures' / 'supplemental_discriminability',
    )
    parser.add_argument('--output-name', default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    require_dependencies()
    mpl.rcParams.update(
        {
            'font.family': 'Arial',
            'pdf.fonttype': 42,
            'ps.fonttype': 42,
        }
    )
    input_dir = args.input_dir.expanduser().resolve()
    wm_input = (
        args.wm_input.expanduser().resolve()
        if args.wm_input
        else default_wm_input(input_dir, args.analysis_set, args.stat, args.distance_metric)
    )
    gm_input = (
        args.gm_input.expanduser().resolve()
        if args.gm_input
        else default_gm_input(input_dir, args.analysis_set, args.stat, args.distance_metric)
    )
    data = pd.concat(
        [
            load_discriminability_table(wm_input, 'wm', args.score_column),
            load_discriminability_table(gm_input, 'gm', args.score_column),
        ],
        ignore_index=True,
    )
    data, categories = add_categories(
        data,
        args.patterns_file.expanduser().resolve(),
        args.category_level,
    )
    output_name = args.output_name or (
        f'discriminability_{args.analysis_set}_{args.stat}_{args.distance_metric}_'
        f'by_{args.category_level.replace("-", "_")}'
    )
    plot_faceted_heatmaps(
        data,
        categories,
        SCORE_COLUMNS[args.score_column],
        args.output_dir.expanduser().resolve() / output_name,
        args.max_columns_per_row,
    )


if __name__ == '__main__':
    main()
