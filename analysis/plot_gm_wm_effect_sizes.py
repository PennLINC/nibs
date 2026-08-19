#!/usr/bin/env python3
"""Plot MNI voxelwise GM-vs-WM effect sizes by metric."""

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
except ImportError:  # pragma: no cover - checked after argparse handles --help
    mpl = None
    plt = None
    np = None
    pd = None
    Patch = None
    Rectangle = None

sys.path.insert(0, str(Path(__file__).resolve().parent))

from metric_registry import SOURCE_IMAGE_COLORS
from path_utils import DERIVATIVES_ROOT, PROJECT_ROOT


EFFECT_LABELS = {
    'robust_median_d': 'Robust standardized GM - WM median difference',
    'cohen_d': "Cohen's d for GM - WM",
    'hedges_g': "Hedges' g for GM - WM",
    'signed_auc': 'Signed AUC effect for GM > WM',
    'median_difference': 'Median GM - WM difference',
    'mean_difference': 'Mean GM - WM difference',
    'percent_median_difference': 'Median GM - WM difference (% of |WM median|)',
}


def require_dependencies() -> None:
    missing = [
        name
        for name, module in (
            ('matplotlib', mpl),
            ('numpy', np),
            ('pandas', pd),
        )
        if module is None
    ]
    if missing:
        raise RuntimeError(
            'Missing required Python packages: '
            f'{", ".join(missing)}. Activate the NIBS analysis environment first.'
        )


def source_display_label(source: str) -> str:
    return {
        'T1w/T2w': 'T₁w/T₂w',
        'R1': 'R₁',
    }.get(source, source)


def load_subject_effects(path: Path, effect: str) -> pd.DataFrame:
    data = pd.read_csv(path, sep='\t')
    required = {'metric_key', 'display_metric', 'source_image', 'subject', effect}
    missing = required - set(data.columns)
    if missing:
        raise RuntimeError(f'{path} is missing required columns: {", ".join(sorted(missing))}')
    data[effect] = pd.to_numeric(data[effect], errors='coerce')
    return data.dropna(subset=[effect]).copy()


def summarize_for_plot(data: pd.DataFrame, effect: str) -> pd.DataFrame:
    rows = []
    for (metric_key, display_metric, source_image), group in data.groupby(
        ['metric_key', 'display_metric', 'source_image'],
        sort=False,
    ):
        values = group[effect].to_numpy(dtype=float)
        values = values[np.isfinite(values)]
        if values.size == 0:
            continue
        rows.append(
            {
                'metric_key': metric_key,
                'display_metric': display_metric,
                'source_image': source_image,
                'n_subjects': int(group['subject'].nunique()),
                'mean': float(np.mean(values)),
                'median': float(np.median(values)),
                'q25': float(np.percentile(values, 25)),
                'q75': float(np.percentile(values, 75)),
            }
        )
    return pd.DataFrame(rows).sort_values(['mean', 'display_metric']).reset_index(drop=True)


def axis_limits(values: np.ndarray) -> tuple[float, float]:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return -1.0, 1.0
    low, high = np.percentile(finite, [1, 99])
    low = min(low, float(np.min(finite)))
    high = max(high, float(np.max(finite)))
    span = max(high - low, 1e-6)
    pad = 0.08 * span
    low -= pad
    high += pad
    if low < 0 < high:
        max_abs = max(abs(low), abs(high))
        return -max_abs, max_abs
    return low, high


def plot_effect_sizes(
    data: pd.DataFrame,
    out_prefix: Path,
    effect: str,
    show_subject_points: bool,
) -> None:
    summary = summarize_for_plot(data, effect)
    if summary.empty:
        raise RuntimeError(f'No finite {effect} values to plot.')

    order = summary['metric_key'].tolist()
    order_lookup = {metric: index for index, metric in enumerate(order)}
    y = np.arange(len(order))
    fig_height = max(6.8, 0.31 * len(order) + 1.9)
    fig, ax = plt.subplots(figsize=(8.2, fig_height), constrained_layout=False)

    rng = np.random.default_rng(20260818)
    for _, row in summary.iterrows():
        y_pos = order_lookup[row['metric_key']]
        color = SOURCE_IMAGE_COLORS.get(row['source_image'], SOURCE_IMAGE_COLORS['Other'])
        ax.add_patch(
            Rectangle(
                (row['q25'], y_pos - 0.24),
                max(row['q75'] - row['q25'], 1e-8),
                0.48,
                facecolor=color,
                edgecolor='#2b2b2b',
                linewidth=0.8,
                alpha=0.84,
                zorder=2,
            )
        )
        ax.plot([row['median'], row['median']], [y_pos - 0.27, y_pos + 0.27], color='white', lw=1.8, zorder=3)
        ax.scatter([row['mean']], [y_pos], s=34, facecolor='white', edgecolor='#2b2b2b', linewidth=0.8, zorder=4)
        if show_subject_points:
            metric_values = data.loc[data['metric_key'] == row['metric_key'], effect].to_numpy(dtype=float)
            jitter = rng.uniform(-0.16, 0.16, size=metric_values.size)
            ax.scatter(
                metric_values,
                y_pos + jitter,
                s=8,
                color='black',
                alpha=0.22,
                linewidth=0,
                zorder=1,
            )
        ax.text(
            1.01,
            y_pos,
            f"{row['mean']:.2f} ({row['median']:.2f} [{row['q25']:.2f}, {row['q75']:.2f}])",
            transform=ax.get_yaxis_transform(),
            ha='left',
            va='center',
            fontsize=8.0,
            color='black',
            clip_on=False,
        )

    labels = summary['display_metric'].tolist()
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8.5)
    ax.tick_params(axis='y', length=0)
    x_low, x_high = axis_limits(data[effect].to_numpy(dtype=float))
    ax.set_xlim(x_low, x_high)
    ax.set_ylim(-0.8, len(order) - 0.2)
    ax.axvline(0, color='#6a6a6a', lw=1.0, ls=':', zorder=0)
    ax.grid(axis='x', color='#d6d6d6', linewidth=0.7, alpha=0.75)
    ax.grid(axis='y', visible=False)
    ax.set_xlabel(EFFECT_LABELS.get(effect, effect), fontsize=10.5)
    ax.set_ylabel('')
    ax.text(
        0.01,
        1.01,
        'WM > GM',
        transform=ax.transAxes,
        ha='left',
        va='bottom',
        fontsize=9,
        color='#555555',
    )
    ax.text(
        0.99,
        1.01,
        'GM > WM',
        transform=ax.transAxes,
        ha='right',
        va='bottom',
        fontsize=9,
        color='#555555',
    )
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    sources = [source for source in SOURCE_IMAGE_COLORS if source in set(summary['source_image'])]
    handles = [
        Patch(
            facecolor=SOURCE_IMAGE_COLORS[source],
            edgecolor='none',
            label=source_display_label(source),
        )
        for source in sources
    ]
    fig.legend(
        handles=handles,
        loc='lower center',
        ncol=min(4, len(handles)),
        title='Source image',
        frameon=False,
        bbox_to_anchor=(0.5, 0.006),
        fontsize=9.0,
        title_fontsize=9.5,
    )
    fig.subplots_adjust(left=0.26, right=0.78, top=0.965, bottom=0.105)

    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    for extension in ('png', 'pdf'):
        out_file = out_prefix.with_suffix(f'.{extension}')
        fig.savefig(out_file, bbox_inches='tight', dpi=300)
        print(f'Wrote: {out_file}')
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--input',
        type=Path,
        default=DERIVATIVES_ROOT / 'mni_gm_wm_effect_sizes' / 'mni_gm_wm_effect_sizes_primary_subject.tsv',
        help='Subject-averaged effect-size TSV from compute_mni_gm_wm_effect_sizes.py.',
    )
    parser.add_argument(
        '--output',
        type=Path,
        default=PROJECT_ROOT / 'figures' / 'gm_wm_effect_sizes' / 'gm_wm_effect_sizes_primary_robust_median_d',
    )
    parser.add_argument(
        '--effect',
        choices=tuple(EFFECT_LABELS),
        default='robust_median_d',
    )
    parser.add_argument(
        '--show-subject-points',
        action='store_true',
        help='Overlay subject-level points behind the metric summaries.',
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    require_dependencies()
    mpl.rcParams['font.family'] = 'Arial'
    mpl.rcParams['pdf.fonttype'] = 42
    mpl.rcParams['ps.fonttype'] = 42
    data = load_subject_effects(args.input.expanduser().resolve(), args.effect)
    plot_effect_sizes(
        data,
        args.output.expanduser().resolve(),
        args.effect,
        args.show_subject_points,
    )


if __name__ == '__main__':
    main()
