#!/usr/bin/env python3
"""Plot ICC distributions from voxelwise and parcel/bundle ICC outputs."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

try:
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch, Rectangle
except ImportError:  # pragma: no cover - checked after argparse handles --help
    mpl = None
    plt = None
    nib = None
    np = None
    pd = None
    Line2D = None
    Patch = None
    Rectangle = None

try:
    import nibabel as nib
except ImportError:  # pragma: no cover - only required for voxelwise maps
    nib = None

sys.path.insert(0, str(Path(__file__).resolve().parent))

from metric_registry import (
    SOURCE_IMAGE_COLORS,
    build_metric_specs,
    metric_display_labels,
)
from path_utils import CODE_ROOT, DERIVATIVES_ROOT, PROJECT_ROOT


SPACE = 'MNI152NLin2009cAsym'
TISSUE_NAMES = {'wm': 'WM', 'gm': 'GM'}
BENCHMARKS = (0.0, 0.5, 0.75, 0.9)


def require_dependencies(need_nibabel: bool) -> None:
    missing = [
        name
        for name, module in (
            ('matplotlib', mpl),
            ('numpy', np),
            ('pandas', pd),
        )
        if module is None
    ]
    if need_nibabel and nib is None:
        missing.append('nibabel')
    if missing:
        raise RuntimeError(
            'Missing required Python packages: '
            f'{", ".join(missing)}. Activate the NIBS analysis environment first.'
        )


def default_mni_icc_dir() -> Path:
    return DERIVATIVES_ROOT / 'mni_voxelwise_icc'


def default_parcel_bundle_icc_dir() -> Path:
    return DERIVATIVES_ROOT / 'parcel_bundle_icc'


def safe_label(value: object) -> str:
    return re.sub(r'[^A-Za-z0-9]+', '-', str(value)).strip('-')


def metric_slug(metric_key: str) -> str:
    label = str(metric_key)
    label = label.replace('*', 'star')
    label = label.replace('χ', 'X')
    label = label.replace('⊥', 'perp')
    return safe_label(label)


def source_and_display_lookup(patterns_file: Path, analysis_set: str) -> tuple[dict[str, str], dict[str, str]]:
    specs = build_metric_specs(patterns_file)
    source_by_label = {spec.label: spec.source_image for spec in specs}
    display_by_label: dict[str, str] = {}
    for tissue in ('wm', 'gm'):
        display_by_label.update(metric_display_labels(specs, analysis_set, tissue=tissue))
    return source_by_label, display_by_label


def color_for_source(source: str) -> str:
    return SOURCE_IMAGE_COLORS.get(source, SOURCE_IMAGE_COLORS['Other'])


def format_summary(values: np.ndarray) -> str:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return 'NA [NA, NA]'
    median = np.median(finite)
    q25, q75 = np.percentile(finite, [25, 75])
    return f'{median:.2f} [{q25:.2f}, {q75:.2f}]'


def summarize_metric_values(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (tissue, metric_key, metric, source), group in df.groupby(
        ['tissue', 'metric_key', 'metric', 'source_image'],
        sort=False,
    ):
        values = group['icc'].to_numpy(float)
        finite = values[np.isfinite(values)]
        if finite.size == 0:
            continue
        rows.append(
            {
                'tissue': tissue,
                'metric_key': metric_key,
                'metric': metric,
                'source_image': source,
                'n_values': int(finite.size),
                'mean': float(np.mean(finite)),
                'median': float(np.median(finite)),
                'q25': float(np.percentile(finite, 25)),
                'q75': float(np.percentile(finite, 75)),
                'label': f'{metric}  {format_summary(finite)}',
            }
        )
    return pd.DataFrame(rows)


def load_parcel_bundle_icc(
    icc_dir: Path,
    analysis_set: str,
    stat: str,
    patterns_file: Path,
) -> pd.DataFrame:
    source_by_label, display_by_label = source_and_display_lookup(patterns_file, analysis_set)
    inputs = {
        'wm': icc_dir / f'icc_wm_bundles_{analysis_set}_{stat}.csv',
        'gm': icc_dir / f'icc_gm_parcels_{analysis_set}_{stat}.csv',
    }
    rows = []
    for tissue, path in inputs.items():
        if not path.exists():
            print(f'[WARN] Missing input, skipping: {path}', file=sys.stderr)
            continue
        table = pd.read_csv(path)
        if 'ICC2_1' not in table.columns:
            raise RuntimeError(f'{path} is missing required column ICC2_1')
        if 'metric_key' not in table.columns:
            table['metric_key'] = table['metric'].astype(str)
        for _, row in table.iterrows():
            metric_key = str(row['metric_key'])
            rows.append(
                {
                    'tissue': tissue,
                    'metric_key': metric_key,
                    'metric': display_by_label.get(metric_key, str(row.get('metric', metric_key))),
                    'source_image': row.get('source_image', source_by_label.get(metric_key, 'Other')),
                    'icc': row['ICC2_1'],
                }
            )
    return pd.DataFrame(rows)


def find_mask(icc_dir: Path, tissue: str) -> Path:
    label = tissue.upper()
    candidates = sorted(
        icc_dir.glob(
            f'space-{SPACE}_label-{label}_desc-templateProb*Eroded*mm_mask.nii.gz'
        )
    )
    if not candidates:
        candidates = sorted(icc_dir.glob(f'space-{SPACE}_label-{label}_desc-*mask.nii.gz'))
    if not candidates:
        raise FileNotFoundError(f'Could not find {label} mask in {icc_dir}')
    return candidates[0]


def load_voxel_values(
    map_file: Path,
    mask_data: np.ndarray,
    max_voxels: int | None,
    seed: int,
) -> np.ndarray:
    values = np.asanyarray(nib.load(map_file).dataobj, dtype=np.float32)[mask_data]
    values = values[np.isfinite(values)]
    if max_voxels is not None and values.size > max_voxels:
        rng = np.random.default_rng(seed)
        values = values[rng.choice(values.size, size=max_voxels, replace=False)]
    return values.astype(float)


def load_voxelwise_icc(
    icc_dir: Path,
    analysis_set: str,
    analysis: str,
    patterns_file: Path,
    max_voxels: int | None,
) -> pd.DataFrame:
    source_by_label, display_by_label = source_and_display_lookup(patterns_file, analysis_set)
    summary_path = icc_dir / 'voxelwise_icc_summary.tsv'
    if not summary_path.exists():
        raise FileNotFoundError(f'Missing voxelwise ICC summary: {summary_path}')
    summary = pd.read_csv(summary_path, sep='\t')
    summary = summary.loc[
        (summary['analysis_set'].astype(str) == analysis_set)
        & (summary['analysis'].astype(str) == analysis)
        & (summary['tissue'].astype(str).isin(['wm', 'gm']))
    ].copy()
    if summary.empty:
        raise RuntimeError(
            f'No voxelwise ICC rows found for analysis_set={analysis_set}, analysis={analysis}'
        )

    mask_data = {
        tissue: np.asanyarray(nib.load(find_mask(icc_dir, tissue)).dataobj).astype(bool)
        for tissue in ('wm', 'gm')
    }

    rows = []
    for row_index, row in summary.reset_index(drop=True).iterrows():
        tissue = str(row['tissue'])
        metric_key = str(row['metric_key'])
        map_file = (
            icc_dir
            / f'metric-{metric_slug(metric_key)}_space-{SPACE}_desc-{analysis}_stat-icc2p1.nii.gz'
        )
        if not map_file.exists():
            print(f'[WARN] Missing ICC map, skipping: {map_file}', file=sys.stderr)
            continue
        values = load_voxel_values(map_file, mask_data[tissue], max_voxels, seed=row_index + 719)
        display = display_by_label.get(metric_key, str(row.get('metric', metric_key)))
        source = str(row.get('source_image', source_by_label.get(metric_key, 'Other')))
        rows.extend(
            {
                'tissue': tissue,
                'metric_key': metric_key,
                'metric': display,
                'source_image': source,
                'icc': value,
            }
            for value in values
        )
    return pd.DataFrame(rows)


def metric_order_for_tissue(summary: pd.DataFrame, tissue: str) -> list[str]:
    tissue_summary = summary.loc[summary['tissue'] == tissue].copy()
    tissue_summary = tissue_summary.sort_values(['median', 'metric'], ascending=[True, True])
    return tissue_summary['metric_key'].tolist()


def draw_interval_panel(
    ax,
    data: pd.DataFrame,
    summary: pd.DataFrame,
    tissue: str,
    xlabel: str,
) -> None:
    order = metric_order_for_tissue(summary, tissue)
    tissue_data = data.loc[data['tissue'] == tissue]
    tissue_summary = summary.loc[summary['tissue'] == tissue].set_index('metric_key')
    if not order:
        ax.text(0.5, 0.5, f'No {TISSUE_NAMES[tissue]} ICC data', ha='center', va='center')
        ax.set_axis_off()
        return

    positions = np.arange(len(order))
    for position, metric_key in zip(positions, order):
        source = tissue_summary.loc[metric_key, 'source_image']
        color = color_for_source(source)
        median = tissue_summary.loc[metric_key, 'median']
        q25 = tissue_summary.loc[metric_key, 'q25']
        q75 = tissue_summary.loc[metric_key, 'q75']
        ax.hlines(position, 0.0, 1.0, color='#ececec', lw=0.8, zorder=0)
        ax.add_patch(
            Rectangle(
                (q25, position - 0.20),
                max(q75 - q25, 0.001),
                0.40,
                facecolor=color,
                edgecolor='#2b2b2b',
                linewidth=0.8,
                alpha=0.86,
                zorder=2,
            )
        )
        ax.plot([median, median], [position - 0.23, position + 0.23], color='white', lw=1.8, zorder=3)
        ax.scatter([median], [position], s=24, facecolor='white', edgecolor='#2b2b2b', zorder=4)
        ax.text(
            -0.40,
            position,
            tissue_summary.loc[metric_key, 'metric'],
            transform=ax.get_yaxis_transform(),
            ha='right',
            va='center',
            fontsize=7.4,
            clip_on=False,
        )
        ax.text(
            -0.02,
            position,
            f"{median:.2f} [{q25:.2f}, {q75:.2f}]",
            transform=ax.get_yaxis_transform(),
            ha='right',
            va='center',
            fontsize=7.4,
            color='#303030',
            clip_on=False,
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
    ax.set_yticklabels([])
    ax.tick_params(axis='y', length=0)
    ax.set_ylim(-0.8, len(order) - 0.2)
    ax.set_xlim(0.0, 1.0)
    ax.set_xlabel(xlabel)
    ax.set_title(TISSUE_NAMES[tissue], loc='left', fontweight='bold')
    ax.grid(False)
    ax.set_box_aspect(1)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


def label_offsets(n_labels: int) -> list[tuple[float, float]]:
    pattern = [
        (0.012, 0.012),
        (0.012, -0.018),
        (-0.055, 0.014),
        (-0.055, -0.02),
        (0.02, 0.035),
        (-0.07, 0.036),
    ]
    return [pattern[index % len(pattern)] for index in range(n_labels)]


def draw_scatter_panel(ax, summary: pd.DataFrame, title: str) -> None:
    wide = summary.pivot_table(
        index='metric_key',
        columns='tissue',
        values='median',
        aggfunc='first',
    )
    meta = summary.drop_duplicates('metric_key').set_index('metric_key')
    # Tissue-specific metrics, such as g-ratio, belong in the violins but not
    # in the WM-vs-GM scatter comparison.
    wide = wide.dropna(subset=['wm', 'gm'])
    if wide.empty:
        ax.text(0.5, 0.5, 'No matched WM/GM metrics', ha='center', va='center')
        ax.set_axis_off()
        return

    x = wide['gm'].to_numpy(float)
    y = wide['wm'].to_numpy(float)
    lower = float(np.nanmin([x.min(), y.min()]))
    upper = 1.0
    identity = np.linspace(lower, upper, 200)
    ax.fill_between(identity, lower, identity, color='#eeeeee', zorder=0)
    ax.plot(identity, identity, color='#8c8c8c', lw=1.0, ls='--', zorder=1)
    for benchmark in BENCHMARKS:
        ax.axvline(benchmark, color='#e0e0e0', lw=0.8, zorder=0)
        ax.axhline(benchmark, color='#e0e0e0', lw=0.8, zorder=0)

    offsets = label_offsets(len(wide))
    for (metric_key, row), (dx, dy) in zip(wide.iterrows(), offsets):
        source = meta.loc[metric_key, 'source_image']
        color = color_for_source(source)
        label = meta.loc[metric_key, 'metric']
        ax.scatter(
            row['gm'],
            row['wm'],
            s=48,
            facecolor=color,
            edgecolor='#2b2b2b',
            linewidth=0.7,
            alpha=0.95,
            zorder=3,
        )
        ax.text(
            row['gm'] + dx,
            row['wm'] + dy,
            label,
            fontsize=7.0,
            color=color,
            ha='left' if dx >= 0 else 'right',
            va='center',
        )
    ax.text(
        lower + 0.12 * (upper - lower),
        upper - 0.16 * (upper - lower),
        'WM ICC > GM ICC',
        color='#6a6a6a',
        fontsize=10,
        fontstyle='italic',
    )
    ax.text(
        upper - 0.34 * (upper - lower),
        lower + 0.10 * (upper - lower),
        'GM ICC > WM ICC',
        color='#6a6a6a',
        fontsize=10,
        fontstyle='italic',
    )
    ax.set_xlim(lower, upper)
    ax.set_ylim(lower, upper)
    ax.set_aspect('equal', adjustable='box')
    ax.set_box_aspect(1)
    ax.set_xlabel(f'Median ICC across GM {title}')
    ax.set_ylabel(f'Median ICC across WM {title}')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


def add_source_legend(fig, data: pd.DataFrame) -> None:
    sources = [source for source in SOURCE_IMAGE_COLORS if source in set(data['source_image'])]
    if not sources:
        return
    handles = [
        Patch(facecolor=color_for_source(source), edgecolor='none', label=source)
        for source in sources
    ]
    benchmark_handle = Line2D([0], [0], color='#c7c7c7', lw=1.0, label='ICC benchmarks')
    fig.legend(
        handles=handles + [benchmark_handle],
        loc='lower center',
        ncol=len(handles) + 1,
        frameon=False,
        title='Source image',
        bbox_to_anchor=(0.5, 0.015),
    )


def plot_icc_figure(
    data: pd.DataFrame,
    output_prefix: Path,
    title: str,
    scatter_domain: str,
) -> None:
    if data.empty:
        raise RuntimeError(f'No ICC data available for {title}')
    data = data.copy()
    data['icc'] = pd.to_numeric(data['icc'], errors='coerce')
    data = data.dropna(subset=['icc'])
    summary = summarize_metric_values(data)
    if summary.empty:
        raise RuntimeError(f'No finite ICC values available for {title}')

    mpl.rcParams.update(
        {
            'font.family': 'Arial',
            'pdf.fonttype': 42,
            'ps.fonttype': 42,
            'axes.titlesize': 13,
            'axes.labelsize': 11,
            'xtick.labelsize': 9,
            'ytick.labelsize': 8.5,
        }
    )
    fig = plt.figure(figsize=(10.5, 24.0), constrained_layout=False)
    grid = fig.add_gridspec(
        3,
        1,
        height_ratios=[1.0, 1.0, 1.0],
        left=0.38,
        right=0.96,
        bottom=0.105,
        top=0.985,
        hspace=0.28,
    )
    ax_wm = fig.add_subplot(grid[0, 0])
    ax_gm = fig.add_subplot(grid[1, 0])
    ax_scatter = fig.add_subplot(grid[2, 0])
    draw_interval_panel(ax_wm, data, summary, 'wm', f'ICC(2,1) across WM {scatter_domain}')
    draw_interval_panel(ax_gm, data, summary, 'gm', f'ICC(2,1) across GM {scatter_domain}')
    draw_scatter_panel(ax_scatter, summary, scatter_domain)
    for label, ax in zip(('A', 'B', 'C'), (ax_wm, ax_gm, ax_scatter)):
        ax.text(
            -0.18,
            1.03,
            label,
            transform=ax.transAxes,
            fontsize=17,
            fontweight='bold',
            ha='left',
            va='bottom',
            clip_on=False,
        )
    add_source_legend(fig, data)

    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    for ext in ('pdf', 'png'):
        out_file = output_prefix.with_suffix(f'.{ext}')
        fig.savefig(out_file, dpi=300)
        print(f'Wrote: {out_file}', flush=True)
    plt.close(fig)

    summary_out = output_prefix.with_suffix('.summary.tsv')
    summary.sort_values(['tissue', 'median', 'metric']).to_csv(summary_out, sep='\t', index=False)
    print(f'Wrote: {summary_out}', flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--analysis-set', choices=('primary', 'full'), default='primary')
    parser.add_argument('--stat', choices=('mean', 'median'), default='median')
    parser.add_argument('--mni-icc-dir', type=Path, default=default_mni_icc_dir())
    parser.add_argument('--parcel-bundle-icc-dir', type=Path, default=default_parcel_bundle_icc_dir())
    parser.add_argument('--patterns-file', type=Path, default=CODE_ROOT / 'configuration' / 'patterns.json')
    parser.add_argument('--output-dir', type=Path, default=PROJECT_ROOT / 'figures' / 'icc')
    parser.add_argument('--voxelwise-analysis', default='primary')
    parser.add_argument('--max-voxels-per-metric', type=int, default=None)
    parser.add_argument('--skip-voxelwise', action='store_true')
    parser.add_argument('--skip-parcel-bundle', action='store_true')
    return parser


def main() -> None:
    args = build_parser().parse_args()
    require_dependencies(need_nibabel=not args.skip_voxelwise)

    if not args.skip_parcel_bundle:
        parcel_data = load_parcel_bundle_icc(
            args.parcel_bundle_icc_dir,
            args.analysis_set,
            args.stat,
            args.patterns_file,
        )
        plot_icc_figure(
            parcel_data,
            args.output_dir / f'icc_parcel_bundle_{args.analysis_set}_{args.stat}',
            f'{args.analysis_set.title()} Parcel/Bundle ICC',
            'parcels/bundles',
        )

    if not args.skip_voxelwise:
        voxel_data = load_voxelwise_icc(
            args.mni_icc_dir,
            args.analysis_set,
            args.voxelwise_analysis,
            args.patterns_file,
            args.max_voxels_per_metric,
        )
        plot_icc_figure(
            voxel_data,
            args.output_dir / f'icc_mni_voxelwise_{args.analysis_set}_{args.voxelwise_analysis}',
            f'{args.analysis_set.title()} MNI Voxelwise ICC',
            'voxels',
        )


if __name__ == '__main__':
    main()
