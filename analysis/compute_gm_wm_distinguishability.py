#!/usr/bin/env python3
"""Estimate how distinct each selected metric is between GM and WM."""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path

import matplotlib as mpl

mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import rankdata, wilcoxon

from compute_icc_from_bundle_stats import apply_complete_qc as apply_wm_complete_qc
from compute_icc_from_bundle_stats import apply_metric_qc as apply_wm_metric_qc
from compute_icc_from_bundle_stats import collect_scalarstats, load_qc_table
from compute_icc_from_dkt_stats import apply_complete_qc as apply_gm_complete_qc
from compute_icc_from_dkt_stats import apply_metric_qc as apply_gm_metric_qc
from compute_icc_from_dkt_stats import build_value_table, collect_rows


PROJECT_ROOT = Path('/cbica/projects/nibs')
DEFAULT_QC_FILE = Path(__file__).resolve().parents[1] / 'data' / 'manual_qc_modality.tsv'
DEFAULT_WM_GLOBS = [
    '/cbica/projects/nibs/derivatives/qsirecon/derivatives/qsirecon-*/sub-*/ses-*/dwi/sub-*_ses-*_*_scalarstats.tsv',
    '/cbica/projects/nibs/derivatives/bundle_myelin_stats/sub-*/ses-*/dwi/sub-*_ses-*_acq-HBCD75_run-01_space-T1w_model-*_scalarstats.tsv',
]
DEFAULT_DKT_GLOBS = [
    '/cbica/projects/nibs/derivatives/DKTatlas_myelin_stats/sub-*/sub-*_ses-*_run-*_desc-DKTatlas_scalarstats.csv'
]
QC_MODES = ('metricqc', 'completeqc')


@dataclass(frozen=True)
class MetricSpec:
    key: str
    label: str
    family: str
    aliases: tuple[str, ...]


SELECTED_METRICS = [
    MetricSpec('tortoise_fa', 'FA', 'Tensor', ('TORTOISE-InnerShell-FA',)),
    MetricSpec('tortoise_md', 'MD', 'Tensor', ('TORTOISE-InnerShell-MD',)),
    MetricSpec('tortoise_rd', 'RD', 'Tensor', ('TORTOISE-InnerShell-RD',)),
    MetricSpec('noddi_icvf', 'ICVF', 'NODDI', ('NODDI-ICVF',)),
    MetricSpec('dki_mkt', 'MKT', 'DKI', ('DKI-MKT',)),
    MetricSpec('dki_rk', 'RK', 'DKI', ('DKI-RK',)),
    MetricSpec('mapmri_rtop', 'RTOP', 'MAPMRI', ('MAPMRI-RTOP',)),
    MetricSpec('mapmri_rtap', 'RTAP', 'MAPMRI', ('MAPMRI-RTAP',)),
    MetricSpec('mapmri_ng', 'NG', 'MAPMRI', ('MAPMRI-NG',)),
    MetricSpec('gqi_gfa', 'GFA', 'GQI', ('DSIStudio-GQI-GFA',)),
    MetricSpec(
        'qsm_dia',
        'QSM-X-R2p-E5-Dia',
        'QSM',
        ('QSM-X-R2p-E5-Dia', "QSM-X-R2'-E5-Dia"),
    ),
    MetricSpec(
        'qsm_para',
        'QSM-X-R2p-E5-Para',
        'QSM',
        ('QSM-X-R2p-E5-Para', "QSM-X-R2'-E5-Para"),
    ),
    MetricSpec(
        'qsm_x',
        'QSM-X-R2p-E5-X',
        'QSM',
        ('QSM-X-R2p-E5-X', "QSM-X-R2'-E5-X"),
    ),
    MetricSpec(
        'qsm_e5',
        'QSM-SEPIA-E5',
        'QSM',
        ('QSM-SEPIA-E5', 'QSM-X-R2p-E5', "QSM-X-R2'-E5"),
    ),
    MetricSpec(
        'scaled_mprage_myelinw',
        'Scaled MPRAGE-MyelinW',
        'T1w/T2w',
        ('Scaled MPRAGE-MyelinW',),
    ),
    MetricSpec(
        'scaled_space_myelinw',
        'Scaled SPACE-MyelinW',
        'T1w/T2w',
        ('Scaled SPACE-MyelinW', 'Scaled Space-MyelinW'),
    ),
    MetricSpec('ihmtr', 'ihMTR', 'ihMTR', ('ihMTR',)),
    MetricSpec('ihmtsat_b1c', 'ihMTsat-B1c', 'ihMTR', ('ihMTsat-B1c',)),
    MetricSpec('r1_b1c', 'R1-B1c', 'R1', ('R1-B1c',)),
    MetricSpec('r1', 'R1', 'R1', ('R1',)),
]

SOURCE_IMAGE_COLORS = {
    'DWI': '#4477AA',
    'QSM': '#AA3377',
    'T1w/T2w': '#CCBB44',
    'ihMT': '#228833',
    'R1': '#EE7733',
    'Other': '#999999',
}
SOURCE_IMAGE_ORDER = {source: index for index, source in enumerate(SOURCE_IMAGE_COLORS)}
METRIC_FAMILY_LEGEND_TITLE = 'Metric family'
SOURCE_IMAGE_DISPLAY_LABELS = {
    'T1w/T2w': 'T₁w/T₂w',
    'R1': 'MP2RAGE',
}


def source_image_display_label(source: str) -> str:
    return SOURCE_IMAGE_DISPLAY_LABELS.get(source, source)


def norm_token(text: object) -> str:
    return re.sub(r'[^a-z0-9]+', '', str(text).lower())


def source_image_from_family(family: object) -> str:
    family_label = str(family)
    if family_label in {'Tensor', 'NODDI', 'DKI', 'MAPMRI', 'GQI'}:
        return 'DWI'
    if family_label == 'ihMTR':
        return 'ihMT'
    if family_label in SOURCE_IMAGE_COLORS:
        return family_label
    return 'Other'


def shade_color(base_color: str, amount: float) -> str:
    rgb = np.asarray(mpl.colors.to_rgb(base_color), dtype=float)
    if amount >= 0:
        shaded = rgb + (1.0 - rgb) * amount
    else:
        shaded = rgb * (1.0 + amount)
    return mpl.colors.to_hex(np.clip(shaded, 0.0, 1.0))


def build_alias_map() -> dict[str, MetricSpec]:
    aliases: dict[str, MetricSpec] = {}
    for spec in SELECTED_METRICS:
        for alias in (spec.key, spec.label, *spec.aliases):
            aliases[norm_token(alias)] = spec
        if spec.label.startswith('QSM-X-R2p'):
            aliases[norm_token(spec.label.replace('R2p', 'R2'))] = spec
            aliases[norm_token(spec.label.replace('R2p', "R2'"))] = spec
    return aliases


ALIAS_MAP = build_alias_map()
SPEC_BY_LABEL = {spec.label: spec for spec in SELECTED_METRICS}


def metric_spec(metric: object) -> MetricSpec | None:
    return ALIAS_MAP.get(norm_token(metric))


def build_metric_colors() -> dict[str, str]:
    colors: dict[str, str] = {}
    for source_image, base_color in SOURCE_IMAGE_COLORS.items():
        source_specs = [
            spec
            for spec in SELECTED_METRICS
            if source_image_from_family(spec.family) == source_image
        ]
        shade_amounts = (
            [0.0] if len(source_specs) == 1 else np.linspace(0.42, -0.28, len(source_specs))
        )
        for spec, amount in zip(source_specs, shade_amounts):
            colors[spec.label] = shade_color(base_color, float(amount))
    return colors


METRIC_COLORS = build_metric_colors()


def apply_qc_mode(df: pd.DataFrame, qc: pd.DataFrame, qc_mode: str, tissue: str) -> pd.DataFrame:
    if tissue == 'wm':
        if qc_mode == 'metricqc':
            return apply_wm_metric_qc(df, qc)
        if qc_mode == 'completeqc':
            return apply_wm_complete_qc(df, qc)
    if tissue == 'gm':
        if qc_mode == 'metricqc':
            return apply_gm_metric_qc(df, qc)
        if qc_mode == 'completeqc':
            return apply_gm_complete_qc(df, qc)
    raise ValueError(f'Unsupported tissue/qc_mode: {tissue}/{qc_mode}')


def selected_wm_profiles(input_globs: list[str], qc: pd.DataFrame, qc_mode: str) -> pd.DataFrame:
    wm = collect_scalarstats(input_globs)
    if wm.empty:
        raise RuntimeError(f'No WM scalarstats matched: {input_globs}')
    wm = apply_qc_mode(wm, qc, qc_mode, tissue='wm')
    wm['value'] = pd.to_numeric(wm['masked_median'], errors='coerce')
    wm['spec'] = wm['metric'].map(metric_spec)
    wm = wm[wm['spec'].notna() & np.isfinite(wm['value'].to_numpy(dtype=float))].copy()
    wm['metric_key'] = wm['spec'].map(lambda spec: spec.key)
    wm['metric_label'] = wm['spec'].map(lambda spec: spec.label)
    wm['family'] = wm['spec'].map(lambda spec: spec.family)
    return (
        wm.groupby(
            ['subject_id', 'session_id', 'metric_key', 'metric_label', 'family'], as_index=False
        )['value']
        .median()
        .rename(columns={'subject_id': 'subject', 'session_id': 'session', 'value': 'wm_value'})
    )


def selected_gm_profiles(input_globs: list[str], qc: pd.DataFrame, qc_mode: str) -> pd.DataFrame:
    tables = [collect_rows(pattern) for pattern in input_globs]
    nonempty_tables = [table for table in tables if not table.empty]
    if not nonempty_tables:
        raise RuntimeError(f'No DKT scalarstats matched: {input_globs}')
    gm_rows = pd.concat(nonempty_tables, ignore_index=True)
    gm = build_value_table(gm_rows, stat='median')
    gm = apply_qc_mode(gm, qc, qc_mode, tissue='gm')
    gm['value'] = pd.to_numeric(gm['value'], errors='coerce')
    gm['spec'] = gm['metric'].map(metric_spec)
    gm = gm[gm['spec'].notna() & np.isfinite(gm['value'].to_numpy(dtype=float))].copy()
    gm['metric_key'] = gm['spec'].map(lambda spec: spec.key)
    gm['metric_label'] = gm['spec'].map(lambda spec: spec.label)
    gm['family'] = gm['spec'].map(lambda spec: spec.family)
    return (
        gm.groupby(['subject', 'session', 'metric_key', 'metric_label', 'family'], as_index=False)[
            'value'
        ]
        .median()
        .rename(columns={'value': 'gm_value'})
    )


def paired_rank_biserial(diffs: np.ndarray) -> float:
    nonzero = diffs[np.isfinite(diffs) & (diffs != 0)]
    if nonzero.size == 0:
        return np.nan
    ranks = rankdata(np.abs(nonzero), method='average')
    rank_sum_positive = float(ranks[nonzero > 0].sum())
    rank_sum_negative = float(ranks[nonzero < 0].sum())
    denom = float(ranks.sum())
    if denom == 0:
        return np.nan
    return (rank_sum_positive - rank_sum_negative) / denom


def bootstrap_ci(values: np.ndarray, func, n_boot: int, seed: int) -> tuple[float, float]:
    clean = values[np.isfinite(values)]
    if clean.size < 2 or n_boot <= 0:
        return (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    estimates = np.empty(n_boot, dtype=float)
    for index in range(n_boot):
        sample = rng.choice(clean, size=clean.size, replace=True)
        estimates[index] = func(sample)
    estimates = estimates[np.isfinite(estimates)]
    if estimates.size == 0:
        return (np.nan, np.nan)
    low, high = np.percentile(estimates, [2.5, 97.5])
    return float(low), float(high)


def fdr_bh(p_values: pd.Series) -> pd.Series:
    p = pd.to_numeric(p_values, errors='coerce')
    out = pd.Series(np.nan, index=p.index, dtype=float)
    valid = p.dropna().sort_values()
    m = len(valid)
    if m == 0:
        return out
    adjusted = (valid.to_numpy() * m / np.arange(1, m + 1))[::-1]
    adjusted = np.minimum.accumulate(adjusted)[::-1]
    out.loc[valid.index] = np.clip(adjusted, 0.0, 1.0)
    return out


def compute_summary(paired: pd.DataFrame, n_boot: int, seed: int) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for metric_label, dfg in paired.groupby('metric_label', sort=False):
        spec = SPEC_BY_LABEL[metric_label]
        diffs = dfg['gm_minus_wm'].to_numpy(dtype=float)
        nonzero = diffs[np.isfinite(diffs) & (diffs != 0)]
        try:
            p_value = (
                float(wilcoxon(nonzero, zero_method='wilcox', alternative='two-sided').pvalue)
                if nonzero.size
                else np.nan
            )
        except ValueError:
            p_value = np.nan

        rb = paired_rank_biserial(diffs)
        rb_low, rb_high = bootstrap_ci(diffs, paired_rank_biserial, n_boot=n_boot, seed=seed)
        diff_low, diff_high = bootstrap_ci(diffs, np.median, n_boot=n_boot, seed=seed + 1)
        rows.append(
            {
                'metric_key': spec.key,
                'metric_label': spec.label,
                'family': spec.family,
                'source_image': source_image_from_family(spec.family),
                'n_profiles': int(len(dfg)),
                'n_subjects': int(dfg['subject'].nunique()),
                'gm_median': float(np.nanmedian(dfg['gm_value'])),
                'wm_median': float(np.nanmedian(dfg['wm_value'])),
                'median_gm_minus_wm': float(np.nanmedian(diffs)),
                'median_gm_minus_wm_ci95_low': diff_low,
                'median_gm_minus_wm_ci95_high': diff_high,
                'paired_rank_biserial': rb,
                'paired_rank_biserial_ci95_low': rb_low,
                'paired_rank_biserial_ci95_high': rb_high,
                'wilcoxon_p': p_value,
            }
        )
    out = pd.DataFrame(rows)
    out['wilcoxon_p_fdr'] = fdr_bh(out['wilcoxon_p'])
    out['abs_paired_rank_biserial'] = out['paired_rank_biserial'].abs()
    return out.sort_values('paired_rank_biserial').reset_index(drop=True)


def plot_effect_sizes(summary: pd.DataFrame, out_prefix: Path) -> None:
    plot_data = summary.copy()
    plot_data['source_order'] = (
        plot_data['source_image'].map(SOURCE_IMAGE_ORDER).fillna(len(SOURCE_IMAGE_ORDER))
    )
    plot_data = plot_data.sort_values(
        ['paired_rank_biserial', 'source_order', 'metric_label']
    ).reset_index(drop=True)

    y = np.arange(len(plot_data))
    colors = [
        METRIC_COLORS.get(label, SOURCE_IMAGE_COLORS['Other'])
        for label in plot_data['metric_label']
    ]
    fig_height = max(7.0, 0.38 * len(plot_data) + 2.3)
    fig, ax = plt.subplots(figsize=(10.5, fig_height), constrained_layout=True)
    ax.axvline(0, color='#777777', linestyle=':', linewidth=1.0, zorder=0)
    for index, row in plot_data.iterrows():
        low = row['paired_rank_biserial_ci95_low']
        high = row['paired_rank_biserial_ci95_high']
        value = row['paired_rank_biserial']
        if np.isfinite(low) and np.isfinite(high):
            ax.hlines(index, low, high, color='#222222', linewidth=1.5, zorder=2)
        ax.scatter(
            value, index, s=52, color=colors[index], edgecolor='black', linewidth=0.45, zorder=3
        )

    ax.set_yticks(y)
    ax.set_yticklabels(plot_data['metric_label'], fontsize=8)
    ax.set_xlim(-1.03, 1.03)
    ax.set_xlabel('Paired rank-biserial effect size for GM - WM')
    ax.set_ylabel('')
    ax.grid(axis='x', alpha=0.22, linewidth=0.6)
    ax.grid(axis='y', visible=False)
    ax.text(-0.98, -0.75, 'WM > GM', ha='left', va='center', fontsize=10, color='#555555')
    ax.text(0.98, -0.75, 'GM > WM', ha='right', va='center', fontsize=10, color='#555555')

    used_sources = [
        source for source in SOURCE_IMAGE_COLORS if source in set(plot_data['source_image'])
    ]
    handles = [
        mpl.lines.Line2D(
            [0],
            [0],
            marker='o',
            linestyle='none',
            markersize=7,
            markerfacecolor=SOURCE_IMAGE_COLORS[source],
            markeredgecolor='black',
            markeredgewidth=0.45,
            label=source_image_display_label(source),
        )
        for source in used_sources
    ]
    ax.legend(
        handles=handles,
        title=METRIC_FAMILY_LEGEND_TITLE,
        loc='center left',
        bbox_to_anchor=(1.02, 0.5),
        frameon=False,
    )

    for extension in ('png', 'pdf'):
        out_path = out_prefix.with_suffix(f'.{extension}')
        fig.savefig(out_path, bbox_inches='tight')
        print(f'Wrote: {out_path}')
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--qc-file', type=Path, default=DEFAULT_QC_FILE)
    parser.add_argument('--qc-mode', nargs='+', choices=QC_MODES, default=list(QC_MODES))
    parser.add_argument('--wm-input-globs', nargs='+', default=DEFAULT_WM_GLOBS)
    parser.add_argument('--dkt-input-glob', nargs='+', default=DEFAULT_DKT_GLOBS)
    parser.add_argument('--outdir', type=Path, default=PROJECT_ROOT / 'derivatives' / 'ICC')
    parser.add_argument('--n-bootstrap', type=int, default=10000)
    parser.add_argument('--seed', type=int, default=20260804)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    qc = load_qc_table(args.qc_file)

    for qc_mode in args.qc_mode:
        wm_profiles = selected_wm_profiles(args.wm_input_globs, qc, qc_mode)
        gm_profiles = selected_gm_profiles(args.dkt_input_glob, qc, qc_mode)
        paired = pd.merge(
            gm_profiles,
            wm_profiles[['subject', 'session', 'metric_key', 'wm_value']],
            on=['subject', 'session', 'metric_key'],
            how='inner',
        )
        paired['gm_minus_wm'] = paired['gm_value'] - paired['wm_value']
        paired = paired[np.isfinite(paired['gm_minus_wm'].to_numpy(dtype=float))].copy()
        if paired.empty:
            raise RuntimeError(f'No paired GM/WM profiles remained for {qc_mode}')

        summary = compute_summary(paired, n_boot=args.n_bootstrap, seed=args.seed)
        paired_out = args.outdir / f'gm_wm_distinguishability_profiles_{qc_mode}.csv'
        summary_out = args.outdir / f'gm_wm_distinguishability_summary_{qc_mode}.csv'
        paired.to_csv(paired_out, index=False)
        summary.to_csv(summary_out, index=False)
        print(f'Wrote: {paired_out}')
        print(f'Wrote: {summary_out}')
        plot_effect_sizes(
            summary,
            args.outdir / f'gm_wm_distinguishability_effect_sizes_{qc_mode}',
        )


if __name__ == '__main__':
    main()
