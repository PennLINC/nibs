#!/usr/bin/env python3
"""Compute parcel-wise ICC from DKT per-run scalar summary CSVs."""

from __future__ import annotations

import argparse
import re
from glob import glob
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

try:
    import pingouin as pg

    HAVE_PINGOUIN = True
except Exception:
    HAVE_PINGOUIN = False


FILE_RE = re.compile(r'sub-(?P<sub>[^_]+)_(?P<ses>ses-[^_]+)_(?P<run>run-[^_]+)_')
DEFAULT_QC_FILE = Path(__file__).resolve().parents[1] / 'data' / 'manual_qc_modality.tsv'
DWI_METRIC_PREFIXES = (
    'DKI-',
    'DSIStudio-',
    'MAPMRI-',
    'NODDI-',
    'TORTOISE-',
)
QC_MODES = ('metricqc', 'completeqc')
EXCLUDED_DKT_METRICS = {'G-ihMTsat', 'G-ihMTR'}


def metric_required_modalities(metric: str) -> tuple[str, ...]:
    """Return scan-level QC modalities required to trust a derived metric."""
    if metric.startswith(DWI_METRIC_PREFIXES):
        return ('dMRI',)
    if metric == 'QSM-SEPIA-E5' or metric == 'MEGRE':
        return ('MEGRE',)
    if metric.startswith('QSM-X-R2'):
        return ('MEGRE', 'MESE')
    if metric in {'ihMTw', 'ihMTR', 'MTR'}:
        return ('ihMTRAGE',)
    if metric in {'ihMTsat', 'ihMTsat-B1c'}:
        return ('MP2RAGE', 'ihMTRAGE', 'B1+')
    if metric == 'R1':
        return ('MP2RAGE',)
    if metric == 'R1-B1c':
        return ('MP2RAGE', 'B1+')
    if metric in {'MPRAGE-MyelinW', 'Scaled MPRAGE-MyelinW'}:
        return ('MPRAGE T1w', 'SPACE T2w')
    if metric in {'SPACE-MyelinW', 'Scaled SPACE-MyelinW'}:
        return ('SPACE T1w', 'SPACE T2w')
    if metric == 'G-ihMTR':
        return ('dMRI', 'ihMTRAGE')
    if metric == 'G-ihMTsat':
        return ('MP2RAGE', 'dMRI', 'ihMTRAGE', 'B1+')
    raise ValueError(f'No QC modality mapping defined for metric: {metric}')


def _normalize_subject(value: object) -> str:
    return re.sub(r'^sub-', '', str(value).strip())


def _is_pilot_subject(value: object) -> bool:
    return _normalize_subject(value).upper().startswith('PILOT')


def _session_label(value: object) -> str:
    match = re.search(r'(\d+)', str(value))
    if not match:
        raise ValueError(f'Could not parse session number from: {value}')
    return f'Session {int(match.group(1)):02d}'


def load_qc_table(qc_file: Path) -> pd.DataFrame:
    qc_df = pd.read_csv(qc_file, sep='\t')
    if 'participant_id' not in qc_df.columns:
        raise RuntimeError(f'{qc_file} is missing participant_id')
    qc_df = qc_df.copy()
    qc_df['participant_id'] = qc_df['participant_id'].map(_normalize_subject)
    qc_df = qc_df.loc[~qc_df['participant_id'].map(_is_pilot_subject)].copy()
    return qc_df.set_index('participant_id', drop=False)


def _qc_passes(
    qc_df: pd.DataFrame,
    subject: object,
    session: object,
    modalities: tuple[str, ...],
) -> bool:
    subject_id = _normalize_subject(subject)
    if subject_id not in qc_df.index:
        return False
    session_prefix = _session_label(session)
    row = qc_df.loc[subject_id]
    for modality in modalities:
        column = f'{session_prefix}--{modality}'
        if column not in qc_df.columns:
            raise RuntimeError(f'QC file is missing required column: {column}')
        value = row[column]
        if pd.isna(value) or int(value) != 1:
            return False
    return True


def apply_metric_qc(
    value_df: pd.DataFrame,
    qc_df: pd.DataFrame,
    subject_col: str = 'subject',
    session_col: str = 'session',
) -> pd.DataFrame:
    keep = [
        _qc_passes(
            qc_df,
            row[subject_col],
            row[session_col],
            metric_required_modalities(str(row['metric'])),
        )
        for _, row in value_df.iterrows()
    ]
    return value_df.loc[keep].copy()


def subjects_with_complete_qc(
    value_df: pd.DataFrame,
    qc_df: pd.DataFrame,
    subject_col: str = 'subject',
) -> set[str]:
    modalities = sorted(
        {
            modality
            for metric in value_df['metric'].dropna().astype(str).unique()
            for modality in metric_required_modalities(metric)
        }
    )
    subjects = sorted({_normalize_subject(value) for value in value_df[subject_col].unique()})
    complete_subjects: set[str] = set()
    for subject in subjects:
        if all(
            _qc_passes(qc_df, subject, f'ses-{session:02d}', tuple(modalities))
            for session in (1, 2)
        ):
            complete_subjects.add(subject)
    return complete_subjects


def apply_complete_qc(
    value_df: pd.DataFrame,
    qc_df: pd.DataFrame,
    subject_col: str = 'subject',
) -> pd.DataFrame:
    complete_subjects = subjects_with_complete_qc(value_df, qc_df, subject_col=subject_col)
    subjects = value_df[subject_col].map(_normalize_subject)
    return value_df.loc[subjects.isin(complete_subjects)].copy()


def compute_icc2_fallback(values: np.ndarray, subjects: np.ndarray, sessions: np.ndarray) -> float:
    """Compute ICC(2,1) using a two-way random effects fallback."""
    subs_unique, sub_idx = np.unique(subjects, return_inverse=True)
    sessions_unique, sess_idx = np.unique(sessions, return_inverse=True)
    n_sub, n_ses = len(subs_unique), len(sessions_unique)
    if n_sub < 2 or n_ses < 2:
        return np.nan

    matrix = np.full((n_sub, n_ses), np.nan, dtype=float)
    for val, i_sub, i_ses in zip(values, sub_idx, sess_idx):
        matrix[i_sub, i_ses] = val

    # Complete-case subjects only.
    matrix = matrix[~np.any(np.isnan(matrix), axis=1)]
    if matrix.shape[0] < 2:
        return np.nan

    n_sub = matrix.shape[0]
    grand_mean = matrix.mean()
    row_means = matrix.mean(axis=1)
    col_means = matrix.mean(axis=0)

    ssr = n_ses * np.sum((row_means - grand_mean) ** 2)
    ssc = n_sub * np.sum((col_means - grand_mean) ** 2)
    sse = np.sum((matrix - grand_mean) ** 2) - ssr - ssc

    msr = ssr / (n_sub - 1)
    msc = ssc / (n_ses - 1)
    mse = sse / ((n_sub - 1) * (n_ses - 1))

    denom = msr + (n_ses - 1) * mse + n_ses * (msc - mse) / n_sub
    if denom == 0:
        return np.nan
    return float((msr - mse) / denom)


def parse_filename(path: Path) -> tuple[str, str, str]:
    match = FILE_RE.search(path.name)
    if not match:
        raise ValueError(f'Could not parse subject/session/run from: {path}')
    return match.group('sub'), match.group('ses'), match.group('run')


def collect_rows(input_glob: str) -> pd.DataFrame:
    records: list[pd.DataFrame] = []
    for file_str in sorted(glob(input_glob)):
        path = Path(file_str)
        subject, session, run = parse_filename(path)
        if _is_pilot_subject(subject):
            continue
        df = pd.read_csv(path)
        df['subject'] = subject
        df['session'] = session
        df['run'] = run
        records.append(df)
    if not records:
        return pd.DataFrame()
    return pd.concat(records, ignore_index=True)


def build_value_table(df: pd.DataFrame, stat: str) -> pd.DataFrame:
    base_cols = [
        'subject',
        'session',
        'run',
        'parcel_intensity',
        'parcel_name',
        'parcel_hemi',
        'parcel_count_t1w',
        'parcel_count_acpc',
    ]
    metric_cols = [col for col in df.columns if col.endswith(f'_{stat}')]
    if not metric_cols:
        raise RuntimeError(f'No columns found ending with _{stat}')

    value_df = df[base_cols + metric_cols].melt(
        id_vars=base_cols,
        value_vars=metric_cols,
        var_name='metric_stat',
        value_name='value',
    )
    value_df['metric'] = value_df['metric_stat'].str[: -(len(stat) + 1)]
    value_df['parcel'] = (
        value_df['parcel_hemi'].astype(str) + '_' + value_df['parcel_name'].astype(str)
    )
    value_df = value_df[~value_df['metric'].isin(EXCLUDED_DKT_METRICS)].copy()
    return value_df.drop(columns=['metric_stat'])


def collapse_subject_session_values(value_df: pd.DataFrame) -> pd.DataFrame:
    group_cols = [
        'subject',
        'session',
        'metric',
        'parcel',
        'parcel_intensity',
        'parcel_hemi',
    ]
    collapsed = (
        value_df[group_cols + ['value']]
        .dropna(subset=['value'])
        .groupby(group_cols, as_index=False)['value']
        .mean()
    )
    return collapsed


def compute_icc_table(value_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    value_df = collapse_subject_session_values(value_df)
    grouped = value_df.groupby(['metric', 'parcel', 'parcel_intensity', 'parcel_hemi'], sort=True)
    for (metric, parcel, parcel_intensity, parcel_hemi), dfg in grouped:
        dfg = dfg[np.isfinite(dfg['value'].to_numpy())].copy()
        if dfg.empty:
            continue

        session_counts = dfg.groupby('subject')['session'].nunique()
        valid_subjects = session_counts[session_counts >= 2].index
        dfg = dfg[dfg['subject'].isin(valid_subjects)]
        if dfg['subject'].nunique() < 2 or dfg['session'].nunique() < 2:
            continue

        subjects = dfg['subject'].to_numpy()
        sessions = dfg['session'].to_numpy()
        values = dfg['value'].to_numpy(dtype=float)

        icc_val = np.nan
        ci95 = None
        f_val = np.nan
        df1 = np.nan
        df2 = np.nan
        pval = np.nan

        if HAVE_PINGOUIN:
            try:
                tab = pd.DataFrame(
                    {
                        'targets': subjects,
                        'raters': sessions,
                        'scores': values,
                    }
                )
                icc_tab = pg.intraclass_corr(
                    data=tab,
                    targets='targets',
                    raters='raters',
                    ratings='scores',
                )
                icc_row = icc_tab.query("Type == 'ICC2'").iloc[0]
                icc_val = float(icc_row['ICC'])
                ci95 = str(icc_row.get('CI95%', ''))
                f_val = float(icc_row.get('F', np.nan))
                df1 = float(icc_row.get('df1', np.nan))
                df2 = float(icc_row.get('df2', np.nan))
                pval = float(icc_row.get('pval', np.nan))
            except Exception:
                icc_val = compute_icc2_fallback(values, subjects, sessions)
        else:
            icc_val = compute_icc2_fallback(values, subjects, sessions)

        rows.append(
            {
                'metric': metric,
                'parcel': parcel,
                'parcel_intensity': int(parcel_intensity),
                'parcel_hemi': parcel_hemi,
                'ICC2_1': icc_val,
                'CI95': ci95,
                'F': f_val,
                'df1': df1,
                'df2': df2,
                'pval': pval,
                'n_subjects': int(dfg['subject'].nunique()),
                'n_sessions': int(dfg['session'].nunique()),
            }
        )
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(['metric', 'parcel']).reset_index(drop=True)


def compute_icc_diagnostics(value_df: pd.DataFrame, icc_df: pd.DataFrame) -> pd.DataFrame:
    """Summarize why each metric-parcel did or did not yield an ICC value."""
    value_df = collapse_subject_session_values(value_df)
    icc_lookup = {}
    if not icc_df.empty:
        icc_lookup = {
            (row.metric, row.parcel, int(row.parcel_intensity), row.parcel_hemi): row.ICC2_1
            for row in icc_df.itertuples(index=False)
        }

    rows: list[dict[str, object]] = []
    grouped = value_df.groupby(['metric', 'parcel', 'parcel_intensity', 'parcel_hemi'], sort=True)
    for (metric, parcel, parcel_intensity, parcel_hemi), dfg in grouped:
        finite_mask = np.isfinite(dfg['value'].to_numpy(dtype=float))
        finite = dfg.loc[finite_mask].copy()

        session_counts = finite.groupby('subject')['session'].nunique()
        paired_subjects = session_counts[session_counts >= 2].index
        paired = finite[finite['subject'].isin(paired_subjects)]
        values = paired['value'].to_numpy(dtype=float)
        has_values = values.size > 0

        key = (metric, parcel, int(parcel_intensity), parcel_hemi)
        icc_value = icc_lookup.get(key, np.nan)
        n_unique_values = int(pd.Series(values).nunique(dropna=True)) if has_values else 0

        if finite.empty:
            reason = 'no_finite_values'
        elif len(paired_subjects) < 2:
            reason = 'fewer_than_2_paired_subjects'
        elif key not in icc_lookup:
            reason = 'not_output_by_icc_table'
        elif pd.isna(icc_value) and n_unique_values <= 1:
            reason = 'constant_paired_values'
        elif pd.isna(icc_value):
            reason = 'icc_formula_returned_nan'
        else:
            reason = 'ok'

        rows.append(
            {
                'metric': metric,
                'parcel': parcel,
                'parcel_intensity': int(parcel_intensity),
                'parcel_hemi': parcel_hemi,
                'diagnostic': reason,
                'ICC2_1': icc_value,
                'n_input_rows': int(len(dfg)),
                'n_finite_rows': int(len(finite)),
                'n_subjects_finite': int(finite['subject'].nunique()),
                'n_paired_subjects': int(len(paired_subjects)),
                'n_sessions_finite': int(finite['session'].nunique()),
                'n_unique_paired_values': n_unique_values,
                'paired_value_min': float(np.min(values)) if has_values else np.nan,
                'paired_value_median': float(np.median(values)) if has_values else np.nan,
                'paired_value_max': float(np.max(values)) if has_values else np.nan,
                'paired_value_std': float(np.std(values)) if has_values else np.nan,
            }
        )

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(['metric', 'parcel']).reset_index(drop=True)


def plot_heatmap(df_icc: pd.DataFrame, out_file: Path) -> None:
    pivot = df_icc.pivot(index='metric', columns='parcel', values='ICC2_1')
    nan_counts = pivot.isna().sum(axis=1)
    nan_counts = nan_counts[nan_counts > 0].sort_values(ascending=False)
    if not nan_counts.empty:
        print('[WARN] NaN ICC cells by metric:', flush=True)
        for metric, count in nan_counts.items():
            print(f'  {metric}: {int(count)} parcel(s)', flush=True)
    row_order = pivot.mean(axis=1).sort_values(ascending=False).index
    col_order = pivot.mean(axis=0).sort_values(ascending=False).index
    pivot = pivot.loc[row_order, col_order]

    fig_width = max(12, 0.24 * len(pivot.columns))
    fig_height = max(6, 0.3 * len(pivot.index))
    plt.figure(figsize=(fig_width, fig_height))
    im = plt.imshow(pivot.to_numpy(), aspect='auto', vmin=0, vmax=1, cmap='viridis')
    plt.colorbar(im, label='ICC(2,1)')
    plt.yticks(range(len(pivot.index)), pivot.index)
    plt.xticks(range(len(pivot.columns)), pivot.columns, rotation=90)
    plt.title('Parcel-wise ICC Heatmap (rows/columns ordered by mean ICC)')
    plt.tight_layout()
    plt.savefig(out_file, dpi=200)
    plt.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--input-glob',
        default='/cbica/projects/nibs/derivatives/DKTatlas_myelin_stats/sub-*/sub-*_ses-*_run-*_desc-DKTatlas_scalarstats.csv',
        help='Glob pattern to per-run parcel summary CSVs.',
    )
    parser.add_argument(
        '--outdir',
        default='/cbica/projects/nibs/derivatives/ICC',
        help='Output directory for ICC CSV + heatmap.',
    )
    parser.add_argument(
        '--qc-file',
        type=Path,
        default=DEFAULT_QC_FILE,
        help='Manual modality QC TSV.',
    )
    parser.add_argument(
        '--qc-mode',
        nargs='+',
        choices=QC_MODES,
        default=list(QC_MODES),
        help='QC-filtered ICC versions to write.',
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    all_rows = collect_rows(args.input_glob)
    if all_rows.empty:
        raise RuntimeError(f'No input files found with glob: {args.input_glob}')
    qc_df = load_qc_table(args.qc_file)

    for stat in ('mean', 'median'):
        value_df = build_value_table(all_rows, stat=stat)
        for qc_mode in args.qc_mode:
            if qc_mode == 'metricqc':
                filtered_df = apply_metric_qc(value_df, qc_df)
            elif qc_mode == 'completeqc':
                filtered_df = apply_complete_qc(value_df, qc_df)
            else:
                raise ValueError(f'Unsupported QC mode: {qc_mode}')

            icc_df = compute_icc_table(filtered_df)
            if icc_df.empty:
                raise RuntimeError(
                    f'No valid ICC results for stat={stat}, qc_mode={qc_mode}. '
                    'Check QC and session coverage.'
                )
            diagnostics_df = compute_icc_diagnostics(filtered_df, icc_df)
            diagnostics_df.insert(0, 'qc_mode', qc_mode)
            diagnostics_csv = outdir / f'icc_diagnostics_DKTatlas_{stat}_{qc_mode}.csv'
            diagnostics_df.to_csv(diagnostics_csv, index=False)

            icc_df.insert(0, 'qc_mode', qc_mode)

            icc_csv = outdir / f'icc_summary_DKTatlas_{stat}_{qc_mode}.csv'
            icc_df.to_csv(icc_csv, index=False)

            heatmap_png = outdir / f'icc_heatmap_DKTatlas_{stat}_{qc_mode}.png'
            plot_heatmap(icc_df, heatmap_png)

            print(
                f'Wrote: {icc_csv} '
                f'(rows={len(filtered_df)}, subjects={filtered_df["subject"].nunique()})',
                flush=True,
            )
            print(f'Wrote: {diagnostics_csv}', flush=True)
            print(f'Wrote: {heatmap_png}', flush=True)


if __name__ == '__main__':
    main()
