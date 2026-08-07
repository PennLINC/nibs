#!/usr/bin/env python3
"""Test whether R1-DKI associations vary with FOD-derived fiber complexity.

The script estimates voxelwise R1 associations with RK and MKT within each
subject/session and fiber-complexity bin. It uses subjects/sessions, not voxels,
as the group-level inferential units:

1. Build ACPC-grid voxel data from R1, RK, MKT, fixel-count complexity, and an
   anatomical WM mask.
2. Z-score R1/RK/MKT within each subject/session WM sample.
3. Estimate RK and MKT slopes within categorical fiber-complexity bins.
4. Summarize planned contrasts and fit a mixed model across slope estimates.
"""

from __future__ import annotations

import argparse
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

try:
    import numpy as np
    import pandas as pd
    from scipy import stats
except ImportError:  # pragma: no cover - checked at runtime after argparse handles --help
    np = None
    pd = None
    stats = None

ASEG_WM_LABELS = (2, 41)
SMRIPREP_WM_LABELS = (2,)
METRIC_ORDER = ('RK', 'MKT')


def require_analysis_dependencies() -> None:
    missing = [
        name
        for name, module in (
            ('numpy', np),
            ('pandas', pd),
            ('scipy', stats),
        )
        if module is None
    ]
    if missing:
        raise RuntimeError(
            'Missing required Python packages: '
            f'{", ".join(missing)}. Activate the NIBS processing environment first.'
        )


def complexity_labels(complexity_cap: int) -> tuple[str, ...]:
    if complexity_cap < 2:
        raise ValueError('--complexity-cap must be at least 2')
    return tuple(str(value) for value in range(1, complexity_cap)) + (f'{complexity_cap}plus',)


def bin_complexity(values: np.ndarray, complexity_cap: int) -> np.ndarray:
    values = np.asarray(values, dtype=np.int16)
    return np.where(values >= complexity_cap, f'{complexity_cap}plus', values.astype(str))


def display_complexity(label: object) -> str:
    text = str(label)
    return text.replace('plus', '+')


@dataclass(frozen=True)
class SessionInputs:
    subject: str
    session: str
    r1: Path
    rk: Path
    mkt: Path
    complexity: Path
    anatomical_dseg: Path
    dseg_source: str
    wm_labels: tuple[int, ...]
    t1w_to_acpc_xfm: Path | None


def normalize_subject(value: str) -> str:
    token = value.strip()
    return token if token.startswith('sub-') else f'sub-{token}'


def normalize_session(value: str) -> str:
    token = value.strip()
    return token if token.startswith('ses-') else f'ses-{token}'


def first_glob(patterns: Iterable[Path]) -> Path | None:
    matches: list[Path] = []
    for pattern in patterns:
        matches.extend(sorted(pattern.parent.glob(pattern.name)))
    return sorted(set(matches))[0] if matches else None


def validate_metric_file(path: Path, param: str) -> None:
    expected = f'_param-{param}_'
    if expected not in path.name:
        raise ValueError(f'Expected {expected} in selected {param.upper()} file: {path}')


def discover_subjects(derivatives: Path, fixel_dir: Path) -> list[str]:
    roots = (
        fixel_dir,
        derivatives / 'pymp2rage',
        derivatives / 'qsirecon' / 'derivatives' / 'qsirecon-DIPYDKI',
    )
    return sorted(
        {
            path.name
            for root in roots
            if root.is_dir()
            for path in root.glob('sub-*')
            if path.is_dir()
        }
    )


def discover_sessions(derivatives: Path, fixel_dir: Path, subject: str) -> list[str]:
    roots = (
        fixel_dir / subject,
        derivatives / 'pymp2rage' / subject,
        derivatives / 'qsirecon' / 'derivatives' / 'qsirecon-DIPYDKI' / subject,
    )
    return sorted(
        {
            path.name
            for root in roots
            if root.is_dir()
            for path in root.glob('ses-*')
            if path.is_dir()
        }
    )


def collect_inputs(
    derivatives: Path,
    fixel_dir: Path,
    subject: str,
    session: str,
    r1_kind: str,
    wm_labels: tuple[int, ...] | None,
) -> SessionInputs | None:
    r1_desc = 'desc-B1corrected_' if r1_kind == 'b1corrected' else ''
    r1 = first_glob(
        (
            derivatives
            / 'pymp2rage'
            / subject
            / session
            / 'anat'
            / f'{subject}_{session}_run-01_space-T1w_{r1_desc}R1map.nii*',
            derivatives
            / 'pymp2rage'
            / subject
            / session
            / 'anat'
            / f'{subject}_{session}_*space-T1w*{r1_desc}R1map.nii*',
        )
    )
    dki_dir = derivatives / 'qsirecon' / 'derivatives' / 'qsirecon-DIPYDKI' / subject / session / 'dwi'
    rk = first_glob(
        (
            dki_dir / f'{subject}_{session}_*_space-ACPC_model-dki_param-rk_dwimap.nii*',
            dki_dir / f'{subject}_{session}_*param-rk_dwimap.nii*',
        )
    )
    mkt = first_glob(
        (
            dki_dir / f'{subject}_{session}_*_space-ACPC_model-dki_param-mkt_dwimap.nii*',
            dki_dir / f'{subject}_{session}_*param-mkt_dwimap.nii*',
        )
    )
    complexity = first_glob(
        (
            fixel_dir
            / subject
            / session
            / f'{subject}_{session}_space-ACPC_desc-fiberpopulation_count.nii*',
            fixel_dir / subject / session / f'{subject}_{session}_*fiberpopulation_count.nii*',
        )
    )
    qsiprep_aseg = first_glob(
        (
            derivatives
            / 'qsiprep'
            / subject
            / session
            / 'anat'
            / f'{subject}_{session}_space-ACPC_desc-aseg_dseg.nii*',
            derivatives
            / 'qsiprep'
            / subject
            / session
            / 'anat'
            / f'{subject}_{session}_*space-ACPC*desc-aseg_dseg.nii*',
            derivatives / 'qsiprep' / subject / 'anat' / f'{subject}_space-ACPC_desc-aseg_dseg.nii*',
            derivatives / 'qsiprep' / subject / 'anat' / f'{subject}_*space-ACPC*desc-aseg_dseg.nii*',
        )
    )
    smriprep_dseg = first_glob(
        (
            derivatives
            / 'smriprep'
            / subject
            / session
            / 'anat'
            / f'{subject}_{session}_acq-MPRAGE*run-01_dseg.nii*',
            derivatives / 'smriprep' / subject / session / 'anat' / f'{subject}_{session}_*dseg.nii*',
            derivatives / 'smriprep' / subject / 'anat' / f'{subject}_*dseg.nii*',
        )
    )
    t1w_to_acpc_xfm = first_glob(
        (
            derivatives
            / 't1w_registration'
            / subject
            / 'anat'
            / f'{subject}_from-T1w_to-ACPC_mode-image_xfm.h5',
        )
    )

    if qsiprep_aseg is not None:
        anatomical_dseg = qsiprep_aseg
        dseg_source = 'qsiprep_aseg'
        selected_wm_labels = wm_labels or ASEG_WM_LABELS
    else:
        anatomical_dseg = smriprep_dseg
        dseg_source = 'smriprep_dseg'
        selected_wm_labels = wm_labels or SMRIPREP_WM_LABELS

    prerequisites = (
        ('R1', r1),
        ('RK', rk),
        ('MKT', mkt),
        ('fiber complexity count map', complexity),
        ('QSIPrep ACPC aseg or sMRIPrep dseg', anatomical_dseg),
        ('T1w-to-ACPC transform for R1', t1w_to_acpc_xfm),
    )
    missing = [name for name, value in prerequisites if value is None]
    if missing:
        print(f'Skipping {subject} {session}: missing {", ".join(missing)}')
        return None
    validate_metric_file(rk, 'rk')
    validate_metric_file(mkt, 'mkt')

    return SessionInputs(
        subject=subject,
        session=session,
        r1=r1,
        rk=rk,
        mkt=mkt,
        complexity=complexity,
        anatomical_dseg=anatomical_dseg,
        dseg_source=dseg_source,
        wm_labels=selected_wm_labels,
        t1w_to_acpc_xfm=t1w_to_acpc_xfm,
    )


def load_like(path: Path, reference: object, order: int) -> np.ndarray:
    import nibabel as nib
    from nibabel.processing import resample_from_to

    image = nib.load(str(path))
    if image.shape[:3] != reference.shape[:3] or not np.allclose(
        image.affine, reference.affine, atol=1e-4
    ):
        image = resample_from_to(image, reference, order=order)
    return np.asarray(image.get_fdata(), dtype=np.float32)


def run_ants_apply_transforms(
    input_image: Path,
    reference: Path,
    output: Path,
    interpolation: str,
    transforms: list[Path],
) -> None:
    command = [
        'antsApplyTransforms',
        '-d',
        '3',
        '-i',
        str(input_image),
        '-r',
        str(reference),
        '-o',
        str(output),
        '-n',
        interpolation,
        '--float',
        '1',
    ]
    for transform in transforms:
        command.extend(('-t', str(transform)))
    subprocess.run(command, check=True)


def prepare_acpc_image(
    input_image: Path,
    reference: Path,
    output: Path,
    interpolation: str,
    transforms: list[Path],
    force: bool,
) -> Path:
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() and not force:
        return output
    run_ants_apply_transforms(input_image, reference, output, interpolation, transforms)
    return output


def zscore(values: np.ndarray) -> np.ndarray:
    mean = float(np.mean(values))
    std = float(np.std(values, ddof=0))
    if not np.isfinite(std) or std == 0:
        return np.full(values.shape, np.nan, dtype=np.float32)
    return ((values - mean) / std).astype(np.float32)


def slope_and_stats(x: np.ndarray, y: np.ndarray) -> dict[str, float]:
    finite = np.isfinite(x) & np.isfinite(y)
    x = x[finite].astype(float, copy=False)
    y = y[finite].astype(float, copy=False)
    if x.size < 3 or np.std(x, ddof=0) == 0:
        return {
            'n_voxels': int(x.size),
            'slope': np.nan,
            'intercept': np.nan,
            'r': np.nan,
            'r_squared': np.nan,
            'p_value_voxel_naive': np.nan,
        }
    result = stats.linregress(x, y)
    return {
        'n_voxels': int(x.size),
        'slope': float(result.slope),
        'intercept': float(result.intercept),
        'r': float(result.rvalue),
        'r_squared': float(result.rvalue**2),
        'p_value_voxel_naive': float(result.pvalue),
    }


def prepare_session_voxels(
    inputs: SessionInputs,
    work_dir: Path,
    force: bool,
    complexity_order: tuple[str, ...],
    complexity_cap: int,
) -> pd.DataFrame:
    import nibabel as nib

    reference_img = nib.load(str(inputs.complexity))
    stem = f'{inputs.subject}_{inputs.session}_space-ACPC'
    session_work_dir = work_dir / inputs.subject / inputs.session

    r1_acpc = prepare_acpc_image(
        inputs.r1,
        inputs.complexity,
        session_work_dir / f'{stem}_desc-r1_R1map.nii.gz',
        interpolation='Linear',
        transforms=[inputs.t1w_to_acpc_xfm] if inputs.t1w_to_acpc_xfm is not None else [],
        force=force,
    )
    if inputs.dseg_source == 'qsiprep_aseg':
        dseg_acpc = inputs.anatomical_dseg
    else:
        dseg_acpc = prepare_acpc_image(
            inputs.anatomical_dseg,
            inputs.complexity,
            session_work_dir / f'{stem}_desc-anatomical_dseg.nii.gz',
            interpolation='GenericLabel',
            transforms=[inputs.t1w_to_acpc_xfm] if inputs.t1w_to_acpc_xfm is not None else [],
            force=force,
        )

    r1 = load_like(r1_acpc, reference_img, order=1)
    rk = load_like(inputs.rk, reference_img, order=1)
    mkt = load_like(inputs.mkt, reference_img, order=1)
    complexity = np.rint(load_like(inputs.complexity, reference_img, order=0)).astype(np.int16)
    dseg = np.rint(load_like(dseg_acpc, reference_img, order=0)).astype(np.int16)

    wm_mask = np.isin(dseg, inputs.wm_labels)
    scalar_valid = (
        np.isfinite(r1)
        & np.isfinite(rk)
        & np.isfinite(mkt)
        & (r1 != 0)
        & (rk != 0)
        & (mkt != 0)
    )
    wm_scalar_valid = wm_mask & scalar_valid
    valid = (
        wm_scalar_valid
        & (complexity >= 1)
    )
    if not np.any(valid):
        return pd.DataFrame()

    flat_complexity = complexity[valid]
    complexity_bin = bin_complexity(flat_complexity, complexity_cap)
    r1_values = r1[valid].astype(np.float32, copy=False)
    rk_values = rk[valid].astype(np.float32, copy=False)
    mkt_values = mkt[valid].astype(np.float32, copy=False)

    return pd.DataFrame(
        {
            'subject': inputs.subject,
            'session': inputs.session,
            'n_wm_scalar_valid_voxels': int(np.count_nonzero(wm_scalar_valid)),
            'n_zero_fixel_wm_voxels': int(np.count_nonzero(wm_scalar_valid & (complexity == 0))),
            'n_included_wm_voxels': int(np.count_nonzero(valid)),
            'complexity': pd.Categorical(complexity_bin, categories=complexity_order, ordered=True),
            'R1': r1_values,
            'RK': rk_values,
            'MKT': mkt_values,
        }
    )


def estimate_session_slopes(
    voxels: pd.DataFrame,
    inputs: SessionInputs,
    min_voxels: int,
    complexity_order: tuple[str, ...],
) -> pd.DataFrame:
    if voxels.empty:
        return pd.DataFrame()

    subject = str(voxels['subject'].iloc[0])
    session = str(voxels['session'].iloc[0])
    n_wm_scalar_valid_voxels = int(voxels['n_wm_scalar_valid_voxels'].iloc[0])
    n_zero_fixel_wm_voxels = int(voxels['n_zero_fixel_wm_voxels'].iloc[0])
    n_included_wm_voxels = int(voxels['n_included_wm_voxels'].iloc[0])
    rows: list[dict[str, object]] = []
    z = voxels.copy()
    z['R1_z'] = zscore(z['R1'].to_numpy(dtype=np.float32))
    for metric in METRIC_ORDER:
        z[f'{metric}_z'] = zscore(z[metric].to_numpy(dtype=np.float32))
        for complexity in complexity_order:
            bin_df = z[z['complexity'].astype(str) == complexity]
            result = slope_and_stats(
                bin_df[f'{metric}_z'].to_numpy(dtype=np.float32),
                bin_df['R1_z'].to_numpy(dtype=np.float32),
            )
            if result['n_voxels'] < min_voxels:
                for key in ('slope', 'intercept', 'r', 'r_squared', 'p_value_voxel_naive'):
                    result[key] = np.nan
            rows.append(
                {
                    'subject': subject,
                    'session': session,
                    'metric': metric,
                    'complexity': complexity,
                    'n_wm_scalar_valid_voxels': n_wm_scalar_valid_voxels,
                    'n_zero_fixel_wm_voxels': n_zero_fixel_wm_voxels,
                    'n_included_wm_voxels': n_included_wm_voxels,
                    'r1_file': str(inputs.r1),
                    'rk_file': str(inputs.rk),
                    'mkt_file': str(inputs.mkt),
                    'complexity_file': str(inputs.complexity),
                    'anatomical_dseg_file': str(inputs.anatomical_dseg),
                    'dseg_source': inputs.dseg_source,
                    'wm_labels': ','.join(str(label) for label in inputs.wm_labels),
                    **result,
                }
            )
    return pd.DataFrame(rows)


def paired_contrasts(slopes: pd.DataFrame, complexity_order: tuple[str, ...]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    if slopes.empty:
        return pd.DataFrame()

    wide = slopes.pivot_table(
        index=['subject', 'session', 'metric'],
        columns='complexity',
        values='slope',
        aggfunc='first',
    )
    for (subject, session, metric), row in wide.iterrows():
        for target in complexity_order[1:]:
            rows.append(
                {
                    'subject': subject,
                    'session': session,
                    'contrast': f'{target}_minus_1',
                    'metric': metric,
                    'estimate': row.get(target, np.nan) - row.get('1', np.nan),
                    'interpretation': f'{metric} complexity {target} slope minus complexity 1 slope',
                }
            )

    metric_wide = slopes.pivot_table(
        index=['subject', 'session', 'complexity'],
        columns='metric',
        values='slope',
        aggfunc='first',
    )
    metric_delta_rows: list[dict[str, object]] = []
    for (subject, session, complexity), row in metric_wide.iterrows():
        metric_delta_rows.append(
            {
                'subject': subject,
                'session': session,
                'complexity': complexity,
                'mkt_minus_rk': row.get('MKT', np.nan) - row.get('RK', np.nan),
            }
        )
    metric_delta = pd.DataFrame(metric_delta_rows)
    if not metric_delta.empty:
        delta_wide = metric_delta.pivot_table(
            index=['subject', 'session'],
            columns='complexity',
            values='mkt_minus_rk',
            aggfunc='first',
        )
        for (subject, session), row in delta_wide.iterrows():
            for target in complexity_order[1:]:
                rows.append(
                    {
                        'subject': subject,
                        'session': session,
                        'contrast': f'differential_{target}_minus_1',
                        'metric': 'MKT_minus_RK',
                        'estimate': row.get(target, np.nan) - row.get('1', np.nan),
                        'interpretation': (
                            f'(MKT {target}-1 change) minus (RK {target}-1 change); '
                            'positive means RK weakens more than MKT'
                        ),
                    }
                )
    return pd.DataFrame(rows)


def add_fisher_z(slopes: pd.DataFrame) -> pd.DataFrame:
    df = slopes.copy()
    r = pd.to_numeric(df['r'], errors='coerce').to_numpy(dtype=float)
    df['fisher_z'] = np.arctanh(np.clip(r, -0.999999, 0.999999))
    return df


def summarize_contrasts(contrasts: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    subject_contrasts = contrasts.copy()
    subject_contrasts['estimate'] = pd.to_numeric(subject_contrasts['estimate'], errors='coerce')
    subject_contrasts = (
        subject_contrasts.groupby(['subject', 'contrast', 'metric'], as_index=False)['estimate']
        .mean()
    )
    for (contrast, metric), df in subject_contrasts.groupby(['contrast', 'metric'], sort=True):
        values = pd.to_numeric(df['estimate'], errors='coerce').dropna().to_numpy(dtype=float)
        if values.size == 0:
            continue
        if values.size >= 2:
            t_result = stats.ttest_1samp(values, 0.0, nan_policy='omit')
            try:
                w_result = stats.wilcoxon(values)
                w_p = float(w_result.pvalue)
            except ValueError:
                w_p = np.nan
            ci_low, ci_high = stats.t.interval(
                0.95,
                values.size - 1,
                loc=float(np.mean(values)),
                scale=float(stats.sem(values)),
            )
        else:
            t_result = None
            w_p = np.nan
            ci_low = np.nan
            ci_high = np.nan
        rows.append(
            {
                'contrast': contrast,
                'metric': metric,
                'n': int(values.size),
                'mean': float(np.mean(values)),
                'median': float(np.median(values)),
                'std': float(np.std(values, ddof=1)) if values.size > 1 else np.nan,
                'ci95_low': float(ci_low) if np.isfinite(ci_low) else np.nan,
                'ci95_high': float(ci_high) if np.isfinite(ci_high) else np.nan,
                't': float(t_result.statistic) if t_result is not None else np.nan,
                'p_ttest_two_sided': float(t_result.pvalue) if t_result is not None else np.nan,
                'p_ttest_less_than_zero': (
                    float(stats.ttest_1samp(values, 0.0, alternative='less').pvalue)
                    if values.size >= 2
                    else np.nan
                ),
                'p_ttest_greater_than_zero': (
                    float(stats.ttest_1samp(values, 0.0, alternative='greater').pvalue)
                    if values.size >= 2
                    else np.nan
                ),
                'p_wilcoxon_two_sided': w_p,
            }
        )
    return pd.DataFrame(rows)


def fit_group_model(
    slopes: pd.DataFrame,
    out_file: Path,
    complexity_order: tuple[str, ...],
    outcome: str = 'slope',
) -> None:
    try:
        import statsmodels.formula.api as smf
    except ImportError:
        print('statsmodels is not installed; skipping group mixed model')
        return

    df = slopes.copy()
    df = df[np.isfinite(pd.to_numeric(df[outcome], errors='coerce'))].copy()
    if df.empty:
        print(f'No finite {outcome} values; skipping group mixed model')
        return
    df['metric'] = pd.Categorical(df['metric'], categories=METRIC_ORDER)
    df['complexity'] = pd.Categorical(df['complexity'], categories=complexity_order, ordered=True)
    df['subject_session'] = df['subject'].astype(str) + ':' + df['session'].astype(str)
    formula = f'{outcome} ~ C(metric, Treatment(reference="RK")) * C(complexity, Treatment(reference="1")) + C(session)'
    try:
        model = smf.mixedlm(
            formula,
            df,
            groups=df['subject'],
            re_formula='1',
            vc_formula={'subject_session': '0 + C(subject_session)'},
        )
        fit = model.fit(reml=False, method='lbfgs')
        table = fit.summary().tables[1].reset_index(names='term')
        table.to_csv(out_file, sep='\t', index=False)
    except Exception as exc:
        print(f'Mixed model failed ({exc}); fitting subject-clustered OLS fallback')
        fit = smf.ols(formula, df).fit(
            cov_type='cluster',
            cov_kwds={'groups': df['subject']},
        )
        table = fit.summary2().tables[1].reset_index(names='term')
        table.to_csv(out_file, sep='\t', index=False)


def plot_metric_by_complexity(
    slopes: pd.DataFrame,
    out_file: Path,
    complexity_order: tuple[str, ...],
    value_col: str,
    ylabel: str,
) -> None:
    import matplotlib

    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    df = slopes.copy()
    df = df[np.isfinite(pd.to_numeric(df[value_col], errors='coerce'))]
    if df.empty:
        return

    subject_summary = (
        df.groupby(['subject', 'metric', 'complexity'], observed=False, as_index=False)[value_col]
        .mean()
    )
    summary = (
        subject_summary.groupby(['metric', 'complexity'], observed=False)[value_col]
        .agg(['mean', 'sem', 'count'])
        .reset_index()
    )
    x_positions = {label: idx for idx, label in enumerate(complexity_order)}
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    colors = {'RK': '#4477AA', 'MKT': '#CC6677'}
    for metric in METRIC_ORDER:
        metric_summary = summary[summary['metric'] == metric]
        x = [x_positions[str(value)] for value in metric_summary['complexity']]
        y = metric_summary['mean'].to_numpy(dtype=float)
        yerr = metric_summary['sem'].to_numpy(dtype=float)
        ax.errorbar(
            x,
            y,
            yerr=yerr,
            marker='o',
            linewidth=2,
            capsize=4,
            label=metric,
            color=colors[metric],
        )
    ax.axhline(0, color='0.75', linewidth=1)
    ax.set_xticks(
        range(len(complexity_order)),
        [display_complexity(label) for label in complexity_order],
    )
    ax.set_xlabel('Fiber populations')
    ax.set_ylabel(ylabel)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_file, dpi=300)
    plt.close(fig)


def plot_voxel_counts(
    slopes: pd.DataFrame,
    out_file: Path,
    complexity_order: tuple[str, ...],
) -> None:
    import matplotlib

    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    df = slopes[['subject', 'session', 'complexity', 'n_voxels']].drop_duplicates().copy()
    df = df[np.isfinite(pd.to_numeric(df['n_voxels'], errors='coerce'))]
    if df.empty:
        return

    subject_summary = (
        df.groupby(['subject', 'complexity'], observed=False, as_index=False)['n_voxels']
        .mean()
    )
    summary = (
        subject_summary.groupby('complexity', observed=False)['n_voxels']
        .agg(['mean', 'sem', 'count'])
        .reset_index()
    )
    x_positions = {label: idx for idx, label in enumerate(complexity_order)}
    x = [x_positions[str(value)] for value in summary['complexity']]

    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    ax.bar(
        x,
        summary['mean'].to_numpy(dtype=float),
        yerr=summary['sem'].to_numpy(dtype=float),
        capsize=4,
        color='#888888',
        edgecolor='black',
        linewidth=0.6,
    )
    ax.set_xticks(
        range(len(complexity_order)),
        [display_complexity(label) for label in complexity_order],
    )
    ax.set_xlabel('Fiber populations')
    ax.set_ylabel('WM voxels per subject')
    fig.tight_layout()
    fig.savefig(out_file, dpi=300)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--derivatives-dir', type=Path, default=Path('~/derivatives').expanduser())
    parser.add_argument(
        '--fixel-count-dir',
        type=Path,
        default=Path('~/derivatives/fixel_count').expanduser(),
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=Path('~/derivatives/fixel_count_analysis').expanduser(),
    )
    parser.add_argument(
        '--work-dir',
        type=Path,
        default=Path('~/derivatives/fixel_count_analysis/work').expanduser(),
    )
    parser.add_argument('--subject-id', action='append', help='Subject(s), with or without sub-.')
    parser.add_argument('--session-id', action='append', help='Session(s), with or without ses-.')
    parser.add_argument(
        '--r1-kind',
        choices=('native', 'b1corrected'),
        default='native',
        help='Use either *_R1map or *_desc-B1corrected_R1map.',
    )
    parser.add_argument(
        '--wm-label',
        action='append',
        type=int,
        dest='wm_labels',
        help='WM label to include in the anatomical dseg. May be repeated.',
    )
    parser.add_argument('--min-voxels', type=int, default=100)
    parser.add_argument(
        '--complexity-cap',
        type=int,
        default=5,
        help=(
            'Largest separate-or-higher complexity bin. The default 5 creates '
            '1, 2, 3, 4, and 5+ bins. Use 3 for the original 1, 2, 3+ bins.'
        ),
    )
    parser.add_argument('--force', action='store_true', help='Recreate warped intermediates.')
    parser.add_argument(
        '--keep-voxel-tables',
        action='store_true',
        help='Write per-session voxel tables. These can be large.',
    )
    args = parser.parse_args()
    args.derivatives_dir = args.derivatives_dir.expanduser().resolve()
    args.fixel_count_dir = args.fixel_count_dir.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()
    args.work_dir = args.work_dir.expanduser().resolve()
    args.wm_labels = tuple(args.wm_labels) if args.wm_labels else None
    return args


def main() -> None:
    args = parse_args()
    require_analysis_dependencies()
    complexity_order = complexity_labels(args.complexity_cap)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.work_dir.mkdir(parents=True, exist_ok=True)
    voxel_dir = args.output_dir / 'voxel_tables'
    if args.keep_voxel_tables:
        voxel_dir.mkdir(parents=True, exist_ok=True)

    subjects = (
        [normalize_subject(subject) for subject in args.subject_id]
        if args.subject_id
        else discover_subjects(args.derivatives_dir, args.fixel_count_dir)
    )
    all_slopes: list[pd.DataFrame] = []
    for subject in subjects:
        sessions = (
            [normalize_session(session) for session in args.session_id]
            if args.session_id
            else discover_sessions(args.derivatives_dir, args.fixel_count_dir, subject)
        )
        for session in sessions:
            inputs = collect_inputs(
                args.derivatives_dir,
                args.fixel_count_dir,
                subject,
                session,
                args.r1_kind,
                args.wm_labels,
            )
            if inputs is None:
                continue
            print(f'Processing {subject} {session}', flush=True)
            voxels = prepare_session_voxels(
                inputs,
                args.work_dir,
                args.force,
                complexity_order,
                args.complexity_cap,
            )
            if voxels.empty:
                print(f'Skipping {subject} {session}: no valid WM voxels')
                continue
            if args.keep_voxel_tables:
                voxels.to_csv(
                    voxel_dir / f'{subject}_{session}_voxels.tsv.gz',
                    sep='\t',
                    index=False,
                )
            all_slopes.append(
                estimate_session_slopes(
                    voxels,
                    inputs,
                    min_voxels=args.min_voxels,
                    complexity_order=complexity_order,
                )
            )

    if not all_slopes:
        raise RuntimeError('No slope estimates were generated')

    slopes = pd.concat(all_slopes, ignore_index=True)
    slopes = add_fisher_z(slopes)
    slopes_file = args.output_dir / 'r1_dki_fiber_complexity_slopes.tsv'
    slopes.to_csv(slopes_file, sep='\t', index=False)

    contrasts = paired_contrasts(slopes, complexity_order=complexity_order)
    contrasts_file = args.output_dir / 'r1_dki_fiber_complexity_contrasts.tsv'
    contrasts.to_csv(contrasts_file, sep='\t', index=False)

    fisher_for_contrasts = slopes.copy()
    fisher_for_contrasts['slope'] = fisher_for_contrasts['fisher_z']
    fisher_contrasts = paired_contrasts(fisher_for_contrasts, complexity_order=complexity_order)
    fisher_contrasts_file = args.output_dir / 'r1_dki_fiber_complexity_fisher_z_contrasts.tsv'
    fisher_contrasts.to_csv(fisher_contrasts_file, sep='\t', index=False)

    contrast_summary = summarize_contrasts(contrasts)
    contrast_summary_file = args.output_dir / 'r1_dki_fiber_complexity_contrast_summary.tsv'
    contrast_summary.to_csv(contrast_summary_file, sep='\t', index=False)

    fisher_contrast_summary = summarize_contrasts(fisher_contrasts)
    fisher_contrast_summary_file = (
        args.output_dir / 'r1_dki_fiber_complexity_fisher_z_contrast_summary.tsv'
    )
    fisher_contrast_summary.to_csv(fisher_contrast_summary_file, sep='\t', index=False)

    group_model_file = args.output_dir / 'r1_dki_fiber_complexity_group_model.tsv'
    fit_group_model(slopes, group_model_file, complexity_order=complexity_order, outcome='slope')

    fisher_group_model_file = args.output_dir / 'r1_dki_fiber_complexity_fisher_z_group_model.tsv'
    fit_group_model(
        slopes,
        fisher_group_model_file,
        complexity_order=complexity_order,
        outcome='fisher_z',
    )

    slope_plot_file = args.output_dir / 'r1_dki_fiber_complexity_slopes.png'
    plot_metric_by_complexity(
        slopes,
        slope_plot_file,
        complexity_order=complexity_order,
        value_col='slope',
        ylabel='Voxelwise slope with R1_z',
    )

    correlation_plot_file = args.output_dir / 'r1_dki_fiber_complexity_correlations.png'
    plot_metric_by_complexity(
        slopes,
        correlation_plot_file,
        complexity_order=complexity_order,
        value_col='r',
        ylabel='Voxelwise Pearson r with R1_z',
    )

    fisher_plot_file = args.output_dir / 'r1_dki_fiber_complexity_fisher_z.png'
    plot_metric_by_complexity(
        slopes,
        fisher_plot_file,
        complexity_order=complexity_order,
        value_col='fisher_z',
        ylabel='Fisher-z transformed Pearson r',
    )

    voxel_count_plot_file = args.output_dir / 'r1_dki_fiber_complexity_voxel_counts.png'
    plot_voxel_counts(slopes, voxel_count_plot_file, complexity_order=complexity_order)

    print(f'Wrote {slopes_file}')
    print(f'Wrote {contrasts_file}')
    print(f'Wrote {fisher_contrasts_file}')
    print(f'Wrote {contrast_summary_file}')
    print(f'Wrote {fisher_contrast_summary_file}')
    print(f'Wrote {group_model_file}')
    print(f'Wrote {fisher_group_model_file}')
    print(f'Wrote {slope_plot_file}')
    print(f'Wrote {correlation_plot_file}')
    print(f'Wrote {fisher_plot_file}')
    print(f'Wrote {voxel_count_plot_file}')


if __name__ == '__main__':
    main()
