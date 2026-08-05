#!/usr/bin/env python3
"""Compute test-retest ICCs from QSIRecon bundle scalarstats TSV files."""

from __future__ import annotations

import argparse
import re
from glob import glob
from pathlib import Path
from collections import Counter

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


PATH_RE = re.compile(r'sub-(?P<sub>[^_/]+).*(ses-(?P<ses>[^_/]+))')
DEFAULT_QC_FILE = Path(__file__).resolve().parents[1] / 'data' / 'manual_qc_modality.tsv'
QC_MODES = ('metricqc', 'completeqc')
REQUIRED_COLUMNS = {'bundle', 'variable_name', 'masked_mean', 'masked_median'}
EXCLUDED_BUNDLE_PATTERNS = (
    'AnteriorCommissure',
    'DentatorubrothalamicTract-lr',
    'DentatorubrothalamicTract-rl',
    'DentatorubrothalamicTractlr',
    'DentatorubrothalamicTractrl',
)
BUNDLE_NAME_ALIASES = {
    'Association_CingulumLFrontalParahippocampal': 'Association_CingulumL_FrontalParahippocampal',
    'Association_CingulumLFrontalParietal': 'Association_CingulumL_FrontalParietal',
    'Association_CingulumLParahippocampal': 'Association_CingulumL_Parahippocampal',
    'Association_CingulumLParahippocampalParietal': 'Association_CingulumL_ParahippocampalParietal',
    'Association_CingulumLParolfactory': 'Association_CingulumL_Parolfactory',
    'Association_CingulumLSuperiorLongitudinalFasciculus1': 'Association_CingulumL_SuperiorLongitudinalFasciculus1',
    'Association_CingulumRFrontalParahippocampal': 'Association_CingulumR_FrontalParahippocampal',
    'Association_CingulumRFrontalParietal': 'Association_CingulumR_FrontalParietal',
    'Association_CingulumRParahippocampal': 'Association_CingulumR_Parahippocampal',
    'Association_CingulumRParahippocampalParietal': 'Association_CingulumR_ParahippocampalParietal',
    'Association_CingulumRParolfactory': 'Association_CingulumR_Parolfactory',
    'Association_CingulumRSuperiorLongitudinalFasciculus1': 'Association_CingulumR_SuperiorLongitudinalFasciculus1',
    'Association_SuperiorLongitudinalFasciculusL2': 'Association_SuperiorLongitudinalFasciculusL_2',
    'Association_SuperiorLongitudinalFasciculusL3': 'Association_SuperiorLongitudinalFasciculusL_3',
    'Association_SuperiorLongitudinalFasciculusR2': 'Association_SuperiorLongitudinalFasciculusR_2',
    'Association_SuperiorLongitudinalFasciculusR3': 'Association_SuperiorLongitudinalFasciculusR_3',
    'Commissure_AnteriorCommissureFrontal': 'Commissure_AnteriorCommissure_Frontal',
    'Commissure_AnteriorCommissureOccipital': 'Commissure_AnteriorCommissure_Occipital',
    'Commissure_AnteriorCommissureTemporal': 'Commissure_AnteriorCommissure_Temporal',
    'Commissure_CorpusCallosumBody': 'Commissure_CorpusCallosum_Body',
    'Commissure_CorpusCallosumForcepsMajor': 'Commissure_CorpusCallosum_ForcepsMajor',
    'Commissure_CorpusCallosumForcepsMinor': 'Commissure_CorpusCallosum_ForcepsMinor',
    'Commissure_CorpusCallosumTapetum': 'Commissure_CorpusCallosum_Tapetum',
    'ProjectionBasalGanglia_CorticostriatalTractLAnterior': 'ProjectionBasalGanglia_CorticostriatalTractL_Anterior',
    'ProjectionBasalGanglia_CorticostriatalTractLPosterior': 'ProjectionBasalGanglia_CorticostriatalTractL_Posterior',
    'ProjectionBasalGanglia_CorticostriatalTractLSuperior': 'ProjectionBasalGanglia_CorticostriatalTractL_Superior',
    'ProjectionBasalGanglia_CorticostriatalTractRAnterior': 'ProjectionBasalGanglia_CorticostriatalTractR_Anterior',
    'ProjectionBasalGanglia_CorticostriatalTractRPosterior': 'ProjectionBasalGanglia_CorticostriatalTractR_Posterior',
    'ProjectionBasalGanglia_CorticostriatalTractRSuperior': 'ProjectionBasalGanglia_CorticostriatalTractR_Superior',
    'ProjectionBasalGanglia_ThalamicRadiationLAnterior': 'ProjectionBasalGanglia_ThalamicRadiationL_Anterior',
    'ProjectionBasalGanglia_ThalamicRadiationLPosterior': 'ProjectionBasalGanglia_ThalamicRadiationL_Posterior',
    'ProjectionBasalGanglia_ThalamicRadiationLSuperior': 'ProjectionBasalGanglia_ThalamicRadiationL_Superior',
    'ProjectionBasalGanglia_ThalamicRadiationRAnterior': 'ProjectionBasalGanglia_ThalamicRadiationR_Anterior',
    'ProjectionBasalGanglia_ThalamicRadiationRPosterior': 'ProjectionBasalGanglia_ThalamicRadiationR_Posterior',
    'ProjectionBasalGanglia_ThalamicRadiationRSuperior': 'ProjectionBasalGanglia_ThalamicRadiationR_Superior',
    'ProjectionBrainstem_CorticopontineTractLFrontal': 'ProjectionBrainstem_CorticopontineTractL_Frontal',
    'ProjectionBrainstem_CorticopontineTractLOccipital': 'ProjectionBrainstem_CorticopontineTractL_Occipital',
    'ProjectionBrainstem_CorticopontineTractLParietal': 'ProjectionBrainstem_CorticopontineTractL_Parietal',
    'ProjectionBrainstem_CorticopontineTractRFrontal': 'ProjectionBrainstem_CorticopontineTractR_Frontal',
    'ProjectionBrainstem_CorticopontineTractROccipital': 'ProjectionBrainstem_CorticopontineTractR_Occipital',
    'ProjectionBrainstem_CorticopontineTractRParietal': 'ProjectionBrainstem_CorticopontineTractR_Parietal',
    'ProjectionBrainstem_DentatorubrothalamicTractlr': 'ProjectionBrainstem_DentatorubrothalamicTract-lr',
    'ProjectionBrainstem_DentatorubrothalamicTractrl': 'ProjectionBrainstem_DentatorubrothalamicTract-rl',
}
DKI_METRICS = {
    'DKI-FA',
    'DKI-KFA',
    'DKI-KFA-Micro',
    'DKI-AD',
    'DKI-AD-Micro',
    'DKI-ADE-Micro',
    'DKI-AWF-Micro',
    'DKI-AxonALD-Micro',
    'DKI-AK',
    'DKI-MD',
    'DKI-MD-Micro',
    'DKI-MK',
    'DKI-MKT',
    'DKI-RD',
    'DKI-RD-Micro',
    'DKI-RDE-Micro',
    'DKI-RK',
    'DKI-Linearity',
    'DKI-Planarity',
    'DKI-Sphericity',
    'DKI-Trace-Micro',
    'DKI-Tortuosity-Micro',
    'DKI-MSDKI-AWF',
    'DKI-MSDKI-DI',
    'DKI-MSDKI-MFA',
    'DKI-MSDKI-MSD',
    'DKI-MSDKI-MSK',
}
NODDI_METRICS = {
    'NODDI-ICVF-Modulated',
    'NODDI-ICVF',
    'NODDI-ISOVF',
    'NODDI-OD-Modulated',
    'NODDI-OD',
}
MAPMRI_METRICS = {
    'MAPMRI-NG',
    'MAPMRI-NGPar',
    'MAPMRI-NGPerp',
    'MAPMRI-PA',
    'MAPMRI-PAth',
    'MAPMRI-RTAP',
    'MAPMRI-RTOP',
    'MAPMRI-RTPP',
}
DSISTUDIO_METRICS = {
    'DSIStudio-GQI-GFA',
    'DSIStudio-GQI-ISO',
    'DSIStudio-GQI-QA',
    'DSIStudio-GQI-RDI',
    'DSIStudio-Tensor-AD',
    'DSIStudio-Tensor-FA',
    'DSIStudio-Tensor-MD',
    'DSIStudio-Tensor-RD',
}
TORTOISE_TENSOR_METRICS = {
    'TORTOISE-FullShell-AD',
    'TORTOISE-FullShell-FA',
    'TORTOISE-FullShell-LI',
    'TORTOISE-FullShell-MD',
    'TORTOISE-FullShell-RD',
    'TORTOISE-InnerShell-AD',
    'TORTOISE-InnerShell-FA',
    'TORTOISE-InnerShell-LI',
    'TORTOISE-InnerShell-MD',
    'TORTOISE-InnerShell-RD',
}
MYELIN_METRICS = {
    'MEGRE',
    'QSM-SEPIA-E5',
    'QSM-X-R2p-E5-X',
    'QSM-X-R2p-E5-Para',
    'QSM-X-R2p-E5-Dia',
    'ihMTw',
    'ihMTR',
    'MTR',
    'ihMTsat',
    'ihMTsat-B1c',
    'R1',
    'R1-B1c',
    'MPRAGE-MyelinW',
    'SPACE-MyelinW',
    'Scaled MPRAGE-MyelinW',
    'Scaled SPACE-MyelinW',
    'G-ihMTsat',
    'G-ihMTR',
}
ALL_ALLOWED_METRICS = (
    DKI_METRICS
    | NODDI_METRICS
    | MAPMRI_METRICS
    | DSISTUDIO_METRICS
    | TORTOISE_TENSOR_METRICS
    | MYELIN_METRICS
)

DKI_STD_MAP = {
    'fa': 'DKI-FA',
    'kfa': 'DKI-KFA',
    'ad': 'DKI-AD',
    'ak': 'DKI-AK',
    'md': 'DKI-MD',
    'mk': 'DKI-MK',
    'mkt': 'DKI-MKT',
    'rd': 'DKI-RD',
    'rk': 'DKI-RK',
    'linearity': 'DKI-Linearity',
    'planarity': 'DKI-Planarity',
    'sphericity': 'DKI-Sphericity',
}
DKI_MICRO_MAP = {
    'kfa': 'DKI-KFA-Micro',
    'ad': 'DKI-AD-Micro',
    'ade': 'DKI-ADE-Micro',
    'awf': 'DKI-AWF-Micro',
    'axonald': 'DKI-AxonALD-Micro',
    'md': 'DKI-MD-Micro',
    'rd': 'DKI-RD-Micro',
    'rde': 'DKI-RDE-Micro',
    'trace': 'DKI-Trace-Micro',
    'tortuosity': 'DKI-Tortuosity-Micro',
}
DKI_MSDKI_MAP = {
    'awf': 'DKI-MSDKI-AWF',
    'di': 'DKI-MSDKI-DI',
    'mfa': 'DKI-MSDKI-MFA',
    'msd': 'DKI-MSDKI-MSD',
    'msk': 'DKI-MSDKI-MSK',
}
NODDI_MAP = {
    'icvf': 'NODDI-ICVF',
    'isovf': 'NODDI-ISOVF',
    'od': 'NODDI-OD',
}
MAPMRI_MAP = {
    'ng': 'MAPMRI-NG',
    'ngpar': 'MAPMRI-NGPar',
    'ngperp': 'MAPMRI-NGPerp',
    'pa': 'MAPMRI-PA',
    'path': 'MAPMRI-PAth',
    'rtap': 'MAPMRI-RTAP',
    'rtop': 'MAPMRI-RTOP',
    'rtpp': 'MAPMRI-RTPP',
}
DSISTUDIO_GQI_MAP = {
    'gfa': 'DSIStudio-GQI-GFA',
    'iso': 'DSIStudio-GQI-ISO',
    'qa': 'DSIStudio-GQI-QA',
    'rdi': 'DSIStudio-GQI-RDI',
}
DSISTUDIO_TENSOR_MAP = {
    'ad': 'DSIStudio-Tensor-AD',
    'fa': 'DSIStudio-Tensor-FA',
    'md': 'DSIStudio-Tensor-MD',
    'rd': 'DSIStudio-Tensor-RD',
    'rd1': 'DSIStudio-Tensor-RD1',
    'rd2': 'DSIStudio-Tensor-RD2',
}
TORTOISE_TENSOR_MAP = {
    'ad': 'AD',
    'fa': 'FA',
    'li': 'LI',
    'md': 'MD',
    'rd': 'RD',
}
DIRECT_NAME_MAP = {metric.lower(): metric for metric in ALL_ALLOWED_METRICS}


def metric_required_modalities(metric: str) -> tuple[str, ...]:
    """Return scan-level QC modalities required to trust a derived metric."""
    if metric in (
        DKI_METRICS | NODDI_METRICS | MAPMRI_METRICS | DSISTUDIO_METRICS | TORTOISE_TENSOR_METRICS
    ):
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
    df: pd.DataFrame,
    qc_df: pd.DataFrame,
    subject_col: str = 'subject_id',
    session_col: str = 'session_id',
) -> pd.DataFrame:
    keep = [
        _qc_passes(
            qc_df,
            row[subject_col],
            row[session_col],
            metric_required_modalities(str(row['metric'])),
        )
        for _, row in df.iterrows()
    ]
    return df.loc[keep].copy()


def subjects_with_complete_qc(
    df: pd.DataFrame,
    qc_df: pd.DataFrame,
    subject_col: str = 'subject_id',
) -> set[str]:
    modalities = sorted(
        {
            modality
            for metric in df['metric'].dropna().astype(str).unique()
            for modality in metric_required_modalities(metric)
        }
    )
    subjects = sorted({_normalize_subject(value) for value in df[subject_col].unique()})
    complete_subjects: set[str] = set()
    for subject in subjects:
        if all(
            _qc_passes(qc_df, subject, f'ses-{session:02d}', tuple(modalities))
            for session in (1, 2)
        ):
            complete_subjects.add(subject)
    return complete_subjects


def apply_complete_qc(
    df: pd.DataFrame,
    qc_df: pd.DataFrame,
    subject_col: str = 'subject_id',
) -> pd.DataFrame:
    complete_subjects = subjects_with_complete_qc(df, qc_df, subject_col=subject_col)
    subjects = df[subject_col].map(_normalize_subject)
    return df.loc[subjects.isin(complete_subjects)].copy()


def compute_icc2_fallback(values: np.ndarray, subjects: np.ndarray, sessions: np.ndarray) -> float:
    """Compute ICC(2,1) with complete-case fallback."""
    subs_unique, sub_idx = np.unique(subjects, return_inverse=True)
    sess_unique, ses_idx = np.unique(sessions, return_inverse=True)
    n_sub, n_ses = len(subs_unique), len(sess_unique)
    if n_sub < 2 or n_ses < 2:
        return np.nan

    matrix = np.full((n_sub, n_ses), np.nan, dtype=float)
    for val, i_sub, i_ses in zip(values, sub_idx, ses_idx):
        matrix[i_sub, i_ses] = val

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


def _parse_from_path(path: str) -> tuple[str | None, str | None]:
    match = PATH_RE.search(path)
    if not match:
        return None, None
    return match.group('sub'), f'ses-{match.group("ses")}'


def _extract_param_token(source_file: str) -> str:
    match = re.search(r'_param-([^_]+)', source_file)
    if not match:
        return ''
    return match.group(1).lower()


def _norm_token(text: str) -> str:
    return re.sub(r'[^a-z0-9]+', '', text.lower())


def _param_from_variable_name(variable_name: str, *prefixes: str) -> str:
    """Return a normalized metric token after stripping known prefixes."""
    token = _norm_token(variable_name)
    for prefix in prefixes:
        norm_prefix = _norm_token(prefix)
        if token.startswith(norm_prefix):
            return token[len(norm_prefix) :]
    return token


def _infer_metric_name(row: pd.Series, source_tsv: str) -> str | None:
    var_name = str(row.get('variable_name', '')).strip()
    qsirecon_suffix = str(row.get('qsirecon_suffix', '')).strip()
    source_file = str(row.get('source_file', '')).strip()
    lowered_var = var_name.lower()
    norm_var = _norm_token(var_name)
    lowered_tsv = source_tsv.lower()
    lowered_src = source_file.lower()
    lowered_suffix = qsirecon_suffix.lower()

    # Generated myelin spreadsheet: variable_name should already be canonical.
    if '/bundle_myelin_stats/' in lowered_tsv:
        metric = DIRECT_NAME_MAP.get(lowered_var)
        return metric if metric in MYELIN_METRICS else None

    # DWI DKI spreadsheet
    is_dki = (
        'qsirecon-dipydki' in lowered_tsv
        or 'qsirecon-dipydki' in lowered_src
        or 'dipydki' in lowered_suffix
        or 'dki' in lowered_suffix
    )
    if is_dki:
        param = _extract_param_token(lowered_src)
        var_param = _param_from_variable_name(var_name, 'msdki', 'dki', 'dkimicro')
        is_msdki = (
            'model-msdki' in lowered_src
            or 'msdki' in lowered_suffix
            or norm_var.startswith('msdki')
        )
        is_micro = (
            'model-dkimicro' in lowered_src or 'dkimicro' in lowered_suffix or 'micro' in norm_var
        )
        if is_msdki:
            metric = (
                DKI_MSDKI_MAP.get(param)
                or DKI_MSDKI_MAP.get(var_param)
                or DKI_MSDKI_MAP.get(lowered_var)
            )
            if metric is None:
                for token, out_name in (
                    ('awf', 'DKI-MSDKI-AWF'),
                    ('mfa', 'DKI-MSDKI-MFA'),
                    ('msd', 'DKI-MSDKI-MSD'),
                    ('msk', 'DKI-MSDKI-MSK'),
                    ('di', 'DKI-MSDKI-DI'),
                ):
                    if token in norm_var:
                        metric = out_name
                        break
        elif is_micro:
            metric = (
                DKI_MICRO_MAP.get(param)
                or DKI_MICRO_MAP.get(var_param)
                or DKI_MICRO_MAP.get(lowered_var)
            )
            if metric is None:
                micro_aliases = (
                    ('axonald', 'DKI-AxonALD-Micro'),
                    ('tortuosity', 'DKI-Tortuosity-Micro'),
                    ('trace', 'DKI-Trace-Micro'),
                    ('awf', 'DKI-AWF-Micro'),
                    ('ade', 'DKI-ADE-Micro'),
                    ('rde', 'DKI-RDE-Micro'),
                    ('kfa', 'DKI-KFA-Micro'),
                    ('ad', 'DKI-AD-Micro'),
                    ('md', 'DKI-MD-Micro'),
                    ('rd', 'DKI-RD-Micro'),
                )
                for token, out_name in micro_aliases:
                    if token in norm_var:
                        metric = out_name
                        break
        else:
            metric = (
                DKI_STD_MAP.get(param) or DKI_STD_MAP.get(var_param) or DKI_STD_MAP.get(lowered_var)
            )
            if metric is None:
                std_aliases = (
                    ('linearity', 'DKI-Linearity'),
                    ('planarity', 'DKI-Planarity'),
                    ('sphericity', 'DKI-Sphericity'),
                    ('mkt', 'DKI-MKT'),
                    ('kfa', 'DKI-KFA'),
                    ('fa', 'DKI-FA'),
                    ('ak', 'DKI-AK'),
                    ('mk', 'DKI-MK'),
                    ('rk', 'DKI-RK'),
                    ('ad', 'DKI-AD'),
                    ('md', 'DKI-MD'),
                    ('rd', 'DKI-RD'),
                )
                for token, out_name in std_aliases:
                    if token in norm_var:
                        metric = out_name
                        break
        return metric if metric in DKI_METRICS else None

    # DWI NODDI spreadsheet
    is_noddi = (
        'qsirecon-noddi' in lowered_tsv
        or 'qsirecon-noddi' in lowered_src
        or 'noddi' in lowered_suffix
    )
    if is_noddi:
        param = _extract_param_token(lowered_src) or lowered_var
        metric = NODDI_MAP.get(param)
        if metric is None:
            for token, out_name in (
                ('isovf', 'NODDI-ISOVF'),
                ('icvf', 'NODDI-ICVF'),
                ('nrmse', 'NODDI-NRMSE'),
                ('rmse', 'NODDI-RMSE'),
                ('od', 'NODDI-OD'),
                ('tf', 'NODDI-TF'),
            ):
                if token in norm_var:
                    metric = out_name
                    break
        is_modulated = (
            'desc-modulated' in lowered_src
            or 'modulated' in lowered_src
            or 'modulated' in lowered_var
            or 'modulated' in lowered_suffix
        )
        if metric in {'NODDI-ICVF', 'NODDI-OD'} and is_modulated:
            metric = f'{metric}-Modulated'
        return metric if metric in NODDI_METRICS else None

    # DWI DSIStudio spreadsheet
    is_dsistudio = (
        'qsirecon-dsistudio' in lowered_tsv
        or 'qsirecon-dsistudio' in lowered_src
        or 'dsistudio' in lowered_suffix
        or 'gqi' in lowered_suffix
    )
    if is_dsistudio:
        param = _extract_param_token(lowered_src) or _param_from_variable_name(var_name, 'dti')
        is_gqi = 'model-gqi' in lowered_src or 'gqi' in lowered_suffix or param in DSISTUDIO_GQI_MAP
        is_tensor = (
            'model-tensor' in lowered_src
            or 'tensor' in lowered_suffix
            or param in DSISTUDIO_TENSOR_MAP
        )
        if is_gqi:
            metric = DSISTUDIO_GQI_MAP.get(param) or DSISTUDIO_GQI_MAP.get(lowered_var)
            return metric if metric in DSISTUDIO_METRICS else None
        if is_tensor:
            metric = DSISTUDIO_TENSOR_MAP.get(param) or DSISTUDIO_TENSOR_MAP.get(lowered_var)
            return metric if metric in DSISTUDIO_METRICS else None

    # DWI TORTOISE tensor spreadsheets
    is_tortoise_full_shell = (
        'qsirecon-tortoise_model-mapmri' in lowered_tsv
        or 'qsirecon-tortoise_model-mapmri' in lowered_src
    )
    is_tortoise_inner_shell = (
        'qsirecon-tortoise_model-tensor' in lowered_tsv
        or 'qsirecon-tortoise_model-tensor' in lowered_src
    )
    tortoise_param = _extract_param_token(lowered_src) or lowered_var
    is_tensor_metric = (
        'model-tensor' in lowered_src
        or 'tensor' in lowered_suffix
        or tortoise_param in TORTOISE_TENSOR_MAP
    )
    if is_tensor_metric and (is_tortoise_full_shell or is_tortoise_inner_shell):
        suffix = TORTOISE_TENSOR_MAP.get(tortoise_param) or TORTOISE_TENSOR_MAP.get(lowered_var)
        if suffix is None:
            return None
        prefix = 'TORTOISE-FullShell' if is_tortoise_full_shell else 'TORTOISE-InnerShell'
        metric = f'{prefix}-{suffix}'
        return metric if metric in TORTOISE_TENSOR_METRICS else None

    # DWI MAPMRI spreadsheet
    is_mapmri = (
        'qsirecon-tortoise_model-mapmri' in lowered_tsv
        or 'qsirecon-tortoise_model-mapmri' in lowered_src
        or 'mapmri' in lowered_suffix
    )
    if is_mapmri:
        param = _extract_param_token(lowered_src) or lowered_var
        if 'model-mapmri' not in lowered_src and 'mapmri' not in lowered_suffix:
            return None
        metric = MAPMRI_MAP.get(param)
        if metric is None:
            for token, out_name in (
                ('ngperp', 'MAPMRI-NGPerp'),
                ('ngpar', 'MAPMRI-NGPar'),
                ('ng', 'MAPMRI-NG'),
                ('path', 'MAPMRI-PAth'),
                ('pa', 'MAPMRI-PA'),
                ('rtap', 'MAPMRI-RTAP'),
                ('rtop', 'MAPMRI-RTOP'),
                ('rtpp', 'MAPMRI-RTPP'),
            ):
                if token in norm_var:
                    metric = out_name
                    break
        return metric if metric in MAPMRI_METRICS else None

    # Unknown source folder -> reject rather than mixing source families.
    return None


def collect_scalarstats(input_globs: list[str]) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    all_files: set[str] = set()
    dropped_counter: Counter[tuple[str, str]] = Counter()
    for input_glob in input_globs:
        for file_path in glob(input_glob):
            all_files.add(file_path)

    for file_path in sorted(all_files):
        df = pd.read_csv(file_path, sep='\t')
        missing = REQUIRED_COLUMNS.difference(df.columns)
        if missing:
            raise RuntimeError(f'Missing required columns {missing} in {file_path}')

        if 'subject_id' not in df.columns or df['subject_id'].isna().all():
            parsed_sub, _ = _parse_from_path(file_path)
            if parsed_sub is None:
                raise RuntimeError(f'Could not infer subject_id from {file_path}')
            df['subject_id'] = parsed_sub

        if 'session_id' not in df.columns or df['session_id'].isna().all():
            _, parsed_ses = _parse_from_path(file_path)
            if parsed_ses is None:
                raise RuntimeError(f'Could not infer session_id from {file_path}')
            df['session_id'] = parsed_ses

        # Source-aware canonical metric mapping.
        df['metric'] = df.apply(lambda row: _infer_metric_name(row, file_path), axis=1)
        dropped_df = df[df['metric'].isna()]
        for _, drow in dropped_df.iterrows():
            dropped_counter[
                (str(drow.get('variable_name', '')), str(drow.get('qsirecon_suffix', '')))
            ] += 1
        df = df[df['metric'].isin(ALL_ALLOWED_METRICS)].copy()
        if df.empty:
            continue
        df['source_tsv'] = file_path
        rows.append(df)

    if not rows:
        return pd.DataFrame()
    all_df = pd.concat(rows, ignore_index=True)
    all_df['subject_id'] = all_df['subject_id'].astype(str).str.replace('^sub-', '', regex=True)
    all_df = all_df.loc[~all_df['subject_id'].map(_is_pilot_subject)].copy()
    all_df['session_id'] = all_df['session_id'].astype(str)
    all_df['bundle'] = all_df['bundle'].astype(str)
    all_df['metric'] = all_df['metric'].astype(str)
    all_df['bundle'] = all_df['bundle'].replace(BUNDLE_NAME_ALIASES)
    excluded_mask = all_df['bundle'].str.contains(
        '|'.join(EXCLUDED_BUNDLE_PATTERNS), regex=True, na=False
    )
    if excluded_mask.any():
        excluded_bundles = ', '.join(sorted(all_df.loc[excluded_mask, 'bundle'].unique()))
        print(
            f'[INFO] Excluding {int(excluded_mask.sum())} rows from inconsistent bundles: '
            f'{excluded_bundles}',
            flush=True,
        )
        all_df = all_df.loc[~excluded_mask].copy()
    if dropped_counter:
        print('[WARN] Dropped rows with unmapped metrics (top 20):', flush=True)
        for (var_name, suffix), count in dropped_counter.most_common(20):
            print(f'  variable_name={var_name} qsirecon_suffix={suffix} n={count}', flush=True)
    return all_df


def collapse_subject_session_values(df: pd.DataFrame, value_col: str) -> pd.DataFrame:
    group_cols = ['subject_id', 'session_id', 'metric', 'bundle']
    collapsed = (
        df[group_cols + [value_col]]
        .dropna(subset=[value_col])
        .groupby(group_cols, as_index=False)[value_col]
        .mean()
    )
    return collapsed


def compute_icc_table(df: pd.DataFrame, value_col: str) -> pd.DataFrame:
    out_rows: list[dict[str, object]] = []
    df = collapse_subject_session_values(df, value_col=value_col)
    grp = df.groupby(['metric', 'bundle'], sort=True)
    for (metric, bundle), dfg in grp:
        dfg = dfg[np.isfinite(dfg[value_col].to_numpy(dtype=float))].copy()
        if dfg.empty:
            continue

        ses_count = dfg.groupby('subject_id')['session_id'].nunique()
        valid_subs = ses_count[ses_count >= 2].index
        dfg = dfg[dfg['subject_id'].isin(valid_subs)]
        if dfg['subject_id'].nunique() < 2 or dfg['session_id'].nunique() < 2:
            continue

        subjects = dfg['subject_id'].to_numpy()
        sessions = dfg['session_id'].to_numpy()
        values = dfg[value_col].to_numpy(dtype=float)

        icc = np.nan
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
                icc = float(icc_row['ICC'])
                ci95 = str(icc_row.get('CI95%', ''))
                f_val = float(icc_row.get('F', np.nan))
                df1 = float(icc_row.get('df1', np.nan))
                df2 = float(icc_row.get('df2', np.nan))
                pval = float(icc_row.get('pval', np.nan))
            except Exception:
                icc = compute_icc2_fallback(values, subjects, sessions)
        else:
            icc = compute_icc2_fallback(values, subjects, sessions)

        out_rows.append(
            {
                'metric': metric,
                'bundle': bundle,
                'stat': value_col,
                'ICC2_1': icc,
                'CI95': ci95,
                'F': f_val,
                'df1': df1,
                'df2': df2,
                'pval': pval,
                'n_subjects': int(dfg['subject_id'].nunique()),
                'n_sessions': int(dfg['session_id'].nunique()),
            }
        )

    if not out_rows:
        return pd.DataFrame()
    return pd.DataFrame(out_rows).sort_values(['metric', 'bundle']).reset_index(drop=True)


def plot_heatmap(df_icc: pd.DataFrame, out_png: Path, title_suffix: str) -> None:
    pivot = df_icc.pivot(index='metric', columns='bundle', values='ICC2_1')
    row_order = pivot.mean(axis=1).sort_values(ascending=False).index
    col_order = pivot.mean(axis=0).sort_values(ascending=False).index
    pivot = pivot.loc[row_order, col_order]

    fig_w = max(12, 0.22 * len(pivot.columns))
    fig_h = max(6, 0.28 * len(pivot.index))
    plt.figure(figsize=(fig_w, fig_h))
    im = plt.imshow(pivot.to_numpy(), aspect='auto', vmin=0, vmax=1, cmap='viridis')
    plt.colorbar(im, label='ICC(2,1)')
    plt.yticks(range(len(pivot.index)), pivot.index)
    plt.xticks(range(len(pivot.columns)), pivot.columns, rotation=90)
    plt.title(f'WM Bundle ICC Heatmap ({title_suffix})')
    plt.tight_layout()
    plt.savefig(out_png, dpi=220)
    plt.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--input-globs',
        nargs='+',
        default=[
            '/cbica/projects/nibs/derivatives/qsirecon/derivatives/qsirecon-*/sub-*/ses-*/dwi/sub-*_ses-*_*_scalarstats.tsv',
            '/cbica/projects/nibs/derivatives/bundle_myelin_stats/sub-*/ses-*/dwi/sub-*_ses-*_acq-HBCD75_run-01_space-T1w_model-*_scalarstats.tsv',
        ],
        help='One or more globs for bundle scalarstats TSV files.',
    )
    parser.add_argument(
        '--outdir',
        default='/cbica/projects/nibs/derivatives/ICC',
        help='Output directory.',
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

    all_df = collect_scalarstats(args.input_globs)
    if all_df.empty:
        raise RuntimeError(f'No scalarstats TSV files found for globs: {args.input_globs}')
    qc_df = load_qc_table(args.qc_file)

    for stat in ('masked_mean', 'masked_median'):
        for qc_mode in args.qc_mode:
            if qc_mode == 'metricqc':
                filtered_df = apply_metric_qc(all_df, qc_df)
            elif qc_mode == 'completeqc':
                filtered_df = apply_complete_qc(all_df, qc_df)
            else:
                raise ValueError(f'Unsupported QC mode: {qc_mode}')

            icc_df = compute_icc_table(filtered_df, value_col=stat)
            if icc_df.empty:
                raise RuntimeError(
                    f'No valid ICC rows for {stat}, qc_mode={qc_mode}. '
                    'Check QC and session coverage.'
                )
            icc_df.insert(0, 'qc_mode', qc_mode)

            out_csv = outdir / f'icc_summary_wm_bundles_{stat}_{qc_mode}.csv'
            out_png = outdir / f'icc_heatmap_wm_bundles_{stat}_{qc_mode}.png'
            icc_df.to_csv(out_csv, index=False)
            plot_heatmap(icc_df, out_png, f'{stat}, {qc_mode}')

            print(
                f'Wrote: {out_csv} '
                f'(rows={len(filtered_df)}, subjects={filtered_df["subject_id"].nunique()})',
                flush=True,
            )
            print(f'Wrote: {out_png}', flush=True)


if __name__ == '__main__':
    main()
