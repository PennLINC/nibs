"""Shared scalar metric definitions for NIBS analyses."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class MetricSpec:
    label: str
    primary_label: str
    pattern_key: str
    group: str
    family: str
    source_image: str
    qc_modalities: tuple[str, ...]
    primary: bool = False


PRIMARY_METRIC_LABELS = (
    'FA',
    'MD',
    'RD',
    'MKT',
    'RK',
    'DKI Micro AWF',
    'ICVF',
    'RTOP',
    'RTAP',
    'NG',
    'NG (Perpendicular)',
    'GFA',
    'ihMTsat-B1c',
    'ihMTR',
    'R1',
    'R1-B1c',
    'MPRAGE-MyelinW',
    'SPACE-MyelinW',
    'G-ihMTsat',
    'G-ihMTR',
    'QSM-SEPIA-E5',
    'QSM-X-R2p-E5-X',
    'QSM-X-R2p-E5-Para',
    'QSM-X-R2p-E5-Dia',
)

PRIMARY_PATTERN_KEYS = {
    'FA': 'FA (TORTOISE; Inner Shells)',
    'MD': 'MD (TORTOISE; Inner Shells)',
    'RD': 'RD (TORTOISE; Inner Shells)',
    'MKT': 'DKI MKT',
    'RK': 'DKI RK',
    'DKI Micro AWF': 'DKI Micro AWF',
    'ICVF': 'ICVF',
    'RTOP': 'RTOP',
    'RTAP': 'RTAP',
    'NG': 'NG',
    'NG (Perpendicular)': 'NG Perpendicular',
    'GFA': 'GQI GFA',
    'ihMTsat-B1c': 'ihMTsat-B1c',
    'ihMTR': 'ihMTR',
    'R1': 'R1',
    'R1-B1c': 'R1-B1c',
    'MPRAGE-MyelinW': 'MPRAGE-MyelinW',
    'SPACE-MyelinW': 'SPACE-MyelinW',
    'G-ihMTsat': 'G-ihMTsat',
    'G-ihMTR': 'G-ihMTR',
    'QSM-SEPIA-E5': 'QSM-SEPIA-E5',
    'QSM-X-R2p-E5-X': "QSM-X-R2'-E5-X",
    'QSM-X-R2p-E5-Para': "QSM-X-R2'-E5-Para",
    'QSM-X-R2p-E5-Dia': "QSM-X-R2'-E5-Dia",
}

SOURCE_IMAGE_COLORS = {
    'DWI': '#4477AA',
    'QSM': '#AA3377',
    'T1w/T2w': '#CCBB44',
    'ihMT': '#228833',
    'g-ratio': '#555555',
    'R1': '#EE7733',
    'Other': '#999999',
}


def norm_token(text: object) -> str:
    return re.sub(r'[^a-z0-9]+', '', str(text).lower())


def display_label(pattern_key: str) -> str:
    if pattern_key.startswith("QSM-X-R2'-"):
        return pattern_key.replace("R2'", 'R2p')
    if pattern_key == 'NG Perpendicular':
        return 'NG (Perpendicular)'
    if pattern_key == 'NG Parallel':
        return 'NG (Parallel)'
    if pattern_key == 'GQI GFA':
        return 'GFA'
    if pattern_key == 'DKI MKT':
        return 'MKT'
    if pattern_key == 'DKI RK':
        return 'RK'
    return pattern_key


def primary_label(pattern_key: str) -> str:
    for label, key in PRIMARY_PATTERN_KEYS.items():
        if key == pattern_key:
            return label
    return display_label(pattern_key)


def infer_family(group: str, pattern_key: str) -> str:
    if group == 'dMRI':
        if 'TORTOISE; Inner Shells' in pattern_key:
            return 'Tensor'
        if pattern_key.startswith('DKI Micro'):
            return 'DKI Micro'
        if pattern_key.startswith('DKI '):
            return 'DKI'
        if pattern_key.startswith('ICVF') or pattern_key in {'ISOVF', 'OD', 'OD (Modulated)'}:
            return 'NODDI'
        if pattern_key in {'NG', 'NG Parallel', 'NG Perpendicular', 'PA', 'PAth', 'RTAP', 'RTOP', 'RTPP'}:
            return 'MAPMRI'
        if pattern_key.startswith('GQI '):
            return 'GQI'
        return 'dMRI'
    if group == 'T1w/T2w Ratio':
        return 'T1w/T2w'
    if group == 'G-Ratio':
        return 'g-ratio'
    if group == 'MP2RAGE':
        return 'R1'
    return group


def source_image_from_group(group: str) -> str:
    if group == 'dMRI':
        return 'DWI'
    if group == 'T1w/T2w Ratio':
        return 'T1w/T2w'
    if group == 'G-Ratio':
        return 'g-ratio'
    if group == 'MP2RAGE':
        return 'R1'
    if group == 'ihMT':
        return 'ihMT'
    if group == 'QSM':
        return 'QSM'
    return 'Other'


def load_patterns(path: Path) -> dict[str, dict[str, str]]:
    with path.open() as fobj:
        return json.load(fobj)


def qc_modalities_for(group: str, pattern_key: str) -> tuple[str, ...]:
    if group == 'dMRI':
        return ('dMRI',)
    if group == 'ihMT':
        if pattern_key in {'ihMTsat', 'ihMTsat-B1c'}:
            return ('MP2RAGE', 'ihMTRAGE', 'B1+')
        return ('ihMTRAGE',)
    if group == 'MP2RAGE':
        if pattern_key == 'R1-B1c':
            return ('MP2RAGE', 'B1+')
        return ('MP2RAGE',)
    if group == 'T1w/T2w Ratio':
        if pattern_key in {'MPRAGE-MyelinW', 'Scaled MPRAGE-MyelinW'}:
            return ('MPRAGE T1w', 'SPACE T2w')
        if pattern_key in {'SPACE-MyelinW', 'Scaled SPACE-MyelinW'}:
            return ('SPACE T1w', 'SPACE T2w')
    if group == 'G-Ratio':
        if pattern_key == 'G-ihMTsat':
            return ('MP2RAGE', 'dMRI', 'ihMTRAGE', 'B1+')
        if pattern_key == 'G-ihMTR':
            return ('dMRI', 'ihMTRAGE')
    if group == 'QSM':
        if pattern_key == 'QSM-SEPIA-E5' or pattern_key.endswith('R2pnet-E5-X'):
            return ('MEGRE',)
        if "R2'" in pattern_key:
            return ('MEGRE', 'MESE')
        return ('MEGRE',)
    return ()


def build_metric_specs(
    patterns_file: Path,
) -> list[MetricSpec]:
    nested = load_patterns(patterns_file)
    specs: list[MetricSpec] = []
    primary_keys = set(PRIMARY_PATTERN_KEYS.values())

    for group, group_patterns in nested.items():
        for pattern_key in group_patterns:
            specs.append(
                MetricSpec(
                    label=display_label(pattern_key),
                    primary_label=primary_label(pattern_key),
                    pattern_key=pattern_key,
                    group=group,
                    family=infer_family(group, pattern_key),
                    source_image=source_image_from_group(group),
                    qc_modalities=qc_modalities_for(group, pattern_key),
                    primary=pattern_key in primary_keys,
                )
            )
    return specs


def primary_metric_specs(specs: list[MetricSpec]) -> list[MetricSpec]:
    by_label = {spec.primary_label: spec for spec in specs}
    return [by_label[label] for label in PRIMARY_METRIC_LABELS if label in by_label]


def metric_order(specs: list[MetricSpec], analysis_set: str) -> list[str]:
    if analysis_set == 'primary':
        return [spec.label for spec in primary_metric_specs(specs)]
    if analysis_set == 'full':
        return [spec.label for spec in specs]
    raise ValueError(f'Unsupported metric set: {analysis_set}')


def metric_display_labels(specs: list[MetricSpec], analysis_set: str) -> dict[str, str]:
    if analysis_set == 'primary':
        return {spec.label: spec.primary_label for spec in primary_metric_specs(specs)}
    if analysis_set == 'full':
        return {spec.label: spec.label for spec in specs}
    raise ValueError(f'Unsupported metric set: {analysis_set}')
