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
    tissues: tuple[str, ...]
    primary: bool = False


PRIMARY_METRIC_LABELS = (
    'FA',
    'MD',
    'RD',
    'MKT',
    'RK',
    'AWF',
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
    'Q-Ratio-E5-B1c',
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
    'AWF': 'DKI Micro AWF',
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
    'Q-Ratio-E5-B1c': 'Q-Ratio-E5-B1c',
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
    'MESE': '#66CCEE',
    'MEGRE': '#882255',
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
    if pattern_key == 'DKI Micro AWF':
        return 'AWF'
    return pattern_key


def primary_label(pattern_key: str) -> str:
    if pattern_key == 'ICVF (GM)':
        return 'ICVF'
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
        if (
            pattern_key.startswith('ICVF')
            or pattern_key.startswith('ISOVF')
            or pattern_key.startswith('OD')
        ):
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
    if group in {'MESE', 'MEGRE'}:
        return group
    if group == 'Q-Ratio':
        return 'MEGRE'
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
    if group in {'MESE', 'MEGRE'}:
        return group
    if group == 'Q-Ratio':
        return 'MEGRE'
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
    if group == 'MESE':
        return ('MESE',)
    if group == 'MEGRE':
        if pattern_key.startswith("R2'-"):
            return ('MEGRE', 'MESE')
        return ('MEGRE',)
    if group == 'Q-Ratio':
        if pattern_key.endswith('-B1c'):
            return ('MP2RAGE', 'MEGRE', 'B1+')
        return ('MP2RAGE', 'MEGRE')
    if group == 'QSM':
        if pattern_key == 'QSM-SEPIA-E5' or pattern_key.endswith('R2pnet-E5-X'):
            return ('MEGRE',)
        if "R2'" in pattern_key:
            return ('MEGRE', 'MESE')
        return ('MEGRE',)
    return ()


def is_noddi_pattern(group: str, pattern_key: str) -> bool:
    return group == 'dMRI' and (
        pattern_key.startswith('ICVF')
        or pattern_key.startswith('ISOVF')
        or pattern_key.startswith('OD')
    )


def is_gm_noddi_pattern(group: str, pattern_key: str) -> bool:
    return is_noddi_pattern(group, pattern_key) and '(GM' in pattern_key


def noddi_hybrid_label(pattern_key: str) -> str:
    return display_label(
        pattern_key.replace(' (GM; ', ' (').replace(' (GM)', '')
    )


def tissues_for(group: str, pattern_key: str) -> tuple[str, ...]:
    if group == 'G-Ratio':
        return ('wm',)
    if is_noddi_pattern(group, pattern_key):
        if is_gm_noddi_pattern(group, pattern_key):
            return ('gm',)
        return ('wm', 'gmwm')
    return ('gm', 'wm', 'gmwm')


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
                    tissues=tissues_for(group, pattern_key),
                    primary=pattern_key in primary_keys,
                )
            )
    return specs


def primary_metric_specs(
    specs: list[MetricSpec],
    tissue: str | None = None,
) -> list[MetricSpec]:
    candidates = [
        spec
        for spec in specs
        if tissue is None or tissue in spec.tissues
    ]
    by_label: dict[str, MetricSpec] = {}
    for spec in candidates:
        by_label.setdefault(spec.primary_label, spec)
    return [by_label[label] for label in PRIMARY_METRIC_LABELS if label in by_label]


def metric_specs_for_analysis(
    specs: list[MetricSpec],
    analysis_set: str,
    tissue: str | None = None,
) -> list[MetricSpec]:
    candidates = [
        spec
        for spec in specs
        if tissue is None or tissue in spec.tissues
    ]
    if analysis_set == 'primary':
        return primary_metric_specs(specs, tissue=tissue)
    if analysis_set == 'full':
        return candidates
    raise ValueError(f'Unsupported metric set: {analysis_set}')


def metric_order(
    specs: list[MetricSpec],
    analysis_set: str,
    tissue: str | None = None,
) -> list[str]:
    return [
        spec.label
        for spec in metric_specs_for_analysis(
            specs,
            analysis_set,
            tissue=tissue,
        )
    ]


def metric_display_labels(
    specs: list[MetricSpec],
    analysis_set: str,
    tissue: str | None = None,
) -> dict[str, str]:
    def label_for_spec(spec: MetricSpec) -> str:
        if tissue == 'gm' and spec.group == 'dMRI' and '(GM' in spec.pattern_key:
            return spec.label.replace(' (GM; ', ' (').replace(' (GM)', '')
        return spec.label

    if analysis_set == 'primary':
        return {
            spec.label: spec.primary_label
            for spec in primary_metric_specs(
                specs,
                tissue=tissue,
            )
        }
    if analysis_set == 'full':
        return {
            spec.label: label_for_spec(spec)
            for spec in metric_specs_for_analysis(
                specs,
                analysis_set,
                tissue=tissue,
            )
        }
    raise ValueError(f'Unsupported metric set: {analysis_set}')


def gm_noddi_hybrid_pairs(specs: list[MetricSpec]) -> dict[str, str]:
    """Map regular NODDI labels to GM-NODDI labels for GM+WM hybrids."""

    wm_by_label: dict[str, str] = {}
    gm_by_label: dict[str, str] = {}

    for spec in specs:
        if not is_noddi_pattern(spec.group, spec.pattern_key):
            continue
        hybrid_label = noddi_hybrid_label(spec.pattern_key)
        if is_gm_noddi_pattern(spec.group, spec.pattern_key):
            gm_by_label[hybrid_label] = spec.label
        else:
            wm_by_label[hybrid_label] = spec.label

    return {
        wm_label: gm_by_label[hybrid_label]
        for hybrid_label, wm_label in wm_by_label.items()
        if hybrid_label in gm_by_label
    }
