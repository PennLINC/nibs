"""Utilities for parcel/bundle analyses that consume the metric registry."""

from __future__ import annotations

import re
from pathlib import Path
from fnmatch import fnmatch

from metric_registry import MetricSpec, build_metric_specs, metric_order, norm_token


CANONICAL_ALIASES = {
    'tortoisemapmrifa': 'FA (TORTOISE; Inner Shells)',
    'tortoisemapmrimd': 'MD (TORTOISE; Inner Shells)',
    'tortoisemapmrird': 'RD (TORTOISE; Inner Shells)',
    'tortoisetensorfa': 'FA (TORTOISE; Full Shells)',
    'tortoisetensormd': 'MD (TORTOISE; Full Shells)',
    'tortoisetensorrd': 'RD (TORTOISE; Full Shells)',
    'icvf': 'ICVF',
    'ficvf': 'ICVF',
    'noddiicvf': 'ICVF',
    'intraCellularVolumeFraction': 'ICVF',
    'ngperp': 'NG (Perpendicular)',
    'ngperpendicular': 'NG (Perpendicular)',
    'ngperpendicularity': 'NG (Perpendicular)',
    'ngorthogonal': 'NG (Perpendicular)',
    'qsmsepiae5': 'QSM-SEPIA-E5-X',
    'qsmsepiae5x': 'QSM-SEPIA-E5-X',
    'qsmxr2pe5': 'QSM-X-R2p-E5-X',
    'qsmxr2pe5x': 'QSM-X-R2p-E5-X',
    'qsmxr2primee5': 'QSM-X-R2p-E5-X',
    'qsmxr2primee5x': 'QSM-X-R2p-E5-X',
}


QSI_RECON_ALIASES = {
    'gmnoddi': {
        'icvf': 'ICVF (GM)',
        'ficvf': 'ICVF (GM)',
        'isovf': 'ISOVF (GM)',
        'od': 'OD (GM)',
    },
    'tortoisemodelmapmri': {
        'ad': 'AD (TORTOISE; Inner Shells)',
        'fa': 'FA (TORTOISE; Inner Shells)',
        'li': 'LI (TORTOISE; Inner Shells)',
        'rd': 'RD (TORTOISE; Inner Shells)',
        'md': 'MD (TORTOISE; Inner Shells)',
        'ng': 'NG',
        'ngpar': 'NG Parallel',
        'ngperp': 'NG (Perpendicular)',
        'pa': 'PA',
        'path': 'PAth',
        'rtap': 'RTAP',
        'rtop': 'RTOP',
        'rtpp': 'RTPP',
    },
    'tortoisemodeltensor': {
        'ad': 'AD (TORTOISE; Full Shells)',
        'fa': 'FA (TORTOISE; Full Shells)',
        'li': 'LI (TORTOISE; Full Shells)',
        'rd': 'RD (TORTOISE; Full Shells)',
        'md': 'MD (TORTOISE; Full Shells)',
    },
    'dsistudio': {
        'gfa': 'GFA',
        'iso': 'GQI ISO',
        'qa': 'GQI QA',
        'ad': 'AD (DSIStudio)',
        'fa': 'FA (DSIStudio)',
        'md': 'MD (DSIStudio)',
        'rd': 'RD (DSIStudio)',
    },
    'dipy dki': {
        'ad': 'DKI AD',
        'ak': 'DKI AK',
        'fa': 'DKI FA',
        'kfa': 'DKI KFA',
        'md': 'DKI MD',
        'mk': 'DKI MK',
        'mkt': 'MKT',
        'rd': 'DKI RD',
        'rk': 'RK',
    },
    'dipydki': {
        'ad': 'DKI AD',
        'ak': 'DKI AK',
        'fa': 'DKI FA',
        'kfa': 'DKI KFA',
        'md': 'DKI MD',
        'mk': 'DKI MK',
        'mkt': 'MKT',
        'rd': 'DKI RD',
        'rk': 'RK',
    },
    'noddi': {
        'icvf': 'ICVF',
        'ficvf': 'ICVF',
        'isovf': 'ISOVF',
        'od': 'OD',
    },
}


def default_patterns_file() -> Path:
    return Path(__file__).resolve().parents[1] / 'configuration' / 'patterns.json'


def specs_by_label(patterns_file: Path | None = None) -> dict[str, MetricSpec]:
    specs = build_metric_specs(patterns_file or default_patterns_file())
    return {spec.label: spec for spec in specs}


def flattened_patterns(patterns_file: Path | None = None) -> dict[str, str]:
    import json

    with (patterns_file or default_patterns_file()).open() as fobj:
        nested = json.load(fobj)
    return {
        key: value
        for group_patterns in nested.values()
        for key, value in group_patterns.items()
    }


def pattern_token_aliases(
    specs: list[MetricSpec],
    patterns_file: Path | None = None,
) -> dict[str, str]:
    patterns = flattened_patterns(patterns_file)
    by_pattern_key = {spec.pattern_key: spec for spec in specs}
    aliases: dict[str, str] = {}
    for pattern_key, rel_pattern in patterns.items():
        spec = by_pattern_key.get(pattern_key)
        if spec is None:
            continue
        for token in re.findall(r'(?:param|desc)-([A-Za-z0-9]+)', rel_pattern):
            aliases[token] = spec.label
            aliases[f'param{token}'] = spec.label
        if 'ngperp' in rel_pattern.lower():
            aliases['ngperp'] = spec.label
            aliases['ngperpendicular'] = spec.label
        if 'icvf' in rel_pattern.lower() and spec.label == 'ICVF':
            aliases['icvf'] = spec.label
            aliases['noddiicvf'] = spec.label
    return aliases


def canonical_lookup_token(text: object) -> str:
    """Normalize labels while preserving meaningful metric symbols."""

    value = str(text)
    for source, target in (
        ('*', 'star'),
        ("'", 'prime'),
        ('χ', 'chi'),
        ('⊥', 'perp'),
    ):
        value = value.replace(source, target)
    return norm_token(value)


def canonical_metric_name(
    metric: object,
    specs: list[MetricSpec] | None = None,
    analysis_set: str | None = None,
    patterns_file: Path | None = None,
) -> str | None:
    text = str(metric)
    specs = specs or build_metric_specs(patterns_file or default_patterns_file())
    candidates: dict[str, str] = {}
    for spec in specs:
        if text in {spec.label, spec.pattern_key}:
            return spec.label
    for spec in specs:
        if text == spec.primary_label:
            return spec.label
    for spec in specs:
        candidates.setdefault(canonical_lookup_token(spec.label), spec.label)
        candidates.setdefault(canonical_lookup_token(spec.pattern_key), spec.label)
    for spec in specs:
        candidates.setdefault(canonical_lookup_token(spec.primary_label), spec.label)
    for alias, label in pattern_token_aliases(specs, patterns_file).items():
        candidates.setdefault(canonical_lookup_token(alias), label)
    for alias, label in CANONICAL_ALIASES.items():
        if any(spec.label == label for spec in specs):
            candidates.setdefault(canonical_lookup_token(alias), label)

    canonical = candidates.get(canonical_lookup_token(text))
    if canonical is None:
        return None

    if analysis_set is not None:
        allowed = set(metric_order(specs, analysis_set))
        primary_allowed = {
            spec.primary_label
            for spec in specs
            if spec.label in allowed
        }
        if canonical not in allowed and canonical not in primary_allowed:
            return None
    return canonical


def canonical_metric_from_qsirecon_context(
    variable_name: object,
    qsirecon_suffix: object,
    source_file: object,
    source_tsv: object,
    specs: list[MetricSpec],
) -> str | None:
    variable = norm_token(variable_name)
    context = norm_token(
        ' '.join(
            str(value)
            for value in (qsirecon_suffix, source_file, source_tsv)
            if value is not None
        )
    )
    if not variable or not context:
        return None

    available = {spec.label for spec in specs}
    for recon_key, aliases in QSI_RECON_ALIASES.items():
        if norm_token(recon_key) not in context:
            continue
        label = aliases.get(variable)
        if label in available:
            return label
    return None


def canonical_metric_from_row(
    row: pd.Series,
    patterns_file: Path | None = None,
    spaces: tuple[str, ...] = ('ACPC', 'T1w', 'MNI152NLin2009cAsym'),
) -> str | None:
    specs = build_metric_specs(patterns_file or default_patterns_file())
    source_file = str(row.get('source_file', '') or '')
    source_tsv = str(row.get('source_tsv', '') or '')
    suffix_direct = canonical_metric_from_qsirecon_context(
        row.get('variable_name', ''),
        row.get('qsirecon_suffix', ''),
        source_file,
        source_tsv,
        specs,
    )
    if suffix_direct is not None:
        return suffix_direct

    direct = canonical_metric_name(
        row.get('variable_name', ''),
        specs=specs,
        patterns_file=patterns_file,
    )
    if direct is not None:
        return direct

    haystack = source_file if source_file else source_tsv
    if not haystack:
        return None

    patterns = flattened_patterns(patterns_file)
    for spec in specs:
        rel_pattern = patterns.get(spec.pattern_key)
        if rel_pattern is None:
            continue
        for space in spaces:
            candidate = rel_pattern.format(subject='sub-*', session='ses-*', space=space)
            if fnmatch(haystack, f'*{candidate}'):
                return spec.label
    return None


def add_metric_metadata(
    df: pd.DataFrame,
    metric_col: str,
    patterns_file: Path | None = None,
) -> pd.DataFrame:
    specs = build_metric_specs(patterns_file or default_patterns_file())
    by_label = {spec.label: spec for spec in specs}
    by_primary = {spec.primary_label: spec for spec in specs}

    out = df.copy()
    out['metric'] = [
        canonical_metric_name(value, specs=specs)
        for value in out[metric_col]
    ]
    out = out.dropna(subset=['metric']).copy()
    out['source_image'] = [
        (by_label.get(value) or by_primary.get(value)).source_image
        if (by_label.get(value) or by_primary.get(value)) is not None
        else 'Other'
        for value in out['metric']
    ]
    out['metric_family'] = [
        (by_label.get(value) or by_primary.get(value)).family
        if (by_label.get(value) or by_primary.get(value)) is not None
        else 'Other'
        for value in out['metric']
    ]
    return out


def safe_label(value: object) -> str:
    import re

    return re.sub(r'[^A-Za-z0-9]+', '-', str(value)).strip('-')
