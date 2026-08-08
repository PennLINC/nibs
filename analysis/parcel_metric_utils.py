"""Utilities for parcel/bundle analyses that consume the metric registry."""

from __future__ import annotations

from pathlib import Path
from fnmatch import fnmatch

import pandas as pd

from metric_registry import MetricSpec, build_metric_specs, metric_order, norm_token


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


def canonical_metric_name(
    metric: object,
    specs: list[MetricSpec] | None = None,
    analysis_set: str | None = None,
) -> str | None:
    text = str(metric)
    specs = specs or build_metric_specs(default_patterns_file())
    candidates: dict[str, str] = {}
    for spec in specs:
        candidates[norm_token(spec.label)] = spec.label
        candidates[norm_token(spec.primary_label)] = spec.label
        candidates[norm_token(spec.pattern_key)] = spec.label

    canonical = candidates.get(norm_token(text))
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


def canonical_metric_from_row(
    row: pd.Series,
    patterns_file: Path | None = None,
    spaces: tuple[str, ...] = ('ACPC', 'T1w', 'MNI152NLin2009cAsym'),
) -> str | None:
    specs = build_metric_specs(patterns_file or default_patterns_file())
    direct = canonical_metric_name(row.get('variable_name', ''), specs=specs)
    if direct is not None:
        return direct

    source_file = str(row.get('source_file', '') or '')
    source_tsv = str(row.get('source_tsv', '') or '')
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
