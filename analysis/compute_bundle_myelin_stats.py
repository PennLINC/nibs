#!/usr/bin/env python3
"""Generate QSIRecon-style scalarstats TSVs for warped T1w bundles."""

from __future__ import annotations

import argparse
import os
import re
import sys
from glob import glob
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from bundle_mapping_utils import summarize_bundles
from metric_registry import build_metric_specs, metric_specs_for_analysis

T1W_BUNDLE_GROUPS = {'ihMT', 'MESE', 'MEGRE', 'MP2RAGE', 'T1w/T2w Ratio', 'G-Ratio', 'Q-Ratio', 'QSM'}

BUNDLE_RE = re.compile(r'_bundle-(?P<bundle>.+?)_streamlines\.tck(?:\.gz)?$')
UNDERSCORE_PREFIXES = (
    'ProjectionBasalGanglia',
    'ProjectionBrainstem',
    'Association',
    'Cerebellum',
    'Commissure',
    'CranialNerve',
)


def _project_root() -> Path:
    path = Path(__file__).resolve()
    for parent in path.parents:
        if (
            parent.name == 'code'
            and (parent.parent / 'derivatives').exists()
            and (parent / 'configuration' / 'patterns.json').exists()
        ):
            return parent.parent
        if (
            (parent / 'configuration' / 'patterns.json').exists()
            and (parent / 'analysis').exists()
        ):
            return parent
    return path.parents[1]


def _extract_bundle_name(path: str) -> str:
    match = BUNDLE_RE.search(os.path.basename(path))
    if not match:
        raise ValueError(f'Could not parse bundle name from {path}')
    bundle = match.group('bundle')
    for prefix in UNDERSCORE_PREFIXES:
        if bundle.startswith(prefix + '_'):
            return bundle
        if bundle.startswith(prefix) and len(bundle) > len(prefix):
            return prefix + '_' + bundle[len(prefix) :]
    return bundle


def _metric_specs_t1w(patterns_file: Path):
    return [
        spec
        for spec in metric_specs_for_analysis(
            build_metric_specs(patterns_file),
            'full',
            tissue='wm',
        )
        if spec.group in T1W_BUNDLE_GROUPS
    ]


def _resolve_scalar_specs(
    deriv_dir: str,
    subject: str,
    session: str,
    patterns_file: Path,
) -> tuple[list[dict[str, str]], list[dict[str, object]]]:
    scalar_specs: list[dict[str, str]] = []
    inventory_rows: list[dict[str, object]] = []
    for spec in _metric_specs_t1w(patterns_file):
        rel_pattern = _pattern_for_spec(patterns_file, spec.pattern_key)
        subj_pattern = rel_pattern.format(
            subject=f'sub-{subject}',
            session=session,
            space='T1w',
        )
        matches = sorted(glob(os.path.join(deriv_dir, subj_pattern)))
        inventory_rows.append(
            {
                'subject': f'sub-{subject}',
                'session': session,
                'metric_key': spec.label,
                'primary_label': spec.primary_label,
                'pattern_key': spec.pattern_key,
                'source_image': spec.source_image,
                'space': 'T1w',
                'glob': os.path.join(deriv_dir, subj_pattern),
                'n_matches': len(matches),
                'selected_file': matches[0] if matches else '',
            }
        )
        if not matches:
            print(f'[WARN] Missing scalar for {spec.label}: {subj_pattern}', flush=True)
            continue
        if len(matches) > 1:
            print(
                f'[WARN] Multiple scalar matches for {spec.label}; using first: {matches[0]}',
                flush=True,
            )
        scalar_specs.append(
            {
                'variable_name': spec.label,
                'path': matches[0],
                'source_file': matches[0],
                'qsirecon_suffix': spec.source_image,
            }
        )
    return scalar_specs, inventory_rows


def _pattern_for_spec(patterns_file: Path, pattern_key: str) -> str:
    import json

    with patterns_file.open() as fobj:
        nested = json.load(fobj)
    for group_patterns in nested.values():
        if pattern_key in group_patterns:
            return group_patterns[pattern_key]
    raise KeyError(pattern_key)


def _finalize_qsirecon_style_tsv(
    bundle_stats_file: str,
    out_file: str,
    subject: str,
    session: str,
    bundle_source: str,
    bundle_params_id: str,
    patterns_file: Path,
) -> None:
    df = pd.read_csv(bundle_stats_file, sep='\t')

    # Ensure QSIRecon-style metadata columns exist.
    df['subject_id'] = f'sub-{subject}'
    df['session_id'] = session
    df['task_id'] = pd.NA
    df['dir_id'] = pd.NA
    df['acq_id'] = 'HBCD75'
    df['space_id'] = 'T1w'
    df['rec_id'] = pd.NA
    df['run_id'] = '01'
    df['bundle_source'] = bundle_source
    df['bundle_params_id'] = bundle_params_id

    ordered_cols = [
        'bundle',
        'variable_name',
        'qsirecon_suffix',
        'source_file',
        'zero_proportion',
        'mean',
        'stdev',
        'median',
        'masked_mean',
        'masked_median',
        'masked_stdev',
        'weighted_mean',
        'masked_weighted_mean',
        'subject_id',
        'session_id',
        'task_id',
        'dir_id',
        'acq_id',
        'space_id',
        'rec_id',
        'run_id',
        'bundle_source',
        'bundle_params_id',
    ]
    for col in ordered_cols:
        if col not in df.columns:
            df[col] = pd.NA

    df = df[ordered_cols]
    Path(out_file).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_file, sep='\t', index=False)


def process_subject(
    subject: str,
    deriv_dir: str,
    out_root: str,
    bundle_source: str,
    bundle_params_id: str,
    patterns_file: Path,
) -> None:
    bundles_root = os.path.join(deriv_dir, 'warped_bundles', f'sub-{subject}')
    if not os.path.isdir(bundles_root):
        print(f'[WARN] No warped bundles directory for sub-{subject}: {bundles_root}', flush=True)
        return

    session_dirs = sorted(glob(os.path.join(bundles_root, 'ses-*')))
    if not session_dirs:
        print(f'[WARN] No sessions found for sub-{subject} in {bundles_root}', flush=True)
        return

    for session_dir in session_dirs:
        session = os.path.basename(session_dir)
        dwi_dir = os.path.join(session_dir, 'dwi')
        if not os.path.isdir(dwi_dir):
            print(f'[WARN] Missing dwi directory for sub-{subject} {session}', flush=True)
            continue

        tck_files = sorted(
            glob(
                os.path.join(
                    dwi_dir,
                    f'sub-{subject}_{session}_acq-HBCD75_run-01_space-T1w_model-msmt_bundle-*_streamlines.tck',
                )
            )
        )
        if not tck_files:
            tck_files = sorted(
                glob(
                    os.path.join(
                        dwi_dir,
                        f'sub-{subject}_{session}_acq-HBCD75_run-01_space-T1w_model-msmt_bundle-*_streamlines.tck.gz',
                    )
                )
            )
        if not tck_files:
            print(f'[WARN] No warped bundle TCK files for sub-{subject} {session}', flush=True)
            continue

        bundle_names = [_extract_bundle_name(tck_path) for tck_path in tck_files]
        scalar_specs, inventory_rows = _resolve_scalar_specs(
            deriv_dir,
            subject,
            session,
            patterns_file,
        )
        out_dir = os.path.join(out_root, f'sub-{subject}', session, 'dwi')
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        inventory_file = os.path.join(
            out_dir,
            f'sub-{subject}_{session}_acq-HBCD75_run-01_space-T1w_model-msmt_metric_inventory.tsv',
        )
        pd.DataFrame(inventory_rows).to_csv(inventory_file, sep='\t', index=False)
        print(f'[INFO] Wrote {inventory_file}', flush=True)
        if not scalar_specs:
            print(f'[WARN] No scalar maps found for sub-{subject} {session}', flush=True)
            continue

        # Use first scalar image as the tckmap template (all are in T1w space).
        dwiref = scalar_specs[0]['path']
        bundle_stats_file, _ = summarize_bundles(
            dwiref_image=dwiref,
            tck_files=tck_files,
            bundle_names=bundle_names,
            scalar_specs=scalar_specs,
            out_dir=out_dir,
            bundle_source=bundle_source,
            bundle_params_id=bundle_params_id,
        )

        final_tsv = os.path.join(
            out_dir,
            f'sub-{subject}_{session}_acq-HBCD75_run-01_space-T1w_model-msmt_scalarstats.tsv',
        )
        _finalize_qsirecon_style_tsv(
            bundle_stats_file=bundle_stats_file,
            out_file=final_tsv,
            subject=subject,
            session=session,
            bundle_source=bundle_source,
            bundle_params_id=bundle_params_id,
            patterns_file=patterns_file,
        )
        print(f'[INFO] Wrote {final_tsv}', flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--subject-id', required=True, help='Subject ID without sub- prefix.')
    parser.add_argument(
        '--derivatives-dir',
        default=str(_project_root() / 'derivatives'),
        help='Derivatives root directory.',
    )
    parser.add_argument(
        '--out-root',
        default=None,
        help='Output root for scalarstats TSVs.',
    )
    parser.add_argument(
        '--bundle-source',
        default='warped_msmt',
        help='Value for bundle_source column.',
    )
    parser.add_argument(
        '--bundle-params-id',
        default='default',
        help='Value for bundle_params_id column.',
    )
    parser.add_argument(
        '--patterns-file',
        type=Path,
        default=Path(__file__).resolve().parents[1] / 'configuration' / 'patterns.json',
        help='Metric pattern registry.',
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    derivatives_dir = str(Path(args.derivatives_dir))
    out_root = args.out_root or str(Path(derivatives_dir) / 'bundle_myelin_stats')
    process_subject(
        subject=args.subject_id,
        deriv_dir=derivatives_dir,
        out_root=out_root,
        bundle_source=args.bundle_source,
        bundle_params_id=args.bundle_params_id,
        patterns_file=args.patterns_file,
    )


if __name__ == '__main__':
    main()
