#!/usr/bin/env python3
"""Create voxelwise fiber-population count maps from QSIRecon MSMT FODs.

For each subject/session, this script runs MRtrix inside a QSIRecon Apptainer
container to segment the WM FOD image into fixels and count the number of
retained fixels per voxel. It then resamples an anatomical segmentation to the
count-map grid, preferring QSIPrep's ACPC aseg when present, and writes a
WM-restricted count distribution next to the NIfTI output.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

ASEG_WM_LABELS = (2, 41)
SMRIPREP_WM_LABELS = (2,)


@dataclass(frozen=True)
class SessionInputs:
    subject: str
    session: str
    fod: Path
    brain_mask: Path
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


def prefer_fod(candidates: Iterable[Path]) -> Path | None:
    paths = sorted(set(candidates))
    if not paths:
        return None
    wm_labeled = [path for path in paths if '_label-WM_' in path.name]
    if wm_labeled:
        return wm_labeled[0]
    non_tissue_labeled = [
        path
        for path in paths
        if '_label-CSF_' not in path.name and '_label-GM_' not in path.name
    ]
    paths = non_tissue_labeled or paths
    preferred_terms = ('wmfod', 'wm_fod', 'wmFOD', 'fod')
    for term in preferred_terms:
        matches = [path for path in paths if term.lower() in path.name.lower()]
        if matches:
            return matches[0]
    return paths[0]


def discover_subjects(derivatives: Path) -> list[str]:
    roots = (
        derivatives / 'qsirecon' / 'derivatives' / 'qsirecon-MSMTAutoTrack',
        derivatives / 'smriprep',
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


def discover_sessions(derivatives: Path, subject: str) -> list[str]:
    roots = (
        derivatives / 'qsirecon' / 'derivatives' / 'qsirecon-MSMTAutoTrack' / subject,
        derivatives / 'smriprep' / subject,
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


def find_fod(derivatives: Path, subject: str, session: str) -> Path | None:
    dwi_dir = (
        derivatives
        / 'qsirecon'
        / 'derivatives'
        / 'qsirecon-MSMTAutoTrack'
        / subject
        / session
        / 'dwi'
    )
    candidates: list[Path] = []
    for pattern in (
        '*_model-msmtcsd_param-fod_label-WM_dwimap.mif*',
        '*_param-fod_label-WM_*.mif*',
        '*label-WM*.mif*',
        '*wmfod*.mif*',
        '*wmFOD*.mif*',
        '*WMFOD*.mif*',
        '*wm_fod*.mif*',
        '*wm-FOD*.mif*',
        '*desc-wmFOD*.mif*',
        '*model-CSD*param-wmFOD*.mif*',
        '*fod*.mif*',
        '*FOD*.mif*',
    ):
        candidates.extend(dwi_dir.glob(pattern))
    return prefer_fod(path for path in candidates if path.is_file())


def collect_inputs(
    derivatives: Path,
    subject: str,
    session: str,
    wm_labels: tuple[int, ...] | None,
) -> SessionInputs | None:
    fod = find_fod(derivatives, subject, session)
    brain_mask = first_glob(
        (
            derivatives
            / 'qsiprep'
            / subject
            / session
            / 'dwi'
            / f'{subject}_{session}_space-ACPC_desc-brain_mask.nii*',
            derivatives
            / 'qsiprep'
            / subject
            / session
            / 'dwi'
            / f'{subject}_{session}_*desc-brain_mask.nii*',
            derivatives
            / 'qsiprep'
            / subject
            / 'dwi'
            / f'{subject}_space-ACPC_desc-brain_mask.nii*',
            derivatives
            / 'qsiprep'
            / subject
            / 'dwi'
            / f'{subject}_*desc-brain_mask.nii*',
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
            derivatives
            / 'qsiprep'
            / subject
            / 'anat'
            / f'{subject}_space-ACPC_desc-aseg_dseg.nii*',
            derivatives
            / 'qsiprep'
            / subject
            / 'anat'
            / f'{subject}_*space-ACPC*desc-aseg_dseg.nii*',
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
            derivatives
            / 'smriprep'
            / subject
            / session
            / 'anat'
            / f'{subject}_{session}_*dseg.nii*',
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
        required_transform = None
    else:
        anatomical_dseg = smriprep_dseg
        dseg_source = 'smriprep_dseg'
        selected_wm_labels = wm_labels or SMRIPREP_WM_LABELS
        required_transform = t1w_to_acpc_xfm

    prerequisites = (
        ('FOD', fod),
        ('ACPC brain mask', brain_mask),
        ('QSIPrep ACPC aseg or sMRIPrep tissue dseg', anatomical_dseg),
    )
    if dseg_source == 'smriprep_dseg':
        prerequisites = (
            *prerequisites,
            ('T1w-to-ACPC transform for sMRIPrep dseg fallback', required_transform),
        )
    if any(value is None for _, value in prerequisites):
        missing = [
            name
            for name, value in prerequisites
            if value is None
        ]
        print(f'Skipping {subject} {session}: missing {", ".join(missing)}')
        return None
    return SessionInputs(
        subject,
        session,
        fod,
        brain_mask,
        anatomical_dseg,
        dseg_source,
        selected_wm_labels,
        required_transform,
    )


def bind_args(paths: Iterable[Path]) -> list[str]:
    dirs = sorted({str(path.resolve()) for path in paths})
    args: list[str] = []
    for directory in dirs:
        args.extend(('-B', f'{directory}:{directory}'))
    return args


def run_container(
    runtime: str,
    image: Path,
    command: list[str],
    bind_dirs: Iterable[Path],
    *,
    dry_run: bool,
) -> None:
    full_command = [
        runtime,
        'exec',
        '--cleanenv',
        *bind_args(bind_dirs),
        str(image),
        *command,
    ]
    print(' '.join(full_command), flush=True)
    if dry_run:
        return
    subprocess.run(full_command, check=True)


def create_count_map(
    inputs: SessionInputs,
    output: Path,
    work_dir: Path,
    args: argparse.Namespace,
) -> Path:
    output.parent.mkdir(parents=True, exist_ok=True)
    work_dir.mkdir(parents=True, exist_ok=True)
    temp_parent = work_dir / inputs.subject / inputs.session
    temp_parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix='fixel_count_', dir=temp_parent) as temp_name:
        temp_dir = Path(temp_name)
        fixel_dir = temp_dir / 'fixels'
        mask_binary = temp_dir / 'mask_binary.mif'
        raw_count = temp_dir / 'raw_fixel_count.mif'
        masked_count = temp_dir / 'masked_fixel_count.mif'
        bind_dirs = {
            inputs.fod.parent,
            inputs.brain_mask.parent,
            output.parent,
            temp_dir,
        }
        run_container(
            args.runtime,
            args.apptainer_image,
            ['mrcalc', str(inputs.brain_mask), '0', '-gt', str(mask_binary), '-force'],
            bind_dirs,
            dry_run=args.dry_run,
        )
        fod2fixel = [
            'fod2fixel',
            str(inputs.fod),
            str(fixel_dir),
            '-mask',
            str(mask_binary),
            '-afd',
            'afd.mif',
            '-peak_amp',
            'peak_amp.mif',
            '-fmls_peak_value',
            str(args.peak_threshold),
            '-nthreads',
            str(args.nthreads),
            '-force',
        ]
        if args.max_fixels is not None:
            fod2fixel.extend(('-maxnum', str(args.max_fixels)))
        run_container(args.runtime, args.apptainer_image, fod2fixel, bind_dirs, dry_run=args.dry_run)
        run_container(
            args.runtime,
            args.apptainer_image,
            [
                'fixel2voxel',
                str(fixel_dir / 'afd.mif'),
                'count',
                str(raw_count),
                '-nthreads',
                str(args.nthreads),
                '-force',
            ],
            bind_dirs,
            dry_run=args.dry_run,
        )
        run_container(
            args.runtime,
            args.apptainer_image,
            ['mrcalc', str(raw_count), str(mask_binary), '-mult', str(masked_count), '-force'],
            bind_dirs,
            dry_run=args.dry_run,
        )
        run_container(
            args.runtime,
            args.apptainer_image,
            ['mrconvert', str(masked_count), str(output), '-datatype', 'uint8', '-force'],
            bind_dirs,
            dry_run=args.dry_run,
        )
    return output


def resample_anatomical_dseg_to_count_grid(
    inputs: SessionInputs,
    count_map: Path,
    out_dseg: Path,
    args: argparse.Namespace,
) -> None:
    out_dseg.parent.mkdir(parents=True, exist_ok=True)
    bind_dirs = {
        inputs.anatomical_dseg.parent,
        count_map.parent,
        out_dseg.parent,
    }
    command = [
        'antsApplyTransforms',
        '-d',
        '3',
        '-i',
        str(inputs.anatomical_dseg),
        '-r',
        str(count_map),
        '-o',
        str(out_dseg),
        '-n',
        'GenericLabel',
        '--float',
        '1',
    ]
    if inputs.t1w_to_acpc_xfm is not None:
        bind_dirs.add(inputs.t1w_to_acpc_xfm.parent)
        command.extend(('-t', str(inputs.t1w_to_acpc_xfm)))
    run_container(
        args.runtime,
        args.apptainer_image,
        command,
        bind_dirs,
        dry_run=args.dry_run,
    )


def write_wm_distribution(
    inputs: SessionInputs,
    count_map: Path,
    wm_dseg: Path,
    out_tsv: Path,
    out_json: Path,
    args: argparse.Namespace,
) -> None:
    if args.dry_run:
        return
    import nibabel as nib
    import numpy as np

    count_img = nib.load(str(count_map))
    count_data = np.asarray(count_img.get_fdata(), dtype=np.float32)
    label_data = np.rint(nib.load(str(wm_dseg)).get_fdata()).astype(np.int16)
    wm_label_mask = np.isin(label_data, inputs.wm_labels)
    wm_mask = wm_label_mask & np.isfinite(count_data)
    wm_values = count_data[wm_mask]
    if wm_values.size == 0:
        raise RuntimeError(f'No WM voxels found after resampling {inputs.anatomical_dseg}')
    if not np.allclose(wm_values, np.rint(wm_values), atol=0.01):
        raise RuntimeError(f'Count map does not contain discrete values: {count_map}')

    wm_counts = np.rint(wm_values).astype(np.int16)
    max_count = int(wm_counts.max())
    total = int(wm_counts.size)
    out_tsv.parent.mkdir(parents=True, exist_ok=True)
    with out_tsv.open('w') as fobj:
        fobj.write('population_count\tn_voxels\tpercent_wm_voxels\n')
        for population in range(max_count + 1):
            n_voxels = int(np.count_nonzero(wm_counts == population))
            percent = 100 * n_voxels / total
            fobj.write(f'{population}\t{n_voxels}\t{percent:.6f}\n')

    metadata = {
        'subject': inputs.subject,
        'session': inputs.session,
        'count_map': str(count_map),
        'fod': str(inputs.fod),
        'brain_mask': str(inputs.brain_mask),
        'anatomical_dseg': str(inputs.anatomical_dseg),
        'dseg_source': inputs.dseg_source,
        't1w_to_acpc_xfm': str(inputs.t1w_to_acpc_xfm) if inputs.t1w_to_acpc_xfm else None,
        'wm_labels': list(inputs.wm_labels),
        'total_wm_voxels': total,
        'peak_threshold': args.peak_threshold,
        'max_fixels': args.max_fixels,
        'nthreads': args.nthreads,
    }
    out_json.write_text(json.dumps(metadata, indent=2) + '\n')


def output_stem(subject: str, session: str) -> str:
    return f'{subject}_{session}_space-ACPC_desc-fiberpopulation_count'


def process_session(inputs: SessionInputs, args: argparse.Namespace) -> None:
    out_dir = args.output_dir / inputs.subject / inputs.session
    stem = output_stem(inputs.subject, inputs.session)
    count_map = out_dir / f'{stem}.nii.gz'
    stats_tsv = out_dir / f'{stem}.tsv'
    metadata_json = out_dir / f'{stem}.json'
    wm_dseg = out_dir / f'{inputs.subject}_{inputs.session}_space-ACPC_desc-anatomical_dseg.nii.gz'

    if count_map.exists() and stats_tsv.exists() and not args.force:
        print(f'Skipping existing outputs for {inputs.subject} {inputs.session}')
        return
    if count_map.exists() and args.force:
        count_map.unlink()
    if stats_tsv.exists() and args.force:
        stats_tsv.unlink()
    if metadata_json.exists() and args.force:
        metadata_json.unlink()
    if wm_dseg.exists() and args.force:
        wm_dseg.unlink()

    print(f'Processing {inputs.subject} {inputs.session}', flush=True)
    create_count_map(inputs, count_map, args.work_dir, args)
    resample_anatomical_dseg_to_count_grid(inputs, count_map, wm_dseg, args)
    write_wm_distribution(inputs, count_map, wm_dseg, stats_tsv, metadata_json, args)
    if not args.keep_warped_dseg and wm_dseg.exists():
        wm_dseg.unlink()
    print(f'Wrote {count_map}')
    print(f'Wrote {stats_tsv}')


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--derivatives-dir',
        type=Path,
        default=Path('~/derivatives').expanduser(),
        help='Derivatives root containing qsiprep, qsirecon, smriprep, and t1w_registration.',
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=Path('~/derivatives/fixel_count').expanduser(),
        help='Directory where sub-*/ses-* count maps and TSVs are written.',
    )
    parser.add_argument(
        '--work-dir',
        type=Path,
        default=Path('~/derivatives/fixel_count/work').expanduser(),
        help='Directory for temporary MRtrix files.',
    )
    parser.add_argument(
        '--apptainer-image',
        type=Path,
        default=Path('~/apptainer/qsirecon-26.0.0.sif').expanduser(),
        help='QSIRecon Apptainer image with MRtrix and ANTs.',
    )
    parser.add_argument('--runtime', default='apptainer', help='Container runtime executable.')
    parser.add_argument('--subject-id', action='append', help='Subject(s), with or without sub-.')
    parser.add_argument('--session-id', action='append', help='Session(s), with or without ses-.')
    parser.add_argument(
        '--wm-label',
        action='append',
        type=int,
        dest='wm_labels',
        help=(
            'WM label to count in the anatomical dseg. May be repeated. '
            'Defaults to 2 and 41 for QSIPrep aseg, or 2 for sMRIPrep dseg fallback.'
        ),
    )
    parser.add_argument('--peak-threshold', type=float, default=0.1)
    parser.add_argument('--nthreads', type=int, default=0)
    parser.add_argument('--max-fixels', type=int, default=None)
    parser.add_argument('--force', action='store_true', help='Overwrite existing outputs.')
    parser.add_argument(
        '--keep-warped-dseg',
        action='store_true',
        help='Keep the intermediate count-grid anatomical dseg in each output directory.',
    )
    parser.add_argument('--dry-run', action='store_true', help='Print commands without running them.')
    args = parser.parse_args()
    args.derivatives_dir = args.derivatives_dir.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()
    args.work_dir = args.work_dir.expanduser().resolve()
    args.apptainer_image = args.apptainer_image.expanduser().resolve()
    args.wm_labels = tuple(args.wm_labels) if args.wm_labels else None
    return args


def main() -> None:
    args = parse_args()
    if not args.dry_run:
        if shutil.which(args.runtime) is None:
            raise RuntimeError(f'Container runtime not found: {args.runtime}')
        if not args.apptainer_image.is_file():
            raise FileNotFoundError(args.apptainer_image)

    subjects = (
        [normalize_subject(subject) for subject in args.subject_id]
        if args.subject_id
        else discover_subjects(args.derivatives_dir)
    )
    if not subjects:
        raise RuntimeError(f'No subjects found under {args.derivatives_dir}')

    for subject in subjects:
        sessions = (
            [normalize_session(session) for session in args.session_id]
            if args.session_id
            else discover_sessions(args.derivatives_dir, subject)
        )
        if not sessions:
            print(f'Skipping {subject}: no sessions found')
            continue
        for session in sessions:
            inputs = collect_inputs(args.derivatives_dir, subject, session, args.wm_labels)
            if inputs is not None:
                process_session(inputs, args)


if __name__ == '__main__':
    main()
