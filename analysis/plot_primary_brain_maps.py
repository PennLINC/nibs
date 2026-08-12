#!/usr/bin/env python3
"""Plot group-average axial slices of the 27 primary NIBS brain maps.

The figure uses one axial slice in MNI152NLin2009cAsym space for every map.
Scalar overlays are displayed with one shared colormap, while each map's color
limits are its 5th and 95th percentiles within GM+WM tissue voxels after
voxelwise averaging across QC-passing images.
"""

from __future__ import annotations

import argparse
import csv
import logging
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

if 'MPLCONFIGDIR' not in os.environ:
    mpl_config_dir = Path(os.environ.get('TMPDIR', '/tmp')) / 'nibs_matplotlib'
    mpl_config_dir.mkdir(parents=True, exist_ok=True)
    os.environ['MPLCONFIGDIR'] = str(mpl_config_dir)
if 'XDG_CACHE_HOME' not in os.environ:
    xdg_cache_dir = Path(os.environ.get('TMPDIR', '/tmp')) / 'nibs_cache'
    xdg_cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ['XDG_CACHE_HOME'] = str(xdg_cache_dir)

import matplotlib

matplotlib.use('Agg')
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
matplotlib.rcParams['svg.fonttype'] = 'none'
matplotlib.rcParams['font.family'] = 'Arial'
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from matplotlib import colors as mcolors
from matplotlib.cm import ScalarMappable

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(REPO_ROOT))

from generate_qc_report import (  # noqa: E402
    first_glob,
    load_canonical,
    resample_image,
)
from metric_registry import (  # noqa: E402
    SOURCE_IMAGE_COLORS,
    MetricSpec,
    build_metric_specs,
    load_patterns,
    metric_plot_label,
    primary_metric_specs,
)

LOGGER = logging.getLogger('primary_maps')

MNI_SPACE = 'MNI152NLin2009cAsym'
DISPLAY_PERCENTILES = (5.0, 95.0)

TABLE_ORDER = {
    'dMRI': (
        'ICVF',
        'MKT',
        'RK',
        'AWF',
        'GFA',
        'FA',
        'MD',
        'RD',
        'NG',
        'NG (Perpendicular)',
        'RTAP',
        'RTOP',
    ),
    'QSM': (
        'QSM-SEPIA-E5-X',
        'QSM-X-R2p-E5-X',
        'QSM-X-R2p-E5-Dia',
        'QSM-X-R2p-E5-Para',
    ),
    'T1w/T2w': (
        'MPRAGE-MyelinW',
        'SPACE-MyelinW',
    ),
    'ihMT': (
        'ihMTR',
        'ihMTsat-B1c',
    ),
    'g-ratio': (
        'G-ihMTR',
        'G-ihMTsat',
    ),
    'R1': (
        'R1',
        'R1-B1c',
    ),
    'MEGRE': (
        'R2*',
        'Q-Ratio-E5-B1c',
    ),
    'MESE': ('R2',),
}

LAYOUT_ROWS = (
    ('dMRI',),
    ('QSM', 'T1w/T2w'),
    ('ihMT', 'g-ratio', 'MEGRE'),
    ('R1', 'MESE'),
)

PANEL_LABELS = {
    'MPRAGE-MyelinW': 'MPRAGE T1w/T2w Ratio',
    'SPACE-MyelinW': 'SPACE T1w/T2w Ratio',
    'QSM-X-R2p-E5-Para': 'QSM-X-R2p-E5-para',
    'QSM-X-R2p-E5-Dia': 'QSM-X-R2p-E5-dia',
}


@dataclass(frozen=True)
class ResolvedMetric:
    spec: MetricSpec
    pattern: str
    source_group: str
    space: str
    paths: tuple[Path, ...]
    contributors: tuple[tuple[str, str], ...]


@dataclass(frozen=True)
class TissueSpace:
    reference: nib.spatialimages.SpatialImage
    mask: np.ndarray
    brain_mask: np.ndarray
    background: np.ndarray
    slice_index: int


@dataclass(frozen=True)
class PreparedPanel:
    metric: ResolvedMetric
    data: np.ndarray
    reference: nib.spatialimages.SpatialImage
    tissue_mask: np.ndarray
    contributor_count: np.ndarray
    limits: tuple[float, float]


def normalize_subject(value: str) -> str:
    token = value.strip()
    return token if token.startswith('sub-') else f'sub-{token}'


def normalize_session(value: str) -> str:
    token = value.strip()
    return token if token.startswith('ses-') else f'ses-{token}'


def session_number(session: str) -> str:
    return normalize_session(session).removeprefix('ses-')


def is_pilot_subject(subject: str) -> bool:
    return normalize_subject(subject).upper().startswith('SUB-PILOT')


def load_qc_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline='') as fobj:
        return list(csv.DictReader(fobj, delimiter='\t'))


def sessions_from_qc(rows: Sequence[dict[str, str]]) -> list[str]:
    sessions: set[str] = set()
    for row in rows:
        for column in row:
            if column.startswith('Session ') and '--' in column:
                sessions.add(normalize_session(column.split('--', 1)[0].replace('Session ', '')))
    return sorted(sessions)


def qc_passes(row: dict[str, str], specs: Sequence[MetricSpec], session: str) -> bool:
    ses_num = session_number(session)
    for spec in specs:
        for modality in spec.qc_modalities:
            column = f'Session {ses_num}--{modality}'
            if row.get(column) != '1':
                return False
    return True


def table_metric_specs(patterns_file: Path) -> list[tuple[str, MetricSpec, str]]:
    patterns = load_patterns(patterns_file)
    specs = primary_metric_specs(build_metric_specs(patterns_file))
    by_primary = {spec.primary_label: spec for spec in specs}
    ordered: list[tuple[str, MetricSpec, str]] = []
    for source_group, labels in TABLE_ORDER.items():
        for label in labels:
            spec = by_primary.get(label)
            if spec is None:
                raise KeyError(f'Primary metric {label!r} was not found in {patterns_file}')
            ordered.append((source_group, spec, patterns[spec.group][spec.pattern_key]))
    if len(ordered) != 27:
        raise RuntimeError(f'Expected 27 primary metrics, found {len(ordered)}')
    return ordered


def metric_space(spec: MetricSpec) -> str:
    return MNI_SPACE


def find_metric_path(
    derivatives: Path,
    subject: str,
    session: str,
    pattern: str,
    space: str,
) -> Path | None:
    formatted = pattern.format(subject=subject, session=session, space=space)
    return first_glob([derivatives / formatted])


def candidate_sessions(
    qc_rows: Sequence[dict[str, str]],
    requested_subjects: Sequence[str] | None,
    requested_sessions: Sequence[str] | None,
) -> list[tuple[str, str, dict[str, str] | None]]:
    if requested_subjects:
        subjects = [normalize_subject(subject) for subject in requested_subjects]
    else:
        subjects = [row['participant_id'] for row in qc_rows]
    subjects = [subject for subject in subjects if not is_pilot_subject(subject)]

    if requested_sessions:
        sessions = [normalize_session(session) for session in requested_sessions]
    else:
        sessions = sessions_from_qc(qc_rows)

    rows_by_subject = {row['participant_id']: row for row in qc_rows}
    return [
        (subject, session, rows_by_subject.get(subject))
        for subject in subjects
        for session in sessions
    ]


def resolve_average_metrics(
    derivatives: Path,
    metric_specs: Sequence[tuple[str, MetricSpec, str]],
    qc_rows: Sequence[dict[str, str]],
    requested_subjects: Sequence[str] | None,
    requested_sessions: Sequence[str] | None,
) -> list[ResolvedMetric]:
    candidates = candidate_sessions(qc_rows, requested_subjects, requested_sessions)
    resolved: list[ResolvedMetric] = []
    missing: list[str] = []
    for source_group, spec, pattern in metric_specs:
        paths: list[Path] = []
        contributors: list[tuple[str, str]] = []
        for subject, session, qc_row in candidates:
            if qc_row is None or not qc_passes(qc_row, [spec], session):
                continue
            space = metric_space(spec)
            path = find_metric_path(derivatives, subject, session, pattern, space)
            if path is None:
                continue
            paths.append(path)
            contributors.append((subject, session))
        if not paths:
            missing.append(spec.primary_label)
            continue
        resolved.append(
            ResolvedMetric(
                spec=spec,
                pattern=pattern,
                source_group=source_group,
                space=metric_space(spec),
                paths=tuple(paths),
                contributors=tuple(contributors),
            )
        )
        LOGGER.info('%s: averaging %d QC-passing image(s)', spec.primary_label, len(paths))
    if missing:
        raise RuntimeError(
            'No QC-passing MNI image was found for these primary metrics: '
            + ', '.join(missing)
        )
    return resolved


def middle_axial_slice(mask: np.ndarray) -> int:
    coordinates = np.argwhere(mask)
    if not coordinates.size:
        return int(mask.shape[2] // 2)
    z_min = int(coordinates[:, 2].min())
    z_max = int(coordinates[:, 2].max())
    return int(round((z_min + z_max) / 2))


def load_mni_tissue_space(
    gm_probseg: Path,
    wm_probseg: Path,
    tissue_probability_threshold: float,
) -> TissueSpace:
    if not gm_probseg.exists():
        raise FileNotFoundError(f'GM probability map not found: {gm_probseg}')
    if not wm_probseg.exists():
        raise FileNotFoundError(f'WM probability map not found: {wm_probseg}')
    reference = load_canonical(gm_probseg)
    gm_probability = np.asarray(reference.get_fdata(), dtype=np.float32)
    wm_image = resample_image(load_canonical(wm_probseg), reference, order=1)
    wm_probability = np.asarray(wm_image.get_fdata(), dtype=np.float32)
    tissue_probability = gm_probability + wm_probability
    mask = tissue_probability >= tissue_probability_threshold
    brain_mask = tissue_probability > 0
    return TissueSpace(
        reference=reference,
        mask=mask,
        brain_mask=brain_mask,
        background=tissue_probability,
        slice_index=middle_axial_slice(mask),
    )


def robust_background_limits(data: np.ndarray, mask: np.ndarray) -> tuple[float, float]:
    values = data[np.isfinite(data) & mask]
    if not values.size:
        values = data[np.isfinite(data)]
    if not values.size:
        return 0.0, 1.0
    low, high = np.percentile(values, [1, 99])
    if low == high:
        high = low + 1.0
    return float(low), float(high)


def scalar_limits(data: np.ndarray, mask: np.ndarray) -> tuple[float, float]:
    values = data[np.isfinite(data) & mask & (np.abs(data) > 0)]
    if not values.size:
        raise RuntimeError('No finite nonzero GM/WM values were found')
    low, high = np.percentile(values, DISPLAY_PERCENTILES)
    if low == high:
        padding = max(abs(float(low)) * 0.05, 1.0)
        low -= padding
        high += padding
    return float(low), float(high)


def voxelwise_average(
    paths: Sequence[Path],
    reference: nib.spatialimages.SpatialImage,
) -> tuple[np.ndarray, np.ndarray]:
    total: np.ndarray | None = None
    count: np.ndarray | None = None
    for path in paths:
        scalar_img = resample_image(load_canonical(path), reference, order=1)
        data = np.asarray(scalar_img.get_fdata(), dtype=np.float32)
        finite = np.isfinite(data)
        if total is None:
            total = np.zeros(data.shape, dtype=np.float64)
            count = np.zeros(data.shape, dtype=np.uint16)
        total[finite] += data[finite]
        count[finite] += 1
    if total is None or count is None:
        raise RuntimeError('Cannot average an empty image list')
    average = np.full(total.shape, np.nan, dtype=np.float32)
    valid = count > 0
    average[valid] = (total[valid] / count[valid]).astype(np.float32)
    return average, count


def prepare_panels(
    resolved_metrics: Sequence[ResolvedMetric],
    tissue_spaces: dict[str, TissueSpace],
) -> list[PreparedPanel]:
    panels: list[PreparedPanel] = []
    for metric in resolved_metrics:
        tissue_space = tissue_spaces[metric.space]
        data, contributor_count = voxelwise_average(metric.paths, tissue_space.reference)
        tissue_mask = tissue_space.mask & np.isfinite(data)
        limits = scalar_limits(data, tissue_mask)
        panels.append(
            PreparedPanel(
                metric=metric,
                data=data,
                reference=tissue_space.reference,
                tissue_mask=tissue_mask,
                contributor_count=contributor_count,
                limits=limits,
            )
        )
    return panels


def axial_slice(data: np.ndarray, slice_index: int) -> np.ndarray:
    return np.rot90(data[:, :, slice_index])


def format_label(label: str) -> str:
    return metric_plot_label(label)


def math_label(label: str, bold: bool = False) -> str:
    text = format_label(label)
    replacements = {
        '₁': r'$_1$',
        '₂': r'$_2$',
        'χ': r'$\chi$',
    }
    for source, target in replacements.items():
        text = text.replace(source, target)
    if bold:
        text = text.replace(r'$_1$', r'$_{\mathbf{1}}$')
        text = text.replace(r'$_2$', r'$_{\mathbf{2}}$')
    return text


def display_label(spec: MetricSpec) -> str:
    return math_label(PANEL_LABELS.get(spec.primary_label, spec.primary_label))


def group_label(group: str) -> str:
    return math_label(group, bold=True)


def add_group_background(fig: plt.Figure, spec, color: str):
    axis = fig.add_subplot(spec)
    axis.set_zorder(0)
    axis.set_facecolor(mcolors.to_rgba(color, 0.13))
    axis.set_xticks([])
    axis.set_yticks([])
    for spine in axis.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.9)
        spine.set_edgecolor(mcolors.to_rgba(color, 0.38))
    return axis


def plot_panel(
    ax: plt.Axes,
    panel: PreparedPanel,
    tissue_space: TissueSpace,
    cmap: str,
    bg_limits: tuple[float, float],
) -> None:
    background = np.asarray(tissue_space.background, dtype=np.float32)
    background = np.ma.masked_where(~tissue_space.brain_mask, background)
    z_index = tissue_space.slice_index
    gray_cmap = plt.get_cmap('gray').copy()
    gray_cmap.set_bad((1, 1, 1, 0))
    scalar_cmap = plt.get_cmap(cmap).copy()
    scalar_cmap.set_bad((1, 1, 1, 0))
    ax.imshow(
        axial_slice(background, z_index),
        cmap=gray_cmap,
        vmin=bg_limits[0],
        vmax=bg_limits[1],
        interpolation='nearest',
    )
    overlay = np.ma.masked_where(~panel.tissue_mask, panel.data)
    ax.imshow(
        axial_slice(overlay, z_index),
        cmap=scalar_cmap,
        vmin=panel.limits[0],
        vmax=panel.limits[1],
        alpha=0.82,
        interpolation='nearest',
    )
    slice_shape = axial_slice(background, z_index).shape
    ax.set_xlim(-0.5, slice_shape[1] - 0.5)
    ax.set_ylim(slice_shape[0] - 0.5, -0.5)
    ax.set_aspect('equal')
    ax.set_title(display_label(panel.metric.spec), fontsize=9.2, pad=2.5)
    ax.set_axis_off()


def plot_figure(
    panels: Sequence[PreparedPanel],
    tissue_spaces: dict[str, TissueSpace],
    output_base: Path,
    cmap: str,
    max_columns: int,
) -> None:
    panels_by_group = {
        group: [panel for panel in panels if panel.metric.source_group == group]
        for group in TABLE_ORDER
    }

    packed_rows: list[list[tuple[str, list[PreparedPanel], int, int, int]]] = []
    for layout_row in LAYOUT_ROWS:
        packed_row: list[tuple[str, list[PreparedPanel], int, int, int]] = []
        current_width = 0
        for group in layout_row:
            group_panels = panels_by_group.get(group, [])
            if not group_panels:
                continue
            width = min(max_columns - current_width, len(group_panels))
            rows = int(np.ceil(len(group_panels) / width))
            packed_row.append((group, group_panels, current_width, width, rows))
            current_width += width
        if packed_row:
            packed_rows.append(packed_row)

    row_panel_counts = [max(group[-1] for group in row) for row in packed_rows]
    height_ratios = [0.3 + rows for rows in row_panel_counts]
    figure_height = 0.28 + 1.55 * sum(row_panel_counts) + 0.26 * len(packed_rows)
    figure_width = 11.2
    fig = plt.figure(figsize=(figure_width, figure_height), facecolor='white')
    outer = fig.add_gridspec(
        len(packed_rows),
        max_columns,
        left=0.035,
        right=0.985,
        top=0.985,
        bottom=0.035,
        hspace=0.18,
        wspace=0.035,
        height_ratios=height_ratios,
    )
    bg_limits = {
        space: robust_background_limits(
            np.asarray(tissue.reference.get_fdata(), dtype=np.float32),
            tissue.brain_mask,
        )
        for space, tissue in tissue_spaces.items()
    }

    for row_index, packed_row in enumerate(packed_rows):
        for group, group_panels, start_column, width, rows in packed_row:
            color = SOURCE_IMAGE_COLORS[group]
            group_spec = outer[row_index, start_column : start_column + width]
            add_group_background(fig, group_spec, color)
            nested = group_spec.subgridspec(
                rows + 1,
                width,
                height_ratios=[0.22] + [1] * rows,
                hspace=0.12,
                wspace=0.05,
            )
            title_axis = fig.add_subplot(nested[0, :])
            title_axis.set_axis_off()
            title_axis.text(
                0.03,
                0.48,
                group_label(group),
                ha='left',
                va='center',
                fontsize=11.5,
                fontweight='bold',
                color=color,
            )

            for panel_index in range(rows * width):
                panel_row = panel_index // width
                panel_column = panel_index % width
                axis = fig.add_subplot(nested[panel_row + 1, panel_column])
                axis.set_zorder(2)
                axis.set_facecolor((1, 1, 1, 0))
                if panel_index >= len(group_panels):
                    axis.set_axis_off()
                    continue
                panel = group_panels[panel_index]
                plot_panel(
                    axis,
                    panel,
                    tissue_spaces[panel.metric.space],
                    cmap,
                    bg_limits[panel.metric.space],
                )

    bottom_used_width = max(
        start_column + width
        for _, _, start_column, width, _ in packed_rows[-1]
    )
    if bottom_used_width < max_columns:
        colorbar_spec = outer[-1, bottom_used_width:max_columns]
        colorbar_host = fig.add_subplot(colorbar_spec)
        colorbar_host.set_axis_off()
        colorbar_axis = colorbar_host.inset_axes([0.17, 0.52, 0.74, 0.075])
    else:
        colorbar_axis = fig.add_axes([0.36, 0.026, 0.28, 0.012])
    colorbar = fig.colorbar(
        ScalarMappable(norm=mcolors.Normalize(vmin=5, vmax=95), cmap=cmap),
        cax=colorbar_axis,
        orientation='horizontal',
    )
    colorbar.set_ticks([5, 50, 95])
    colorbar.set_ticklabels(['5th', '50th', '95th'])
    colorbar.ax.tick_params(labelsize=8.2, length=0)
    colorbar.set_label('Percentile', fontsize=9.0, fontweight='bold', labelpad=2)

    output_base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_base.with_suffix('.png'), dpi=300, bbox_inches='tight')
    fig.savefig(output_base.with_suffix('.pdf'), bbox_inches='tight')
    plt.close(fig)


def write_status_tsv(
    output: Path,
    panels: Sequence[PreparedPanel],
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('w', newline='') as fobj:
        writer = csv.writer(fobj, delimiter='\t')
        writer.writerow(
            [
                'participant_id',
                'session',
                'source_group',
                'metric',
                'space',
                'n_images_for_metric',
                'vmin_5th_gmwm',
                'vmax_95th_gmwm',
                'path',
            ]
        )
        for panel in panels:
            for (subject, session), path in zip(panel.metric.contributors, panel.metric.paths):
                writer.writerow(
                    [
                        subject,
                        session,
                        panel.metric.source_group,
                        panel.metric.spec.primary_label,
                        panel.metric.space,
                        len(panel.metric.paths),
                        f'{panel.limits[0]:.10g}',
                        f'{panel.limits[1]:.10g}',
                        path,
                    ]
                )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        '--project-root',
        default=os.environ.get('NIBS_PROJECT_ROOT', '/cbica/projects/nibs'),
        help='Project root containing derivatives/.',
    )
    parser.add_argument(
        '--derivatives-root',
        help='Override derivatives directory. Defaults to <project-root>/derivatives.',
    )
    parser.add_argument(
        '--patterns-file',
        default=REPO_ROOT / 'configuration' / 'patterns.json',
        type=Path,
        help='Scalar path pattern registry.',
    )
    parser.add_argument(
        '--qc-file',
        default=REPO_ROOT / 'data' / 'manual_qc_modality.tsv',
        type=Path,
        help='Manual modality QC table.',
    )
    parser.add_argument(
        '--gm-probseg',
        default=REPO_ROOT / 'data' / f'tpl-{MNI_SPACE}_res-01_label-GM_probseg.nii.gz',
        type=Path,
        help='MNI-space gray-matter probability map.',
    )
    parser.add_argument(
        '--wm-probseg',
        default=REPO_ROOT / 'data' / f'tpl-{MNI_SPACE}_res-01_label-WM_probseg.nii.gz',
        type=Path,
        help='MNI-space white-matter probability map.',
    )
    parser.add_argument(
        '--tissue-probability-threshold',
        type=float,
        default=0.2,
        help='Minimum GM+WM probability used for percentile scaling.',
    )
    parser.add_argument(
        '--subject-id',
        action='append',
        help='Limit averaging to this subject, with or without sub-. Repeat as needed.',
    )
    parser.add_argument(
        '--session-id',
        action='append',
        help='Limit averaging to this session, with or without ses-. Repeat as needed.',
    )
    parser.add_argument(
        '--output',
        default=REPO_ROOT / 'figures' / 'scalars' / 'primary_brain_maps',
        type=Path,
        help='Output path stem, or a .png/.pdf path whose suffix will be replaced.',
    )
    parser.add_argument(
        '--cmap',
        default='viridis',
        help='Matplotlib colormap used for every scalar overlay.',
    )
    parser.add_argument(
        '--max-columns',
        type=int,
        default=6,
        help='Maximum number of map panels per source-group row.',
    )
    parser.add_argument(
        '--log-level',
        choices=('DEBUG', 'INFO', 'WARNING', 'ERROR'),
        default='INFO',
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.max_columns < 1:
        parser.error('--max-columns must be at least 1')
    if args.tissue_probability_threshold < 0:
        parser.error('--tissue-probability-threshold must be nonnegative')
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format='%(asctime)s | %(levelname)s | %(message)s',
    )

    project_root = Path(args.project_root).expanduser().resolve()
    derivatives = (
        Path(args.derivatives_root).expanduser().resolve()
        if args.derivatives_root
        else project_root / 'derivatives'
    )
    output_base = Path(args.output).expanduser().resolve()
    if output_base.suffix.lower() in {'.png', '.pdf'}:
        output_base = output_base.with_suffix('')

    metric_specs = table_metric_specs(Path(args.patterns_file).expanduser().resolve())
    qc_rows = load_qc_rows(Path(args.qc_file).expanduser().resolve())
    resolved_metrics = resolve_average_metrics(
        derivatives,
        metric_specs,
        qc_rows,
        args.subject_id,
        args.session_id,
    )
    tissue_spaces = {
        MNI_SPACE: load_mni_tissue_space(
            Path(args.gm_probseg).expanduser().resolve(),
            Path(args.wm_probseg).expanduser().resolve(),
            args.tissue_probability_threshold,
        )
    }
    panels = prepare_panels(resolved_metrics, tissue_spaces)
    plot_figure(
        panels,
        tissue_spaces,
        output_base=output_base,
        cmap=args.cmap,
        max_columns=args.max_columns,
    )
    write_status_tsv(
        output_base.with_name(f'{output_base.name}_inputs.tsv'),
        panels,
    )
    LOGGER.info(
        'Wrote %s and %s',
        output_base.with_suffix('.png'),
        output_base.with_suffix('.pdf'),
    )
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
