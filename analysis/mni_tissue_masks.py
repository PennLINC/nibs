#!/usr/bin/env python3
"""Shared deterministic MNI tissue-mask construction utilities."""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Iterable

try:
    import nibabel as nib
    import numpy as np
    from nibabel.processing import resample_from_to
except ImportError:  # pragma: no cover - analysis scripts report missing dependencies
    nib = None
    np = None
    resample_from_to = None


SPACE = 'MNI152NLin2009cAsym'

# FreeSurfer aseg labels. These match the cerebral compartments used by the
# earlier TemplateFlow carpet split and intentionally exclude cerebellum,
# brainstem, ventricles, and ventral diencephalon.
CORTICAL_GM_LABELS = (3, 42)
DEEP_GM_LABELS = (10, 11, 12, 13, 17, 18, 26, 49, 50, 51, 52, 53, 54, 58)
# The aseg divides the corpus callosum into five labels outside 2/41.
WM_LABELS = (2, 41, 251, 252, 253, 254, 255)

GM_TISSUES = ('cortical_gm', 'deep_gm', 'all_gm')
TISSUES = (*GM_TISSUES, 'wm')
TISSUE_TITLES = {
    'cortical_gm': 'Cortical GM',
    'deep_gm': 'Deep GM',
    'all_gm': 'All GM',
    'wm': 'WM',
    'gmwm': 'GM+WM',
}


def require_dependencies() -> None:
    missing = [
        name
        for name, module in (
            ('nibabel', nib),
            ('numpy', np),
            ('nibabel.processing', resample_from_to),
        )
        if module is None
    ]
    if missing:
        raise RuntimeError(
            'Missing required tissue-mask packages: '
            f'{", ".join(missing)}. Activate the NIBS processing environment first.'
        )


def load_like(path: Path, reference, order: int) -> np.ndarray:
    image = nib.load(str(path))
    if image.shape[:3] != reference.shape[:3] or not np.allclose(
        image.affine, reference.affine, atol=1e-4
    ):
        image = resample_from_to(image, reference, order=order)
    return np.asarray(image.get_fdata(), dtype=np.float32)


def erode_mask_mm(mask: np.ndarray, reference, erosion_mm: float) -> np.ndarray:
    mask_3d = np.asarray(mask, dtype=bool).reshape(reference.shape[:3])
    if erosion_mm <= 0:
        return mask_3d.reshape(-1)
    from scipy.ndimage import distance_transform_edt

    voxel_sizes = tuple(float(value) for value in nib.affines.voxel_sizes(reference.affine))
    distance = distance_transform_edt(mask_3d, sampling=voxel_sizes)
    return (distance > float(erosion_mm)).reshape(-1)


def metric_registry_tissue(tissue: str) -> str:
    """Map anatomical GM compartments to the registry's GM eligibility key."""

    return 'gm' if tissue in GM_TISSUES else tissue


def build_template_tissue_masks(
    template_dseg: Path,
    gm_erosion_mm: float = 0.0,
    wm_erosion_mm: float = 0.0,
) -> tuple[object, dict[str, np.ndarray]]:
    """Build fixed cortical GM, deep GM, all-GM, and WM template masks."""

    require_dependencies()
    reference = nib.load(str(template_dseg))
    segmentation = np.rint(load_like(template_dseg, reference, order=0)).astype(np.int16)

    cortical_raw = np.isin(segmentation, CORTICAL_GM_LABELS)
    deep_raw = np.isin(segmentation, DEEP_GM_LABELS)
    all_gm_raw = cortical_raw | deep_raw
    wm_raw = np.isin(segmentation, WM_LABELS)
    masks = {
        'cortical_gm': erode_mask_mm(cortical_raw, reference, gm_erosion_mm),
        'deep_gm': erode_mask_mm(deep_raw, reference, gm_erosion_mm),
        'all_gm': erode_mask_mm(all_gm_raw, reference, gm_erosion_mm),
        'wm': erode_mask_mm(wm_raw, reference, wm_erosion_mm),
    }
    for tissue, mask in masks.items():
        if not np.any(mask):
            raise RuntimeError(
                f'Template {TISSUE_TITLES[tissue]} mask is empty after erosion: {template_dseg}'
            )
    masks['gmwm'] = masks['all_gm'] | masks['wm']
    return reference, masks


def smriprep_anat_dirs(
    derivatives: Path,
    subject: str,
    session: str,
) -> tuple[Path, ...]:
    root = derivatives / 'smriprep' / subject
    return (root / 'anat', root / session / 'anat')


def _preferred_match(paths: Iterable[Path]) -> Path | None:
    matches = sorted(set(paths))
    if not matches:
        return None
    for token in ('_acq-MPRAGE_', '_rec-refaced_', '_run-01_'):
        preferred = [path for path in matches if token in path.name]
        if preferred:
            matches = preferred
    return matches[0]


def find_smriprep_dseg(
    derivatives: Path,
    subject: str,
    session: str,
    space: str = SPACE,
) -> Path | None:
    for anat_dir in smriprep_anat_dirs(derivatives, subject, session):
        match = _preferred_match(anat_dir.glob(f'{subject}*_space-{space}_dseg.nii*'))
        if match is not None:
            return match
    return None


def find_native_ribbon(
    derivatives: Path,
    subject: str,
    session: str,
) -> Path | None:
    for anat_dir in smriprep_anat_dirs(derivatives, subject, session):
        candidates = (
            path
            for path in anat_dir.glob(f'{subject}*_desc-ribbon_mask.nii*')
            if '_space-' not in path.name
        )
        match = _preferred_match(candidates)
        if match is not None:
            return match
    return None


def mni_ribbon_path(native_ribbon: Path, space: str = SPACE) -> Path:
    name = native_ribbon.name
    for extension in ('.nii.gz', '.nii'):
        suffix = f'_desc-ribbon_mask{extension}'
        if name.endswith(suffix):
            prefix = name[: -len(suffix)]
            return native_ribbon.with_name(f'{prefix}_space-{space}_desc-ribbon_mask.nii.gz')
    raise ValueError(f'Unexpected ribbon filename: {native_ribbon}')


def find_t1w_to_mni_transform(
    anat_dir: Path,
    subject: str,
    space: str = SPACE,
) -> Path | None:
    return _preferred_match(anat_dir.glob(f'{subject}*_from-T1w_to-{space}_mode-image_xfm.h5'))


def find_existing_mni_ribbon(
    derivatives: Path,
    subject: str,
    session: str,
    space: str = SPACE,
) -> Path | None:
    for anat_dir in smriprep_anat_dirs(derivatives, subject, session):
        match = _preferred_match(anat_dir.glob(f'{subject}*_space-{space}_desc-ribbon_mask.nii*'))
        if match is not None:
            return match
    return None


def ensure_mni_ribbon(
    derivatives: Path,
    subject: str,
    session: str,
    reference_file: Path,
    space: str = SPACE,
    ants_apply_transforms: str = 'antsApplyTransforms',
) -> Path:
    """Return a cached MNI ribbon, creating it beside the native ribbon if needed."""

    native_ribbon = find_native_ribbon(derivatives, subject, session)
    if native_ribbon is None:
        existing = find_existing_mni_ribbon(derivatives, subject, session, space)
        if existing is not None:
            return existing
        raise FileNotFoundError(f'No native-space cortical ribbon found for {subject} {session}')

    output_file = mni_ribbon_path(native_ribbon, space)
    if output_file.exists():
        return output_file

    transform = find_t1w_to_mni_transform(native_ribbon.parent, subject, space)
    if transform is None:
        raise FileNotFoundError(f'No T1w-to-{space} transform found beside {native_ribbon}')
    executable = shutil.which(ants_apply_transforms)
    if executable is None:
        raise RuntimeError(
            f'Could not find {ants_apply_transforms}; it is required to create {output_file}'
        )

    output_file.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=output_file.parent,
        prefix=f'.{output_file.name}.',
        suffix='.nii.gz',
        delete=False,
    ) as temp_file:
        temp_path = Path(temp_file.name)
    temp_path.unlink(missing_ok=True)
    command = [
        executable,
        '--dimensionality',
        '3',
        '--input',
        str(native_ribbon),
        '--reference-image',
        str(reference_file),
        '--output',
        str(temp_path),
        '--interpolation',
        'GenericLabel',
        '--transform',
        str(transform),
    ]
    try:
        subprocess.run(command, check=True, capture_output=True, text=True)
        if not temp_path.exists():
            raise RuntimeError(f'antsApplyTransforms did not create {temp_path}')
        os.replace(temp_path, output_file)
    except subprocess.CalledProcessError as exc:
        detail = exc.stderr.strip() or exc.stdout.strip() or str(exc)
        raise RuntimeError(f'Failed to transform cortical ribbon: {detail}') from exc
    finally:
        temp_path.unlink(missing_ok=True)
    return output_file


def build_subject_tissue_masks(
    dseg: Path,
    mni_ribbon: Path,
    template_dseg: Path,
    gm_erosion_mm: float = 0.0,
    wm_erosion_mm: float = 0.0,
) -> tuple[object, dict[str, np.ndarray]]:
    """Build subject-specific masks from a cortical ribbon and MNI dseg."""

    require_dependencies()
    reference = nib.load(str(dseg))
    segmentation = np.rint(load_like(dseg, reference, order=0)).astype(np.int16)
    ribbon = load_like(mni_ribbon, reference, order=0) > 0
    template_segmentation = np.rint(load_like(template_dseg, reference, order=0)).astype(np.int16)

    all_gm_raw = segmentation == 1
    masks = {
        'cortical_gm': erode_mask_mm(ribbon, reference, gm_erosion_mm),
        'deep_gm': erode_mask_mm(
            all_gm_raw & np.isin(template_segmentation, DEEP_GM_LABELS),
            reference,
            gm_erosion_mm,
        ),
        'all_gm': erode_mask_mm(all_gm_raw, reference, gm_erosion_mm),
        'wm': erode_mask_mm(segmentation == 2, reference, wm_erosion_mm),
    }
    for tissue, mask in masks.items():
        if not np.any(mask):
            raise RuntimeError(
                f'Subject {TISSUE_TITLES[tissue]} mask is empty after erosion: {dseg}'
            )
    return reference, masks


def subject_mask_source(
    dseg: Path,
    mni_ribbon: Path,
    template_dseg: Path,
) -> str:
    return f'dseg={dseg};ribbon={mni_ribbon};template_dseg={template_dseg}'
