"""Process QSM data.

Steps:

1.  Complex MP-PCA denoise the echoes and compute the RMS across the denoised
    magnitude images as the MEGRE reference.
2.  Calculate R2* map.
3.  Coregister the RMS magnitude reference to the preprocessed T1w image from sMRIPrep.
4.  Warp T1w mask from T1w space into the QSM space by applying the inverse of the coregistration
    transform.
5.  Apply the mask in QSM space to magnitude images.

Notes:

- The R2* map is calculated using a complex single-pool nonlinear least squares
  (NLLS) fit.
- This must be run after sMRIPrep and process_mese.py.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pprint import pformat

import ants
import nibabel as nb
import numpy as np
from bids.layout import BIDSLayout, Query
from nilearn import image, masking
from nireports.assembler.report import Report

from utils import (
    calculate_r_squared,
    coregister_to_t1,
    fit_complex_r2star,
    get_filename,
    load_config,
    plot_coregistration,
    plot_denoise,
    plot_residual,
    plot_scalar_map,
    run_command,
)

CFG = load_config()
CODE_DIR = CFG['code_dir']

# Complex MP-PCA denoising + complex-NLLS R2* estimation, ported from
# complex-tedana/run_megre_fit.py. The fit is a complex single-pool NLLS model
# (S0, R2*, off-resonance frequency, initial phase), implemented in
# utils.fit_complex_r2star. dwidenoise is the only external tool, resolved on
# PATH and overridable with the DWIDENOISE environment variable. dwidenoise
# requires odd -extent values, so the Doniza-style 2x2x2 window is not available
# and 3,3,3 is the default.
DWIDENOISE_EXE = os.environ.get('DWIDENOISE', 'dwidenoise')
DWIDENOISE_EXTENT = os.environ.get('DWIDENOISE_EXTENT', '3,3,3')
MEGRE_FIT_N_THREADS = int(os.environ.get('MEGRE_FIT_N_THREADS', str(os.cpu_count() or 1)))


def _rescale_phase_to_radians(phase_files: list[str], out_dir: str) -> list[str]:
    """Rescale arbitrary-unit MEGRE phase images to radians in [-pi, pi].

    nibs MEGRE phase is stored with ``Units: arbitrary``, so a per-image min/max
    rescale to ``[-pi, pi]`` recovers radians. This is a linear rescale only (no
    smoothing/interpolation), so it is safe to run before complex denoising.
    """
    os.makedirs(out_dir, exist_ok=True)
    scaled_paths = []
    for path in phase_files:
        img = nb.load(path)
        data = np.asarray(img.dataobj, dtype=np.float32)
        finite = np.isfinite(data)
        if not np.any(finite):
            raise ValueError(f'No finite phase values found in {path}.')

        phase_min = float(np.min(data[finite]))
        phase_max = float(np.max(data[finite]))
        if phase_max <= phase_min:
            raise ValueError(f'Cannot rescale phase image with constant values: {path}')

        scaled = (data - phase_min) / (phase_max - phase_min)
        scaled = (scaled * (2.0 * np.pi)) - np.pi
        scaled[~finite] = 0.0

        out_path = os.path.join(
            out_dir, os.path.basename(path).replace('_part-phase_', '_part-phaseRad_')
        )
        header = img.header.copy()
        header.set_data_dtype(np.float32)
        nb.Nifti1Image(scaled.astype(np.float32), img.affine, header).to_filename(out_path)
        scaled_paths.append(out_path)

    return scaled_paths


def _build_complex_megre(mag_files: list[str], phase_rad_files: list[str], out_path: str) -> str:
    """Build a 4D complex MEGRE array (x, y, z, echo) for dwidenoise input."""
    ref_img = nb.load(mag_files[0])
    ref_shape = ref_img.shape
    complex_data = np.zeros((*ref_shape, len(mag_files)), dtype=np.complex64)
    for idx, (mag_path, phase_path) in enumerate(zip(mag_files, phase_rad_files)):
        mag = np.asarray(nb.load(mag_path).dataobj, dtype=np.float64)
        phase = np.asarray(nb.load(phase_path).dataobj, dtype=np.float64)
        complex_data[..., idx] = mag * np.exp(1j * phase)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    header = ref_img.header.copy()
    header.set_data_dtype(np.complex64)
    nb.Nifti1Image(complex_data, ref_img.affine, header).to_filename(out_path)
    return out_path


def _split_complex_to_mag_phase(
    denoised_path: str,
    mag_files: list[str],
    phase_rad_files: list[str],
    mag_out_dir: str,
    phase_out_dir: str,
) -> tuple[list[str], list[str]]:
    """Derive per-echo denoised magnitude (abs) and phase (angle, radians)."""
    img = nb.load(denoised_path)
    data = np.asarray(img.dataobj)
    if data.shape[-1] != len(mag_files):
        raise ValueError(f'Denoised data has {data.shape[-1]} echoes; expected {len(mag_files)}.')

    os.makedirs(mag_out_dir, exist_ok=True)
    os.makedirs(phase_out_dir, exist_ok=True)
    mag_paths = []
    phase_paths = []
    for idx, (mag_src, phase_src) in enumerate(zip(mag_files, phase_rad_files)):
        mag = np.abs(data[..., idx]).astype(np.float32)
        phase = np.angle(data[..., idx]).astype(np.float32)

        mag_header = nb.load(mag_src).header.copy()
        mag_header.set_data_dtype(np.float32)
        mag_path = os.path.join(
            mag_out_dir,
            os.path.basename(mag_src).replace('_MEGRE.nii.gz', '_desc-denoised_MEGRE.nii.gz'),
        )
        nb.Nifti1Image(mag, img.affine, mag_header).to_filename(mag_path)
        mag_paths.append(mag_path)

        phase_header = nb.load(phase_src).header.copy()
        phase_header.set_data_dtype(np.float32)
        phase_path = os.path.join(
            phase_out_dir,
            os.path.basename(phase_src).replace('_MEGRE.nii.gz', '_desc-denoised_MEGRE.nii.gz'),
        )
        nb.Nifti1Image(phase, img.affine, phase_header).to_filename(phase_path)
        phase_paths.append(phase_path)

    return mag_paths, phase_paths


def denoise_complex_megre(
    mag_files: list[str],
    phase_files: list[str],
    temp_dir: str,
    n_threads: int = MEGRE_FIT_N_THREADS,
    extent: str = DWIDENOISE_EXTENT,
) -> tuple[list[str], list[str]]:
    """Complex MP-PCA denoise MEGRE echoes and return denoised mag/phase paths.

    Mirrors the preprocessing in complex-tedana/run_megre_fit.py: rescale phase
    to radians, build a complex MEGRE volume, run ``dwidenoise`` complex MP-PCA,
    then split the denoised complex data back into magnitude and phase.
    """
    dwidenoise = shutil.which(DWIDENOISE_EXE)
    if dwidenoise is None:
        raise RuntimeError(f'Could not find {DWIDENOISE_EXE!r}. Install MRtrix3 or set DWIDENOISE.')

    phase_rad_files = _rescale_phase_to_radians(phase_files, os.path.join(temp_dir, 'phase_rad'))
    complex_path = _build_complex_megre(
        mag_files, phase_rad_files, os.path.join(temp_dir, 'megre_complex.nii.gz')
    )
    denoised_path = os.path.join(temp_dir, 'megre_complex_desc-denoised.nii.gz')
    noise_path = os.path.join(temp_dir, 'megre_desc-noise.nii.gz')
    run_command(
        [
            dwidenoise,
            '-force',
            '-extent',
            extent,
            '-noise',
            noise_path,
            '-nthreads',
            str(n_threads),
            complex_path,
            denoised_path,
        ]
    )
    return _split_complex_to_mag_phase(
        denoised_path,
        mag_files,
        phase_rad_files,
        os.path.join(temp_dir, 'mag'),
        os.path.join(temp_dir, 'phase'),
    )


def _load_echo_stack(paths: list[str]) -> tuple[np.ndarray, nb.Nifti1Image]:
    """Load echo-wise 3D NIfTIs and stack them on a trailing echo axis."""
    ref_img = nb.load(paths[0])
    ref_arr = np.asarray(ref_img.dataobj, dtype=np.float32)
    arrays = [ref_arr]
    for path in paths[1:]:
        arr = np.asarray(nb.load(path).dataobj, dtype=np.float32)
        if arr.shape != ref_arr.shape:
            raise ValueError(f'Echo {path} has shape {arr.shape}; expected {ref_arr.shape}.')
        arrays.append(arr)
    return np.stack(arrays, axis=-1), ref_img


def fit_r2star_complex_nlls(
    mag_files: list[str],
    phase_rad_files: list[str],
    echo_times: list[float],
    mask_file: str,
    n_threads: int = MEGRE_FIT_N_THREADS,
) -> dict[str, nb.Nifti1Image]:
    """Estimate R2*/T2*/S0 and goodness-of-fit maps with the complex-NLLS fit.

    Runs :func:`utils.fit_complex_r2star` (complex S0 + off-resonance frequency
    nonlinear least squares) on denoised magnitude/phase echoes and returns, on
    the MEGRE grid, a dict of NIfTI images keyed by BIDS suffix:

    - ``R2starmap`` -- R2* in s^-1 (Hz)
    - ``T2starmap`` -- T2* in seconds
    - ``S0map`` -- |S0| (arbitrary units)
    - ``Rsquaredmap`` -- magnitude goodness of fit (R^2) of ``|S0|*exp(-R2*.TE)``
      against the denoised magnitude echoes, computed over in-mask voxels.
    """
    magnitude, ref_img = _load_echo_stack(mag_files)
    phase, _ = _load_echo_stack(phase_rad_files)
    mask = np.asarray(nb.load(mask_file).dataobj).astype(bool)
    if mask.shape != magnitude.shape[:-1]:
        raise ValueError(
            f'Mask shape {mask.shape} does not match MEGRE shape {magnitude.shape[:-1]}.'
        )

    echo_times = np.asarray(echo_times, dtype=float)
    maps = fit_complex_r2star(magnitude, phase, echo_times, mask=mask, n_threads=n_threads)

    # Magnitude goodness of fit: |S(TE)| = |S0| * exp(-R2* * TE). Compute R^2 from
    # the fitted |S0|/T2* against the magnitude echoes on in-mask voxels; leave 0
    # elsewhere. Unfit voxels carry NaN T2*/S0, which map to 0 here.
    rsquared = np.zeros(mask.shape, dtype=np.float32)
    if mask.any():
        rsq = calculate_r_squared(
            magnitude[mask], echo_times, maps['s0'][mask], maps['t2star'][mask]
        )
        rsquared[mask] = np.nan_to_num(rsq, nan=0.0, posinf=0.0, neginf=0.0)

    header = ref_img.header.copy()
    header.set_data_dtype(np.float32)

    def _to_img(arr: np.ndarray) -> nb.Nifti1Image:
        return nb.Nifti1Image(np.asarray(arr, dtype=np.float32), ref_img.affine, header)

    return {
        'R2starmap': _to_img(maps['r2star']),
        'T2starmap': _to_img(maps['t2star']),
        'S0map': _to_img(maps['s0']),
        'Rsquaredmap': _to_img(rsquared),
    }


def collect_run_data(layout: object, bids_filters: dict) -> dict[str, str]:
    """Collect required input files for QSM preparation processing.

    Parameters
    ----------
    layout : bids.BIDSLayout
        BIDSLayout indexing the dataset and derivatives.
    bids_filters : dict
        BIDS entity filters (e.g., subject, session, run) to narrow the query.

    Returns
    -------
    run_data : dict
        Mapping of descriptive keys to resolved file paths.
    """
    queries = {
        # SWI images from raw BIDS dataset
        'megre_mag': {
            'datatype': 'anat',
            'acquisition': 'QSM',
            'reconstruction': [Query.NONE, Query.ANY],
            'part': 'mag',
            'echo': Query.ANY,
            'space': Query.NONE,
            'desc': Query.NONE,
            'suffix': 'MEGRE',
            'extension': ['.nii', '.nii.gz'],
        },
        'megre_phase': {
            'datatype': 'anat',
            'acquisition': 'QSM',
            'reconstruction': [Query.NONE, Query.ANY],
            'part': 'phase',
            'echo': Query.ANY,
            'space': Query.NONE,
            'desc': Query.NONE,
            'suffix': 'MEGRE',
            'extension': ['.nii', '.nii.gz'],
        },
        # T1w-space R2 map from MESE pipeline
        'r2_map': {
            'datatype': 'anat',
            'reconstruction': [Query.NONE, Query.ANY],
            'space': 'T1w',
            'desc': 'MESE',
            'suffix': 'R2map',
            'extension': '.nii.gz',
        },
        # T1w-space T1w image from sMRIPrep
        't1w': {
            'datatype': 'anat',
            'session': [Query.NONE, Query.ANY],
            'run': [Query.NONE, Query.ANY],
            'reconstruction': [Query.NONE, Query.ANY],
            'space': Query.NONE,
            'res': Query.NONE,
            'desc': 'preproc',
            'suffix': 'T1w',
            'extension': ['.nii', '.nii.gz'],
        },
        # sMRIPrep T1w-space brain mask
        't1w_mask': {
            'datatype': 'anat',
            'session': [Query.NONE, Query.ANY],
            'run': [Query.NONE, Query.ANY],
            'reconstruction': [Query.NONE, Query.ANY],
            'space': Query.NONE,
            'res': Query.NONE,
            'desc': 'brain',
            'suffix': 'mask',
            'extension': ['.nii', '.nii.gz'],
        },
        # MNI-space T1w image from sMRIPrep
        't1w_mni': {
            'datatype': 'anat',
            'session': [Query.NONE, Query.ANY],
            'run': [Query.NONE, Query.ANY],
            'reconstruction': [Query.NONE, Query.ANY],
            'space': 'MNI152NLin2009cAsym',
            'desc': 'preproc',
            'suffix': 'T1w',
            'extension': ['.nii', '.nii.gz'],
        },
        # Normalization transform from sMRIPrep
        't1w2mni_xfm': {
            'datatype': 'anat',
            'session': [Query.NONE, Query.ANY],
            'run': [Query.NONE, Query.ANY],
            'reconstruction': [Query.NONE, Query.ANY],
            'from': 'T1w',
            'to': 'MNI152NLin2009cAsym',
            'mode': 'image',
            'suffix': 'xfm',
            'extension': '.h5',
        },
        'mni2t1w_xfm': {
            'datatype': 'anat',
            'session': [Query.NONE, Query.ANY],
            'run': [Query.NONE, Query.ANY],
            'reconstruction': [Query.NONE, Query.ANY],
            'from': 'MNI152NLin2009cAsym',
            'to': 'T1w',
            'mode': 'image',
            'suffix': 'xfm',
            'extension': '.h5',
        },
        # MNI-space dseg from sMRIPrep
        'dseg_mni': {
            'datatype': 'anat',
            'session': [Query.NONE, Query.ANY],
            'run': [Query.NONE, Query.ANY],
            'reconstruction': [Query.NONE, Query.ANY],
            'space': 'MNI152NLin2009cAsym',
            'suffix': 'dseg',
            'extension': ['.nii', '.nii.gz'],
        },
        # sMRIPrep MNI-space brain mask
        'mni_mask': {
            'datatype': 'anat',
            'session': [Query.NONE, Query.ANY],
            'run': [Query.NONE, Query.ANY],
            'reconstruction': [Query.NONE, Query.ANY],
            'space': 'MNI152NLin2009cAsym',
            'desc': 'brain',
            'suffix': 'mask',
            'extension': ['.nii', '.nii.gz'],
        },
    }

    run_data = {}
    for key, query in queries.items():
        query = {**bids_filters, **query}
        files = layout.get(**query)
        if key.startswith('megre_'):
            if len(files) != 5:
                raise ValueError(f'Expected 5 files for {key}, got {len(files)}')
            else:
                run_data[key] = sorted([f.path for f in files])
                continue

        elif len(files) != 1:
            raise ValueError(f'Expected 1 file for {key}, got {len(files)} with query {query}')

        file = files[0]
        run_data[key] = file.path

    if len(run_data['megre_mag']) != len(run_data['megre_phase']):
        raise ValueError('Expected same number of magnitude and phase images')

    print(f'Collected run data:\n{pformat(run_data, indent=4)}', flush=True)
    return run_data


def process_run(layout, run_data, out_dir, temp_dir):
    """Process a single run of QSM data.

    Parameters
    ----------
    layout : BIDSLayout
        BIDSLayout object for the dataset.
    run_data : dict
        Dictionary containing the paths to the QSM data.
    out_dir : str
        Path to the output directory.
    temp_dir : str
        Path to the working directory for complex denoising and R2* fitting.
    """
    name_source = run_data['megre_mag'][0]

    megre_metadata = [layout.get_metadata(f) for f in run_data['megre_mag']]
    echo_times = [m['EchoTime'] for m in megre_metadata]  # TEs in seconds

    # Get WM segmentation from sMRIPrep
    wm_seg_img = nb.load(run_data['dseg_mni'])
    wm_seg = wm_seg_img.get_fdata()
    wm_seg = (wm_seg == 2).astype(int)
    wm_seg_file = get_filename(
        name_source=run_data['dseg_mni'],
        layout=layout,
        out_dir=out_dir,
        entities={'space': 'MNI152NLin2009cAsym', 'desc': 'wm', 'suffix': 'mask'},
        dismiss_entities=['reconstruction'],
    )
    wm_seg_img = nb.Nifti1Image(wm_seg, wm_seg_img.affine, wm_seg_img.header)
    wm_seg_img.to_filename(wm_seg_file)

    # Warp WM segmentation to T1w space
    wm_seg_img = ants.image_read(wm_seg_file)
    wm_seg_t1w_img = ants.apply_transforms(
        fixed=ants.image_read(run_data['t1w']),
        moving=wm_seg_img,
        transformlist=[run_data['mni2t1w_xfm']],
        interpolator='nearestNeighbor',
    )
    wm_seg_t1w_file = get_filename(
        name_source=wm_seg_file,
        layout=layout,
        out_dir=out_dir,
        entities={'space': 'T1w', 'desc': 'wm', 'suffix': 'mask'},
        dismiss_entities=['reconstruction'],
    )
    ants.image_write(wm_seg_t1w_img, wm_seg_t1w_file)
    del wm_seg_img, wm_seg_t1w_img, wm_seg

    # Complex MP-PCA denoise all echoes up front so the denoised magnitude feeds
    # both the reference image and the R2* fit below.
    den_mag_files, den_phase_files = denoise_complex_megre(
        run_data['megre_mag'],
        run_data['megre_phase'],
        os.path.join(temp_dir, 'denoise'),
        n_threads=16,
    )

    # The MEGRE reference is the root mean square (RMS) across the denoised
    # magnitude echoes, used for coregistration and the QC figures.
    megre_ref_filename = get_filename(
        name_source=name_source,
        layout=layout,
        out_dir=out_dir,
        entities={'space': 'MEGRE', 'desc': 'rms', 'suffix': 'MEGRE'},
        dismiss_entities=['echo'],
    )
    grid_img = nb.load(den_mag_files[0])
    rms_data = np.sqrt(
        np.mean(
            np.stack(
                [np.asanyarray(nb.load(f).dataobj, dtype=np.float64) ** 2 for f in den_mag_files],
                axis=-1,
            ),
            axis=-1,
        )
    )
    ref_header = grid_img.header.copy()
    ref_header.set_data_dtype(np.float32)
    nb.Nifti1Image(rms_data.astype(np.float32), grid_img.affine, ref_header).to_filename(
        megre_ref_filename
    )

    # Coregister MEGRE data to preprocessed T1w
    coreg_transform = coregister_to_t1(
        name_source=name_source,
        layout=layout,
        in_file=megre_ref_filename,
        t1_file=run_data['t1w'],
        source_space='MEGRE',
        target_space='T1w',
        out_dir=out_dir,
    )
    # coreg_transform = run_data['megre2t1w_xfm']
    t1_megre_ref_img = ants.apply_transforms(
        fixed=ants.image_read(run_data['t1w']),
        moving=ants.image_read(megre_ref_filename),
        transformlist=[coreg_transform],
        interpolator='linear',
    )
    t1_megre_ref_filename = get_filename(
        name_source=megre_ref_filename,
        layout=layout,
        out_dir=out_dir,
        entities={'space': 'T1w', 'desc': 'rms', 'suffix': 'MEGRE'},
        dismiss_entities=['echo', 'part', 'reconstruction'],
    )
    ants.image_write(t1_megre_ref_img, t1_megre_ref_filename)
    plot_coregistration(
        name_source=t1_megre_ref_filename,
        layout=layout,
        in_file=t1_megre_ref_filename,
        t1_file=run_data['t1w_mni'],
        out_dir=out_dir,
        source_space='MEGRE',
        target_space='MNI152NLin2009cAsym',
        wm_seg=wm_seg_t1w_file,
    )

    mni_megre_ref_img = ants.apply_transforms(
        fixed=ants.image_read(run_data['t1w_mni']),
        moving=ants.image_read(megre_ref_filename),
        transformlist=[run_data['t1w2mni_xfm'], coreg_transform],
        interpolator='linear',
    )
    mni_megre_ref_filename = get_filename(
        name_source=t1_megre_ref_filename,
        layout=layout,
        out_dir=out_dir,
        entities={'space': 'MNI152NLin2009cAsym', 'desc': 'rms', 'suffix': 'MEGRE'},
        dismiss_entities=['echo', 'part', 'reconstruction'],
    )
    ants.image_write(mni_megre_ref_img, mni_megre_ref_filename)
    plot_coregistration(
        name_source=mni_megre_ref_filename,
        layout=layout,
        in_file=mni_megre_ref_filename,
        t1_file=run_data['t1w_mni'],
        out_dir=out_dir,
        source_space='MEGRE',
        target_space='MNI152NLin2009cAsym',
        wm_seg=wm_seg_file,
    )

    # Warp R2 map from T1w space to MEGRE space
    r2_qsm_filename = get_filename(
        name_source=run_data['r2_map'],
        layout=layout,
        out_dir=out_dir,
        entities={'space': 'MEGRE'},
        dismiss_entities=['echo', 'part'],
    )
    r2_qsm_img = ants.apply_transforms(
        fixed=ants.image_read(megre_ref_filename),
        moving=ants.image_read(run_data['r2_map']),
        transformlist=[coreg_transform],
        whichtoinvert=[True],
        interpolator='linear',
    )
    ants.image_write(r2_qsm_img, r2_qsm_filename)

    # Warp brain mask from T1w space to MEGRE space. This is needed before the
    # R2* fit because the complex-NLLS fit only fits voxels in the mask.
    mask_qsm_filename = get_filename(
        name_source=name_source,
        layout=layout,
        out_dir=out_dir,
        entities={'space': 'MEGRE', 'desc': 'brain', 'suffix': 'mask'},
    )
    mask_qsm_img = ants.apply_transforms(
        fixed=ants.image_read(megre_ref_filename),
        moving=ants.image_read(run_data['t1w_mask']),
        transformlist=[coreg_transform],
        whichtoinvert=[True],
        interpolator='nearestNeighbor',
    )
    ants.image_write(mask_qsm_img, mask_qsm_filename)

    # Reportlet comparing raw vs denoised magnitude echoes (MEGRE grid).
    denoise_report = get_filename(
        name_source=name_source,
        layout=layout,
        out_dir=out_dir,
        entities={'datatype': 'figures', 'desc': 'denoise', 'extension': '.svg'},
        dismiss_entities=['echo', 'part'],
    )
    raw_mag_concat = os.path.join(temp_dir, 'megre_mag_concat.nii.gz')
    den_mag_concat = os.path.join(temp_dir, 'megre_mag_concat_desc-denoised.nii.gz')
    image.concat_imgs(run_data['megre_mag']).to_filename(raw_mag_concat)
    image.concat_imgs(den_mag_files).to_filename(den_mag_concat)
    # Noise map written by denoise_complex_megre (see fixed name therein).
    noise_file = os.path.join(temp_dir, 'denoise', 'megre_desc-noise.nii.gz')
    plot_denoise(
        raw_file=raw_mag_concat,
        denoised_file=den_mag_concat,
        out_file=denoise_report,
        mask=mask_qsm_filename,
        noise_file=noise_file if os.path.isfile(noise_file) else None,
    )

    # Residual (|raw - denoised|): should look like noise, not anatomy.
    residual_report = get_filename(
        name_source=name_source,
        layout=layout,
        out_dir=out_dir,
        entities={'datatype': 'figures', 'desc': 'residual', 'extension': '.svg'},
        dismiss_entities=['echo', 'part'],
    )
    plot_residual(
        raw_file=raw_mag_concat,
        denoised_file=den_mag_concat,
        out_file=residual_report,
        mask=mask_qsm_filename,
    )

    # Calculate R2* and R2' maps with the complex-NLLS fit using the denoised
    # echoes from above (the E2345 set drops the first echo).
    # R2starmap is in s^-1 (Hz), matching the R2 map units so R2' = R2* - R2.
    # echo_set: (label, denoised magnitude files, denoised phase files, echo times)
    echo_sets = [
        ('MEGRE+E12345', den_mag_files, den_phase_files, echo_times),
        ('MEGRE+E2345', den_mag_files[1:], den_phase_files[1:], echo_times[1:]),
    ]
    # Brain mask and R2 map (MESE, warped into MEGRE space) on the R2* fit grid.
    brain_mask_arr = np.asarray(nb.load(mask_qsm_filename).dataobj).astype(bool)
    r2_qsm_data = np.asarray(nb.load(r2_qsm_filename).dataobj, dtype=np.float32)
    for desc, set_mag_files, set_phase_files, set_echo_times in echo_sets:
        fit_maps = fit_r2star_complex_nlls(
            mag_files=set_mag_files,
            phase_rad_files=set_phase_files,
            echo_times=set_echo_times,
            mask_file=mask_qsm_filename,
        )
        affine = fit_maps['R2starmap'].affine
        header = fit_maps['R2starmap'].header

        # Valid R2* voxels are inside the brain mask and successfully fit. The
        # complex-NLLS fit leaves NaN at unfit/out-of-mask voxels and is bounded
        # at R2* >= 0, so finite positive values mark fitted voxels.
        r2s_raw = np.asarray(fit_maps['R2starmap'].dataobj, dtype=np.float32)
        valid = brain_mask_arr & np.isfinite(r2s_raw) & (r2s_raw > 0)
        r2s_data = np.where(valid, r2s_raw, 0.0).astype(np.float32)

        # Write the complex-NLLS maps (R2*, T2*, S0, R^2) on the MEGRE grid,
        # zeroing invalid/out-of-mask voxels so the maps are 0 outside the brain.
        map_files = {}
        for suffix in ('R2starmap', 'T2starmap', 'S0map', 'Rsquaredmap'):
            arr = np.asarray(fit_maps[suffix].dataobj, dtype=np.float32)
            # Guard finiteness: T2* = 1/R2* can overflow float32 to inf where
            # R2* is ~0, which would break the percentile range and the plot.
            map_data = np.where(valid & np.isfinite(arr), arr, 0.0).astype(np.float32)
            map_filename = get_filename(
                name_source=r2_qsm_filename,
                layout=layout,
                out_dir=out_dir,
                entities={'space': 'MEGRE', 'desc': desc, 'suffix': suffix},
            )
            nb.Nifti1Image(map_data, affine, header).to_filename(map_filename)
            map_files[suffix] = map_filename

        # R2' = R2* - R2, computed only where R2* is valid; elsewhere 0. This
        # avoids the non-physical negative shell the previous code produced by
        # subtracting a nonzero (whole-FOV) R2 from a zeroed, undefined R2*.
        r2prime_data = np.where(valid, r2s_data - r2_qsm_data, 0.0).astype(np.float32)
        r2prime_hz_filename = get_filename(
            name_source=r2_qsm_filename,
            layout=layout,
            out_dir=out_dir,
            entities={'space': 'MEGRE', 'desc': desc, 'suffix': 'R2primemap'},
        )
        nb.Nifti1Image(r2prime_data, affine, header).to_filename(r2prime_hz_filename)
        map_files['R2primemap'] = r2prime_hz_filename

        # Scalar reportlets. Warp each MEGRE-space map to MNI152NLin2009cAsym
        # (matching the other scalar reportlets) and plot it over the sMRIPrep
        # T1w with the tissue segmentation. R2*/T2*/S0 are non-negative
        # (sequential 'Reds', 2nd-98th percentile range); R2' is signed
        # (diverging 'RdBu_r' with a symmetric colorbar); R^2 is a [0, 1]
        # goodness-of-fit. The '+' in the echo-set desc is stripped from the
        # figure desc because nireports' figures config only captures
        # [a-zA-Z0-9]; a '+' would truncate the desc on indexing and the
        # report's '.*scalar' query would miss the figure.
        scalar_specs = [
            ('R2starmap', 'Reds', 'percentile'),
            ('R2primemap', 'RdBu_r', 'symmetric'),
            ('S0map', 'Reds', 'percentile'),
            ('T2starmap', 'Reds', 'percentile'),
            ('Rsquaredmap', 'Reds', 'unit'),
        ]
        for suffix, cmap, scale in scalar_specs:
            map_file = map_files[suffix]
            mni_map_img = ants.apply_transforms(
                fixed=ants.image_read(run_data['t1w_mni']),
                moving=ants.image_read(map_file),
                transformlist=[run_data['t1w2mni_xfm'], coreg_transform],
                interpolator='linear',
            )
            mni_map_file = os.path.join(
                temp_dir,
                f'{suffix}_{desc.replace("+", "")}_space-MNI152NLin2009cAsym.nii.gz',
            )
            ants.image_write(mni_map_img, mni_map_file)

            scalar_report = get_filename(
                name_source=map_file,
                layout=layout,
                out_dir=out_dir,
                entities={
                    'datatype': 'figures',
                    'space': 'MNI152NLin2009cAsym',
                    'desc': desc.replace('+', '') + 'scalar',
                    'suffix': suffix,
                    'extension': '.svg',
                },
                dismiss_entities=['echo', 'part'],
            )
            if scale == 'unit':
                vmin, vmax, symmetric = 0.0, 1.0, False
            else:
                data = masking.apply_mask(mni_map_file, run_data['mni_mask'])
                vmin = float(np.minimum(np.percentile(data, 2), 0))
                vmax = float(np.percentile(data, 98))
                symmetric = scale == 'symmetric'
            plot_scalar_map(
                underlay=run_data['t1w_mni'],
                overlay=mni_map_file,
                mask=run_data['mni_mask'],
                dseg=run_data['dseg_mni'],
                out_file=scalar_report,
                vmin=vmin,
                vmax=vmax,
                cmap=cmap,
                symmetric=symmetric,
            )


def _get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--subject-id',
        type=lambda label: label.removeprefix('sub-'),
        default=None,
        help='Subject to process. If not provided, all subjects are processed.',
    )
    return parser


def _main(argv=None):
    """Run the process_qsm_prep workflow."""
    options = _get_parser().parse_args(argv)
    kwargs = vars(options)
    main(**kwargs)


def main(subject_id):
    code_dir = CFG['code_dir']
    in_dir = CFG['bids_dir']
    smriprep_dir = CFG['derivatives']['smriprep']
    mese_dir = CFG['derivatives']['mese']
    out_dir = CFG['derivatives']['megre']
    os.makedirs(out_dir, exist_ok=True)
    temp_dir = os.path.join(CFG['work_dir'], 'megre')
    os.makedirs(temp_dir, exist_ok=True)

    bootstrap_file = os.path.join(CODE_DIR, 'configuration', 'reports_spec_megre.yml')
    assert os.path.isfile(bootstrap_file), f'Bootstrap file {bootstrap_file} not found'

    # Write the QSM derivatives dataset_description.json before building the
    # layout. Without it, pybids silently refuses to index this directory as a
    # derivative, so process_qsm.py cannot find the prep outputs (brain mask,
    # R2*/R2' maps) written here.
    dataset_description_file = os.path.join(out_dir, 'dataset_description.json')
    if not os.path.isfile(dataset_description_file):
        dataset_description = {
            'Name': 'NIBS MEGRE Derivatives',
            'BIDSVersion': '1.10.0',
            'DatasetType': 'derivative',
            'DatasetLinks': {
                'raw': in_dir,
                'smriprep': smriprep_dir,
                'mese': mese_dir,
            },
            'GeneratedBy': [
                {
                    'Name': 'Custom code',
                    'Description': 'Custom Python code combining ANTsPy, MRtrix3, and a '
                    'complex-NLLS R2* fit.',
                    'CodeURL': 'https://github.com/PennLINC/nibs',
                }
            ],
        }
        with open(dataset_description_file, 'w') as fobj:
            json.dump(dataset_description, fobj, sort_keys=True, indent=4)

    layout = BIDSLayout(
        in_dir,
        config=os.path.join(code_dir, 'configuration', 'nibs_bids_config.json'),
        validate=False,
        derivatives=[smriprep_dir, mese_dir],
    )

    if subject_id:
        subjects = [subject_id]
    else:
        subjects = layout.get_subjects(suffix='MEGRE')

    for subject_id in subjects:
        print(f'Processing subject {subject_id}')
        sessions = layout.get_sessions(subject=subject_id, suffix='MEGRE')
        for session in sessions:
            print(f'Processing session {session}')
            megre_files = layout.get(
                subject=subject_id,
                session=session,
                acquisition='QSM',
                echo=1,
                part='mag',
                suffix='MEGRE',
                extension=['.nii', '.nii.gz'],
            )
            for megre_file in megre_files:
                entities = megre_file.get_entities()
                entities.pop('echo')
                entities.pop('part')
                entities.pop('acquisition')
                try:
                    run_data = collect_run_data(layout, entities)
                except ValueError as e:
                    print(f'Failed {megre_file}')
                    print(e)
                    continue

                fname = os.path.basename(megre_file.path).split('.')[0]
                run_temp_dir = os.path.join(temp_dir, fname.replace('-', '').replace('_', ''))
                os.makedirs(run_temp_dir, exist_ok=True)
                process_run(layout, run_data, out_dir, run_temp_dir)

            report_dir = os.path.join(out_dir, f'sub-{subject_id}', f'ses-{session}')
            robj = Report(
                report_dir,
                run_uuid=None,
                bootstrap_file=bootstrap_file,
                out_filename=f'sub-{subject_id}_ses-{session}.html',
                reportlets_dir=out_dir,
                plugins=None,
                plugin_meta=None,
                subject=subject_id,
                session=session,
            )
            robj.generate_report()

    print('DONE!')


if __name__ == '__main__':
    _main()
