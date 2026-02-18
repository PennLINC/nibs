"""Shared utilities for processing pipelines.

Provides ``load_config``, ``run_command``, ``get_filename``,
``coregister_to_t1``, ``fit_monoexponential``, ``plot_scalar_map``,
and ``calculate_r_squared``.
"""

from __future__ import annotations

import os
import sys
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from config import load_config  # noqa: E402, F401


def run_command(command: str | list[str], env: dict[str, str] | None = None) -> None:
    """Run a shell command, streaming stdout line-by-line.

    Parameters
    ----------
    command : str or list of str
        Command to execute. Strings are split with :func:`shlex.split`.
    env : dict, optional
        Extra environment variables merged into the current environment.

    Raises
    ------
    RuntimeError
        If the process exits with a non-zero return code.
    """
    import shlex
    import subprocess

    if isinstance(command, str):
        command = shlex.split(command)

    merged_env = os.environ.copy()
    if env:
        merged_env.update(env)

    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        shell=False,
        env=merged_env,
    )
    output_lines = []
    while True:
        line = process.stdout.readline()
        line = str(line, 'utf-8')[:-1]
        print(line)
        output_lines.append(line)
        if line == '' and process.poll() is not None:
            break

    if process.returncode != 0:
        raise RuntimeError(
            f'Non zero return code: {process.returncode}\n'
            f'{" ".join(command)}\n\n{"".join(output_lines)}'
        )


def get_filename(
    name_source: str,
    layout: object,
    out_dir: str,
    entities: dict[str, object],
    dismiss_entities: list[str] | None = None,
) -> str:
    """Build an output file path from a BIDS name source and new entities.

    Parameters
    ----------
    name_source : str
        Path to an existing BIDS file whose entities serve as defaults.
    layout : bids.BIDSLayout
        BIDSLayout used to construct the output path.
    out_dir : str
        Root directory for the output file (replaces the layout root).
    entities : dict
        Entity key-value pairs to override or add to the source entities.
    dismiss_entities : list of str, optional
        Entity keys to drop from the source before merging.

    Returns
    -------
    out_file : str
        Absolute path to the (newly created directory for the) output file.
    """
    from bids.layout import parse_file_entities

    if dismiss_entities is None:
        dismiss_entities = []

    source_entities = parse_file_entities(name_source)
    # source_entities = layout.get_file(name_source).get_entities()
    source_entities = {k: v for k, v in source_entities.items() if k not in dismiss_entities}
    entities = {**source_entities, **entities}
    out_file = layout.build_path(entities, validate=False, strict=True)
    out_file = out_file.replace(os.path.abspath(layout.root), os.path.abspath(out_dir))
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    return out_file


def coregister_to_t1(
    name_source: str,
    layout: object,
    in_file: str,
    t1_file: str,
    out_dir: str,
    source_space: str,
    target_space: str,
) -> str:
    """Coregister an image to a T1w image.

    Parameters
    ----------
    name_source : str
        Name of the source file to use for output file names.
    layout : BIDSLayout
        BIDSLayout object.
    in_file : str
        Path to the input image.
    t1_file : str
        Path to the T1w image.
    out_dir : str
        Directory to write output files.
    source_space : str
        Source space of the input image.
    target_space : str
        Target space of the T1w image.

    Returns
    -------
    transform_file : str
        Path to the transform file.
    """
    import shutil

    import ants
    import antspynet

    # Step 1: Apply N4 bias field correction and skull-stripping to T1.
    t1_img = ants.image_read(t1_file)
    n4_img = ants.n4_bias_field_correction(t1_img)
    dseg_img = antspynet.utilities.brain_extraction(n4_img, modality='t1threetissue')
    dseg_img = dseg_img['segmentation_image']

    dseg_file = get_filename(
        name_source=name_source,
        layout=layout,
        out_dir=out_dir,
        entities={'space': target_space, 'desc': 't1threetissue', 'suffix': 'dseg'},
        dismiss_entities=['echo', 'inv', 'reconstruction'],
    )
    ants.image_write(dseg_img, dseg_file)

    # Binarize the brain mask
    mask_img = ants.threshold_image(
        dseg_img,
        low_thresh=1,
        high_thresh=1,
        inval=1,
        outval=0,
        binary=True,
    )
    mask_file = get_filename(
        name_source=name_source,
        layout=layout,
        out_dir=out_dir,
        entities={'space': target_space, 'desc': 'brain', 'suffix': 'mask'},
        dismiss_entities=['echo', 'inv', 'reconstruction'],
    )
    ants.image_write(mask_img, mask_file)

    n4_img_masked = n4_img * mask_img

    # Step 2: Coregister the brain-extracted image to the T1w image.
    registered_img = ants.registration(
        fixed=n4_img_masked,
        moving=ants.image_read(in_file),
        type_of_transform='Rigid',
    )
    transform = registered_img['fwdtransforms'][0]
    transform_file = get_filename(
        name_source=name_source,
        layout=layout,
        out_dir=out_dir,
        entities={
            'from': source_space,
            'to': target_space,
            'mode': 'image',
            'suffix': 'xfm',
            'extension': 'txt' if transform.endswith('.txt') else 'mat',
        },
        dismiss_entities=['acquisition', 'inv', 'reconstruction', 'mt', 'echo', 'part'],
    )
    shutil.copyfile(transform, transform_file)

    return transform_file


def plot_coregistration(
    name_source: str,
    layout: object,
    in_file: str,
    t1_file: str,
    out_dir: str,
    source_space: str,
    target_space: str,
    wm_seg: str | None = None,
) -> None:
    """Generate an SVG report comparing an image before and after coregistration to T1w.

    Parameters
    ----------
    name_source : str
        Path to the BIDS file used to derive the output report filename.
    layout : bids.BIDSLayout
        BIDSLayout for path construction.
    in_file : str
        Path to the coregistered image (the "before" view).
    t1_file : str
        Path to the T1w reference image (the "after" view).
    out_dir : str
        Root directory for the output report.
    source_space : str
        Label for the source image space.
    target_space : str
        Label for the T1w target space.
    wm_seg : str, optional
        Path to a white-matter segmentation for edge overlay.
    """
    from nireports.interfaces.reporting.base import SimpleBeforeAfterRPT

    desc = 'coreg'
    if 'desc-' in name_source:
        # Append the desc to the target desc
        desc = name_source.split('desc-')[-1].split('_')[0] + 'coreg'

    out_report = get_filename(
        name_source=name_source,
        layout=layout,
        out_dir=out_dir,
        entities={'datatype': 'figures', 'space': target_space, 'desc': desc, 'extension': '.svg'},
    )
    if wm_seg is not None:
        kwargs = {'wm_seg': wm_seg}
    else:
        kwargs = {}

    coreg_report = SimpleBeforeAfterRPT(
        before_label=source_space,
        after_label=target_space,
        dismiss_affine=True,
        before=in_file,
        after=t1_file,
        out_report=out_report,
        **kwargs,
    )
    coreg_report.run()


def fit_monoexponential(in_files: list[str], echo_times: list[float]) -> tuple:
    """Fit monoexponential decay model to MESE data.

    Parameters
    ----------
    in_files : list of str
        List of paths to MESE data.
    echo_times : list of float
        List of echo times in milliseconds.

    Returns
    -------
    t2s_s_img : nibabel.Nifti1Image
        T2* map in seconds.
    r2s_hz_img : nibabel.Nifti1Image
        R2* map in Hertz.
    s0_img : nibabel.Nifti1Image
        S0 map in arbitrary units.
    """
    import numpy as np
    from tedana import io, decay

    data_cat, ref_img = io.load_data(in_files, n_echos=len(echo_times))

    # Fit model on all voxels, using all echoes
    mask = np.ones(data_cat.shape[0], dtype=int)
    masksum = mask * len(echo_times)

    echo_times_ms = [te * 1000 for te in echo_times]
    t2s_limited, s0_limited, _, _ = decay.fit_monoexponential(
        data_cat=data_cat,
        echo_times=echo_times_ms,
        adaptive_mask=masksum,
        report=False,
    )
    # Limit positive infinite values to maximum finite value
    t2s_limited[np.isinf(t2s_limited) & (t2s_limited > 0)] = np.nanmax(
        t2s_limited[np.isfinite(t2s_limited)]
    )
    s0_limited[np.isinf(s0_limited) & (s0_limited > 0)] = np.nanmax(
        s0_limited[np.isfinite(s0_limited)]
    )
    # Set negative infinite values to minimum finite value
    t2s_limited[np.isinf(t2s_limited) & (t2s_limited < 0)] = np.nanmin(
        t2s_limited[np.isfinite(t2s_limited)]
    )
    s0_limited[np.isinf(s0_limited) & (s0_limited < 0)] = np.nanmin(
        s0_limited[np.isfinite(s0_limited)]
    )
    # Set nan values to 0
    t2s_limited[np.isnan(t2s_limited)] = 0
    s0_limited[np.isnan(s0_limited)] = 0

    r_squared = calculate_r_squared(np.squeeze(data_cat), echo_times_ms, s0_limited, t2s_limited)

    t2s_s = t2s_limited / 1000

    r2s_hz = np.zeros_like(t2s_s)
    np.divide(1, t2s_s, out=r2s_hz, where=t2s_s != 0)

    t2s_s_img = io.new_nii_like(ref_img, t2s_s)
    r2s_hz_img = io.new_nii_like(ref_img, r2s_hz)
    s0_img = io.new_nii_like(ref_img, s0_limited)
    r_squared_img = io.new_nii_like(ref_img, r_squared)
    return t2s_s_img, r2s_hz_img, s0_img, r_squared_img


def plot_scalar_map(
    underlay: str,
    overlay: str,
    mask: str,
    out_file: str,
    dseg: str | None = None,
    vmin: float | None = None,
    vmax: float | None = None,
    cmap: str = 'Reds',
) -> None:
    """Plot a scalar map overlaid on an anatomical underlay with a tissue-type histogram.

    Produces a three-panel figure: a KDE histogram of voxel intensities by
    tissue type (left), axial slice overlay (center), and a colorbar (right).

    Parameters
    ----------
    underlay : str
        Path to the anatomical underlay NIfTI image.
    overlay : str
        Path to the scalar map NIfTI image to plot.
    mask : str
        Path to a binary brain mask NIfTI image.
    out_file : str
        Path for the saved figure.
    dseg : str, optional
        Path to a discrete segmentation image (1=GM, 2=WM, 3=CSF).
        If None, the mask is used as a single "Brain" tissue class.
    vmin : float, optional
        Minimum value for the color scale.
    vmax : float, optional
        Maximum value for the color scale.
    cmap : str, optional
        Matplotlib colormap name. Default is ``'Reds'``.
    """
    import matplotlib.pyplot as plt
    import nibabel as nb
    import numpy as np
    import pandas as pd
    import seaborn as sns
    from matplotlib import cm
    from nilearn import image, maskers, masking, plotting
    from nireports.reportlets.utils import cuts_from_bbox

    if not os.path.isdir(os.path.dirname(out_file)):
        os.makedirs(os.path.dirname(out_file), exist_ok=True)

    cuts = cuts_from_bbox(nb.load(underlay), cuts=6)
    z_cuts = cuts['z']
    masker = maskers.NiftiMasker(mask_img=mask)
    overlay_masked = masker.inverse_transform(masker.fit_transform(overlay))
    underlay_masked = masking.unmask(masking.apply_mask(underlay, mask), mask)

    if dseg is not None:
        tissue_types = ['GM', 'WM', 'CSF']
        tissue_values = [1, 2, 3]
        tissue_colors = ['#1b60a5', '#2da467', '#9d8f25']
    else:
        tissue_types = ['Brain']
        tissue_values = [1]
        tissue_colors = ['#1b60a5']
        dseg = mask

    tissue_palette = dict(zip(tissue_types, tissue_colors))

    # Histogram time
    dfs = []
    for i_tissue_type, tissue_type in enumerate(tissue_types):
        tissue_type_val = tissue_values[i_tissue_type]
        mask_img = image.math_img(
            f'(img == {tissue_type_val}).astype(np.int32)',
            img=dseg,
        )
        masker = maskers.NiftiMasker(mask_img=mask_img)
        tissue_type_vals = np.squeeze(masker.fit_transform(overlay))
        df = pd.DataFrame(
            columns=['Data', 'Tissue Type'],
            data=list(map(list, zip(*[tissue_type_vals, [tissue_type] * tissue_type_vals.size]))),
        )
        dfs.append(df)

    df = pd.concat(dfs, axis=0)
    fig, axes = plt.subplots(
        figsize=(43, 6),
        ncols=3,
        gridspec_kw=dict(width_ratios=[6, 36, 0.25], wspace=0),
    )
    ax0, ax1, ax2 = axes
    with sns.axes_style('whitegrid'), sns.plotting_context(font_scale=3):
        sns.kdeplot(
            data=df,
            x='Data',
            palette=tissue_palette,
            hue='Tissue Type',
            fill=True,
            ax=ax0,
        )

    xticks = ax0.get_xticklabels()
    xlim = list(ax0.get_xlim())
    if vmin is not None:
        xlim[0] = vmin

    if vmax is not None:
        xlim[1] = vmax

    ax0.set_xlim(xlim)

    xticks = [
        i for i in xticks if i.get_position()[0] <= xlim[1] and i.get_position()[0] >= xlim[0]
    ]
    xticklabels = [xtick.get_text() for xtick in xticks]
    xticks = [xtick.get_position()[0] for xtick in xticks]
    xmin = xlim[0]
    xmax = xlim[1]
    if xmin < 0:
        kwargs = {'symmetric_cbar': True}
    else:
        kwargs = {'symmetric_cbar': False, 'vmin': xmin}

    plotting.plot_stat_map(
        stat_map_img=overlay_masked,
        bg_img=underlay_masked,
        resampling_interpolation='nearest',
        display_mode='z',
        cut_coords=z_cuts,
        threshold=0.00001,
        draw_cross=False,
        colorbar=False,
        black_bg=False,
        vmax=xmax,
        axes=ax1,
        cmap=cmap,
        **kwargs,
    )
    mappable = cm.ScalarMappable(norm=plt.Normalize(vmin=xmin, vmax=xmax), cmap=cmap)
    cbar = plt.colorbar(cax=ax2, mappable=mappable)
    cbar.set_ticks(xticks)
    cbar.set_ticklabels(xticklabels)
    fig.savefig(out_file, bbox_inches=0)


def calculate_r_squared(
    data: np.ndarray, echo_times: list[float], s0: np.ndarray, t2s: np.ndarray
) -> np.ndarray:
    """Calculate R-squared from data and T2*/S0 estimates.

    R-squared is a measure of goodness of fit of a monoexponential model to the data.

    Parameters
    ----------
    data : numpy.ndarray of shape (n_samples, n_echos)
        Data to calculate R-squared for.
    echo_times : list of float
        Echo times in milliseconds.
    s0 : numpy.ndarray of shape (n_samples,)
        S0 values from a monoexponential fit of echo times against the data.
    t2s : numpy.ndarray of shape (n_samples,)
        T2* values from a monoexponential fit of echo times against the data.

    Returns
    -------
    r_squared : numpy.ndarray of shape (n_samples,)
        R-squared values.
    """
    import numpy as np

    n_voxels, n_echos = data.shape
    echo_times_rep = np.tile(echo_times, (n_voxels, 1))
    s_pred = s0[:, np.newaxis] * np.exp(-echo_times_rep / t2s[:, np.newaxis])  # monoexp

    # Calculate residuals (observed - predicted) for each voxel
    residuals = data - s_pred

    # Sum of squared residuals per voxel
    ss_resid = np.sum(residuals**2, axis=1)

    # Calculate mean signal per voxel
    mean_signal = np.mean(data, axis=1)

    # Total sum of squares per voxel (sum of squared deviations from mean)
    ss_total = np.sum((data - mean_signal[:, np.newaxis]) ** 2, axis=1)

    # Handle division by zero
    ss_total[ss_total == 0] = np.spacing(1)

    # R-squared = 1 - (SS_residual / SS_total)
    r_squared = 1 - (ss_resid / ss_total)

    return r_squared
