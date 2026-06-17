"""Shared utilities for processing pipelines.

Provides ``load_config``, ``run_command``, ``get_filename``,
``coregister_to_t1``, ``fit_monoexponential``, ``fit_complex_r2star``,
``plot_scalar_map``, ``plot_denoise``, ``plot_residual``, and ``calculate_r_squared``.
"""

from __future__ import annotations

import os
import sys

# Force matplotlib's non-interactive Agg backend. The default interactive
# backend (e.g. TkAgg) starts a Tk interpreter whose figure destructors run
# from joblib's loky result-handler threads during GC; a Tk __del__ outside the
# main loop blocks and deadlocks the worker pool used by fit_complex_r2star
# (and spams "main thread is not in main loop"). Setting MPLBACKEND only works
# if matplotlib has not been imported yet, so we also force=True to override a
# backend that an earlier import (nilearn, nireports, ...) may have selected.
os.environ.setdefault('MPLBACKEND', 'Agg')
import matplotlib

matplotlib.use('Agg', force=True)

import numpy as np
import scipy.optimize

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover - tqdm is an optional progress dependency

    class tqdm:  # noqa: N801 - mimic tqdm's lowercase public name
        """Minimal no-op fallback used when tqdm is unavailable."""

        def __init__(self, iterable=None, total=None, desc=None, **kwargs):
            self._iterable = iterable

        def __iter__(self):
            return iter(self._iterable or [])

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def update(self, n=1):
            pass


sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from configuration.config import load_config  # noqa: E402, F401


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
        dismiss_entities=['echo', 'inv', 'reconstruction', 'part'],
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
        dismiss_entities=['echo', 'inv', 'reconstruction', 'part'],
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
        dismiss_entities=[
            'acquisition',
            'desc',
            'inv',
            'reconstruction',
            'mt',
            'echo',
            'part',
            'space',
        ],
    )
    shutil.copyfile(transform, transform_file)

    return transform_file


def _ensure_valid_cwd() -> None:
    """Restore a valid working directory if the current one became invalid.

    On WSL/``/mnt/c`` (9p) the working-directory handle can go stale mid-run,
    making ``os.getcwd()`` raise ``FileNotFoundError`` (nipype's ``indirectory``
    and many libraries read it). Pin the cwd to a fixed ext4 path so it cannot be
    invalidated again. The entry-point scripts do this at startup; this is a
    defensive backstop.
    """
    try:
        os.getcwd()
        return
    except (FileNotFoundError, OSError):
        pass

    for path in ('/tmp/nibs-cwd', '/tmp', '/'):
        try:
            os.makedirs(path, exist_ok=True)
            os.chdir(path)
            return
        except OSError:
            continue


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
    out_dir = os.path.dirname(out_report)
    os.makedirs(out_dir, exist_ok=True)
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
    # nipype reads os.getcwd() to save/restore the working directory; if that
    # directory was removed mid-run (e.g. scratch cleanup) this raises
    # FileNotFoundError before any work happens. Restore a stable cwd first.
    _ensure_valid_cwd()
    coreg_report.run()


def fit_monoexponential(
    in_files: list[str],
    echo_times: list[float],
    mask: str | None = None,
    n_threads: int = 4,
) -> tuple:
    """Fit monoexponential decay model to MESE data.

    Parameters
    ----------
    in_files : list of str
        List of paths to MESE data.
    echo_times : list of float
        List of echo times in seconds.
    mask : str or None
        Path to a brain mask on the same grid as ``in_files``. The fit is
        restricted to in-mask voxels and out-of-mask voxels are set to 0. If
        None, every voxel is fit (legacy behavior).
    n_threads : int
        Number of threads to use.

    Returns
    -------
    t2s_s_img : nibabel.Nifti1Image
        T2 map in seconds.
    r2s_hz_img : nibabel.Nifti1Image
        R2 map in Hertz.
    s0_img : nibabel.Nifti1Image
        S0 map in arbitrary units.
    """
    import nibabel as nb
    import numpy as np
    from nilearn import masking
    from tedana import decay

    in_img = nb.load(in_files[0])
    if mask is None:
        mask_arr = np.ones(in_img.shape[:3], dtype=int)
        mask_img = nb.Nifti1Image(mask_arr, in_img.affine, in_img.header)
    else:
        mask_img = nb.load(mask)
    data_arrays = [masking.apply_mask(nb.load(f), mask_img)[:, None] for f in in_files]
    data_cat = np.stack(data_arrays, axis=1)

    # Fit model on all voxels, using all echoes
    masksum = np.full(data_cat.shape[0], len(echo_times))

    t2s, s0 = decay.fit_monoexponential(
        data_cat=data_cat,
        echo_times=echo_times,
        adaptive_mask=masksum,
        report=False,
        n_threads=n_threads,
    )[:2]
    # Limit positive infinite values to maximum finite value
    t2s[np.isinf(t2s) & (t2s > 0)] = np.nanmax(
        t2s[np.isfinite(t2s)]
    )
    s0[np.isinf(s0) & (s0 > 0)] = np.nanmax(
        s0[np.isfinite(s0)]
    )
    # Set negative infinite values to minimum finite value
    t2s[np.isinf(t2s) & (t2s < 0)] = np.nanmin(
        t2s[np.isfinite(t2s)]
    )
    s0[np.isinf(s0) & (s0 < 0)] = np.nanmin(
        s0[np.isfinite(s0)]
    )

    # Set negative values to 0
    t2s[t2s < 0] = 0
    s0[s0 < 0] = 0

    # Set nan values to 0
    t2s[np.isnan(t2s)] = 0
    s0[np.isnan(s0)] = 0

    r_squared = calculate_r_squared(np.squeeze(data_cat), echo_times, s0, t2s)

    r2s_hz = np.zeros_like(t2s)
    np.divide(1, t2s, out=r2s_hz, where=t2s != 0)

    t2s_s_img = masking.unmask(t2s, mask_img)
    t2s_s_img.header.set_data_dtype(np.float32)
    r2s_hz_img = masking.unmask(r2s_hz, mask_img)
    r2s_hz_img.header.set_data_dtype(np.float32)
    s0_img = masking.unmask(s0, mask_img)
    s0_img.header.set_data_dtype(np.float32)
    r_squared_img = masking.unmask(r_squared, mask_img)
    r_squared_img.header.set_data_dtype(np.float32)
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
    symmetric: bool = False,
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
    symmetric : bool, optional
        If True, force the color scale (and colorbar) to be symmetric about zero,
        spanning ``[-bound, bound]`` with ``bound = max(|vmin|, |vmax|)``. Use for
        signed maps (e.g. R2') with a diverging colormap. Default is False.
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

    if symmetric:
        bound = max(abs(xlim[0]), abs(xlim[1]))
        xlim = [-bound, bound]

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
    plt.close(fig)


def plot_scalar_comparison(
    x_file: str,
    y_file: str,
    mask_file: str,
    out_file: str,
    x_label: str,
    y_label: str,
    title: str | None = None,
) -> None:
    """Head-to-head voxelwise comparison of two scalar maps over a brain mask.

    Renders a log-density hexbin scatter of the in-mask voxel values of two
    maps (``x_file`` on the x-axis, ``y_file`` on the y-axis) with an identity
    line and the Pearson correlation, for comparing two estimates of the same
    quantity (e.g. R2* from two pipelines).

    Parameters
    ----------
    x_file, y_file : str
        Scalar maps to compare. ``y_file`` (and the mask) are resampled onto the
        ``x_file`` grid if their shape/affine differ.
    mask_file : str
        Brain mask restricting the voxels that enter the scatter.
    out_file : str
        Path for the saved SVG figure.
    x_label, y_label : str
        Axis labels.
    title : str, optional
        Figure title.
    """
    import matplotlib.pyplot as plt
    import nibabel as nb
    import numpy as np
    from nilearn import image

    if not os.path.isdir(os.path.dirname(out_file)):
        os.makedirs(os.path.dirname(out_file), exist_ok=True)

    x_img = nb.load(x_file)
    y_img = nb.load(y_file)
    mask_img = nb.load(mask_file)
    # Put y and the mask on the x grid if they differ (both are nominally in
    # MEGRE space, but resample defensively so the voxels line up).
    if y_img.shape != x_img.shape or not np.allclose(y_img.affine, x_img.affine):
        y_img = image.resample_to_img(y_img, x_img, interpolation='linear')
    if mask_img.shape != x_img.shape or not np.allclose(mask_img.affine, x_img.affine):
        mask_img = image.resample_to_img(mask_img, x_img, interpolation='nearest')

    x = np.asarray(x_img.dataobj, dtype=np.float64)
    y = np.asarray(y_img.dataobj, dtype=np.float64)
    m = np.asarray(mask_img.dataobj) > 0
    # Drop non-finite and the zero-fill used for unfit/out-of-mask voxels.
    sel = m & np.isfinite(x) & np.isfinite(y) & (x != 0) & (y != 0)
    xv = x[sel]
    yv = y[sel]

    fig, ax = plt.subplots(figsize=(6, 6))
    if xv.size > 1:
        lo = float(min(xv.min(), yv.min()))
        hi = float(max(xv.max(), yv.max()))
        hb = ax.hexbin(xv, yv, gridsize=80, bins='log', cmap='viridis', mincnt=1)
        fig.colorbar(hb, ax=ax, label='log10(voxel count)')
        ax.plot([lo, hi], [lo, hi], 'r--', linewidth=1, label='identity')
        r = float(np.corrcoef(xv, yv)[0, 1])
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_aspect('equal')
        ax.text(
            0.05,
            0.95,
            f'r = {r:.3f}\nn = {xv.size}',
            transform=ax.transAxes,
            va='top',
            ha='left',
            bbox={'boxstyle': 'round', 'facecolor': 'white', 'alpha': 0.7},
        )
        ax.legend(loc='lower right')
    else:
        ax.text(
            0.5,
            0.5,
            'insufficient overlapping voxels',
            ha='center',
            va='center',
            transform=ax.transAxes,
        )

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    if title:
        ax.set_title(title)
    fig.savefig(out_file, bbox_inches='tight')
    plt.close(fig)


def _plot_denoise_panel(
    short_te_img,
    long_te_img,
    div_id: str,
    cuts: dict,
    label: str,
    contour_img=None,
    contour_levels: list | None = None,
    contour_colors: list | None = None,
    cmap: str | None = None,
    order: tuple = ('z', 'x', 'y'),
) -> list:
    """Render one denoising panel (a short-TE and a long-TE echo) as SVG figures.

    Ported from qsiprep's ``plot_denoise``. Each echo is plotted along three
    orientations, optionally with the noise map drawn as a few contour lines on
    top, and returned as a list of :class:`SVGFigure` objects for
    :func:`compose_view`.
    """
    from uuid import uuid4

    import nireports._vendored.svgutils.transform as svgt
    from lxml import etree
    from nilearn import image, plotting
    from nireports.reportlets.utils import SVGNS, extract_svg, robust_set_limits

    out_files = []
    panels = (
        (short_te_img, f'{label}: short TE'),
        (long_te_img, f'{label}: long TE'),
    )
    for echo_img, title in panels:
        echo_data = echo_img.get_fdata(dtype='float32')
        plot_params = robust_set_limits(echo_data.reshape(-1), {})
        if cmap is not None:
            plot_params['cmap'] = cmap
        cropped = echo_img if np.all(echo_data <= 1e-8) else image.crop_img(echo_img)
        for i, mode in enumerate(order):
            plot_params['display_mode'] = mode
            plot_params['cut_coords'] = cuts[mode]
            plot_params['title'] = title if i == 0 else None
            display = plotting.plot_anat(cropped, **plot_params)
            if contour_img is not None and contour_levels:
                display.add_contours(
                    contour_img,
                    levels=contour_levels,
                    colors=contour_colors,
                    linewidths=0.75,
                )

            svg = extract_svg(display, compress=False)
            display.close()

            # Make the figure id unique so multiple panels can coexist in one SVG.
            xml_data = etree.fromstring(svg)  # noqa: S320
            find_text = etree.ETXPath(f"//{{{SVGNS}}}g[@id='figure_1']")
            find_text(xml_data)[0].set('id', f'{div_id}-{mode}-{uuid4()}')

            svg_fig = svgt.SVGFigure()
            svg_fig.root = xml_data
            out_files.append(svg_fig)

    return out_files


def plot_denoise(
    raw_file: str,
    denoised_file: str,
    out_file: str,
    mask: str | None = None,
    noise_file: str | None = None,
    n_cuts: int = 7,
) -> None:
    """Generate a before/after dwidenoise reportlet as a flickering SVG.

    Mirrors qsiprep's dwidenoise reportlet so the effect of MP-PCA denoising can
    be evaluated visually. A bright (short-TE) and a dim (long-TE) echo are shown
    before and after denoising as a flickering overlay. The estimated noise map
    is drawn as a few clean contour lines (rather than per-voxel residual
    speckle): if those contours follow anatomical boundaries the denoising window
    is too large, and the raw/denoised flicker shows whether denoising changes
    the images at all.

    Parameters
    ----------
    raw_file : str
        Path to the raw 4D multi-echo image (echoes along the 4th dimension).
    denoised_file : str
        Path to the denoised 4D multi-echo image, matching ``raw_file``.
    out_file : str
        Path for the saved SVG reportlet.
    mask : str, optional
        Brain mask (on the same grid as the inputs) used to place the slices and
        to restrict the noise-contour levels. If omitted, a threshold of the
        denoised short-TE echo is used for slice placement.
    noise_file : str, optional
        dwidenoise noise map (on the same grid as the inputs), contoured over the
        flicker. If omitted, no contours are drawn.
    n_cuts : int, optional
        Number of slices per orientation. Default is 7.
    """
    import nibabel as nb
    from nilearn.image import load_img, threshold_img
    from nireports.reportlets.utils import compose_view, cuts_from_bbox

    _ensure_valid_cwd()

    if not os.path.isdir(os.path.dirname(out_file)):
        os.makedirs(os.path.dirname(out_file), exist_ok=True)

    raw_img = nb.load(raw_file)
    denoised_img = nb.load(denoised_file)

    # Use the brightest echo as "short TE" and the dimmest as "long TE".
    raw_data = raw_img.get_fdata(dtype='float32')
    echo_means = [raw_data[..., i].mean() for i in range(raw_data.shape[-1])]
    short_te_index = int(np.argmax(echo_means))
    long_te_index = int(np.argmin(echo_means))

    raw_short = raw_img.slicer[..., short_te_index]
    raw_long = raw_img.slicer[..., long_te_index]
    denoised_short = denoised_img.slicer[..., short_te_index]
    denoised_long = denoised_img.slicer[..., long_te_index]

    # Place the slices using the brain mask when available.
    mask_img = load_img(mask) if mask is not None else None
    cuts_ref = mask_img if mask_img is not None else threshold_img(denoised_short, 50)
    cuts = cuts_from_bbox(cuts_ref, cuts=n_cuts)

    # Contour the noise map with a few iso-noise levels (clean, not speckle).
    contour_img = contour_levels = contour_colors = None
    if noise_file is not None:
        contour_img = load_img(noise_file)
        noise_data = contour_img.get_fdata()
        finite = np.isfinite(noise_data) & (noise_data > 0)
        if mask_img is not None:
            finite &= np.asarray(mask_img.dataobj).astype(bool)
        noise_vals = noise_data[finite]
        if noise_vals.size:
            contour_levels = sorted(set(np.percentile(noise_vals, [50, 75, 90]).tolist()))
            # yellow -> red, increasing with noise level
            contour_colors = ['#fee08b', '#fc8d59', '#d73027'][: len(contour_levels)]

    compose_view(
        _plot_denoise_panel(
            raw_short,
            raw_long,
            'moving-image',
            cuts=cuts,
            label='Raw',
            contour_img=contour_img,
            contour_levels=contour_levels,
            contour_colors=contour_colors,
        ),
        _plot_denoise_panel(
            denoised_short,
            denoised_long,
            'fixed-image',
            cuts=cuts,
            label='Denoised',
            contour_img=contour_img,
            contour_levels=contour_levels,
            contour_colors=contour_colors,
        ),
        out_file=out_file,
    )


def plot_residual(
    raw_file: str,
    denoised_file: str,
    out_file: str,
    mask: str | None = None,
    n_cuts: int = 7,
) -> None:
    """Render the denoising residual (|raw - denoised|) as a static reportlet.

    The signal removed by denoising should look like structureless noise. Any
    anatomical structure here -- edges, gyri, vessels following tissue
    boundaries -- means real signal was removed (over-denoising / window too
    large). A bright (short-TE) and a dim (long-TE) echo are shown along three
    orientations. This is more sensitive to signal leakage than the raw/denoised
    flicker in :func:`plot_denoise`.

    Parameters
    ----------
    raw_file : str
        Path to the raw 4D multi-echo image (echoes along the 4th dimension).
    denoised_file : str
        Path to the denoised 4D multi-echo image, matching ``raw_file``.
    out_file : str
        Path for the saved SVG reportlet.
    mask : str, optional
        Brain mask (same grid as the inputs) used to place the slices. If
        omitted, a threshold of the raw short-TE echo is used.
    n_cuts : int, optional
        Number of slices per orientation. Default is 7.
    """
    import nibabel as nb
    from nilearn.image import load_img, threshold_img
    from nireports.reportlets.utils import compose_view, cuts_from_bbox

    _ensure_valid_cwd()

    if not os.path.isdir(os.path.dirname(out_file)):
        os.makedirs(os.path.dirname(out_file), exist_ok=True)

    raw_img = nb.load(raw_file)
    denoised_img = nb.load(denoised_file)

    # Use the brightest echo as "short TE" and the dimmest as "long TE".
    raw_data = raw_img.get_fdata(dtype='float32')
    echo_means = [raw_data[..., i].mean() for i in range(raw_data.shape[-1])]
    short_te_index = int(np.argmax(echo_means))
    long_te_index = int(np.argmin(echo_means))

    raw_short = raw_img.slicer[..., short_te_index]
    raw_long = raw_img.slicer[..., long_te_index]
    denoised_short = denoised_img.slicer[..., short_te_index]
    denoised_long = denoised_img.slicer[..., long_te_index]

    short_residual = nb.Nifti1Image(
        np.abs(raw_short.get_fdata() - denoised_short.get_fdata()),
        affine=raw_short.affine,
    )
    long_residual = nb.Nifti1Image(
        np.abs(raw_long.get_fdata() - denoised_long.get_fdata()),
        affine=raw_long.affine,
    )

    # Place the slices using the brain mask when available.
    cuts_ref = load_img(mask) if mask is not None else threshold_img(raw_short, 50)
    cuts = cuts_from_bbox(cuts_ref, cuts=n_cuts)

    compose_view(
        _plot_denoise_panel(
            short_residual,
            long_residual,
            'residual-image',
            cuts=cuts,
            label='|Raw - Denoised|',
            cmap='inferno',
        ),
        [],
        out_file=out_file,
    )


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


# ---------------------------------------------------------------------------
# Complex-NLLS R2* estimation
#
# Single-pool complex decay model
#     S(TE) = S0 * exp((-R2* + 1j*2*pi*f) * TE)
# with complex S0 (absorbing the initial phase) and a per-voxel off-resonance
# frequency, fit by nonlinear least squares over the real and imaginary parts.
# ---------------------------------------------------------------------------


def _complex_decay_model(echo_times_seconds, s0, r2star, frequency_hz=0.0):
    """Evaluate the single-pool complex R2* decay model.

    ``S(TE) = S0 * exp((-R2* + 1j*2*pi*f) * TE)``, with complex ``S0`` absorbing
    the initial phase term.
    """
    echo_times_seconds = np.asarray(echo_times_seconds, dtype=float)
    decay = -np.asarray(r2star) + 1j * 2.0 * np.pi * np.asarray(frequency_hz)
    return np.asarray(s0) * np.exp(decay * echo_times_seconds)


def _initial_complex_decay_params(signal, echo_times_seconds):
    """Derive initial ``[log(abs(S0)), R2*, frequency_hz, phase0]`` for one fit."""
    amplitude = np.maximum(np.abs(signal), np.finfo(float).tiny)

    if echo_times_seconds.size > 1:
        slope, intercept = np.polyfit(echo_times_seconds, np.log(amplitude), 1)
        r2star = max(0.0, -float(slope))
        phase = np.unwrap(np.angle(signal))
        phase_slope, phase_intercept = np.polyfit(echo_times_seconds, phase, 1)
        frequency_hz = float(phase_slope / (2.0 * np.pi))
        phase0 = float(phase_intercept)
    else:
        intercept = float(np.log(amplitude[0]))
        r2star = 0.0
        frequency_hz = 0.0
        phase0 = float(np.angle(signal[0]))

    return np.array([float(intercept), r2star, frequency_hz, phase0], dtype=float)


def _fit_complex_decay_1d(
    signal,
    echo_times_seconds,
    *,
    lower_bounds,
    upper_bounds,
    max_nfev,
):
    """Fit one complex echo train with nonlinear least squares.

    Returns a dict with ``s0``, ``r2star``, ``frequency_hz``, ``phase0``,
    ``cost``, ``nfev``, and ``success``, or None when too few valid echoes are
    available or optimization fails.
    """
    valid = np.isfinite(signal.real) & np.isfinite(signal.imag)
    if int(valid.sum()) < 2:
        return None

    y_valid = signal[valid]
    te_valid = echo_times_seconds[valid]
    x0 = _initial_complex_decay_params(y_valid, te_valid)
    x0 = np.minimum(np.maximum(x0, lower_bounds), upper_bounds)

    def residuals(params):
        log_s0_abs, r2, freq, phi0 = params
        pred = _complex_decay_model(
            te_valid,
            np.exp(log_s0_abs) * np.exp(1j * phi0),
            r2,
            freq,
        )
        residual = pred - y_valid
        return np.concatenate([residual.real, residual.imag])

    try:
        result = scipy.optimize.least_squares(
            residuals,
            x0,
            bounds=(lower_bounds, upper_bounds),
            max_nfev=max_nfev,
        )
    except (ValueError, RuntimeError, FloatingPointError):
        return None

    log_s0_abs, r2star, frequency_hz, phase0 = result.x
    return {
        's0': np.exp(log_s0_abs) * np.exp(1j * phase0),
        'r2star': r2star,
        'frequency_hz': frequency_hz,
        'phase0': phase0,
        'cost': result.cost,
        'nfev': result.nfev,
        'success': result.success,
    }


def _chunk_indices(indices, n_threads):
    """Yield small index chunks for parallel voxel-wise fitting."""
    if indices.size == 0:
        return
    target_chunks = max(int(n_threads) * 2048, 1)
    chunk_size = min(64, max(1, int(np.ceil(indices.size / target_chunks))))
    for start in range(0, indices.size, chunk_size):
        yield indices[start : start + chunk_size]


def _fit_complex_decay_chunk(
    chunk_indices,
    chunk_signals,
    echo_times_seconds,
    lower_bounds,
    upper_bounds,
    max_nfev,
):
    """Fit a chunk of complex echo trains; picklable worker for joblib.

    Receives only the signals for ``chunk_indices`` (not the full volume) so
    each task ships a small array. Returns ``(idx, result)`` pairs, mirroring
    the original per-voxel ``fit_one`` output.
    """
    results = []
    for idx, signal in zip(chunk_indices, chunk_signals):
        result = _fit_complex_decay_1d(
            signal,
            echo_times_seconds,
            lower_bounds=lower_bounds,
            upper_bounds=upper_bounds,
            max_nfev=max_nfev,
        )
        results.append((idx, result))
    return results


def fit_complex_r2star(
    magnitude: np.ndarray,
    phase: np.ndarray,
    echo_times_seconds,
    mask: np.ndarray | None = None,
    n_threads: int = 1,
    max_r2star: float = np.inf,
    max_frequency_hz: float = np.inf,
    max_nfev: int | None = None,
) -> dict[str, np.ndarray]:
    """Estimate R2*/T2*/S0 maps with a complex single-pool NLLS fit.

    Parameters
    ----------
    magnitude, phase : numpy.ndarray
        Echo-wise magnitude and (radian) phase with the echo on the last axis,
        e.g. shape ``(X, Y, Z, E)``. Phase must be in radians.
    echo_times_seconds : sequence of float
        Echo times in seconds, one per echo.
    mask : numpy.ndarray of bool, optional
        Boolean mask over the non-echo dimensions. Voxels outside the mask are
        returned as NaN. If omitted, every voxel is fit.
    n_threads : int, optional
        Worker processes (joblib loky backend) for voxel-wise fitting. Use
        <= 0 for all CPUs. A value of 1 runs serially in-process.
    max_r2star, max_frequency_hz, max_nfev :
        Nonlinear least-squares fit bounds/options.

    Returns
    -------
    maps : dict of numpy.ndarray
        ``r2star`` (s^-1 / Hz), ``t2star`` (s), ``s0`` (magnitude),
        ``frequency_hz``, ``phase0``, ``success``, ``cost``, and ``nfev`` maps,
        each with the non-echo shape.
    """
    magnitude = np.asarray(magnitude)
    phase = np.asarray(phase)
    if magnitude.shape != phase.shape:
        raise ValueError(
            f'magnitude shape {magnitude.shape} does not match phase shape {phase.shape}.'
        )
    echo_times_seconds = np.asarray(echo_times_seconds, dtype=float)
    if echo_times_seconds.ndim != 1:
        raise ValueError('echo_times_seconds must be 1D.')
    if magnitude.shape[-1] != echo_times_seconds.size:
        raise ValueError(
            f'data has {magnitude.shape[-1]} echoes but '
            f'{echo_times_seconds.size} echo times were provided.'
        )
    if max_r2star <= 0:
        raise ValueError('max_r2star must be positive.')
    if max_frequency_hz <= 0:
        raise ValueError('max_frequency_hz must be positive.')

    signal = magnitude * np.exp(1j * phase)
    leading_shape = signal.shape[:-1]
    n_echoes = echo_times_seconds.size

    if mask is None:
        fit_mask = np.ones(leading_shape, dtype=bool)
    else:
        fit_mask = np.asarray(mask, dtype=bool)
        if fit_mask.shape != leading_shape:
            raise ValueError(f'mask has shape {fit_mask.shape}; expected {leading_shape}.')

    if n_threads is None or n_threads <= 0:
        n_threads = os.cpu_count() or 1
    n_threads = max(1, int(n_threads))

    flat_signal = signal.reshape((-1, n_echoes))
    flat_mask = fit_mask.reshape(-1)
    n_voxels = flat_signal.shape[0]

    s0 = np.full(n_voxels, np.nan + 1j * np.nan, dtype=np.complex128)
    r2star = np.full(n_voxels, np.nan, dtype=float)
    frequency_hz = np.full(n_voxels, np.nan, dtype=float)
    phase0 = np.full(n_voxels, np.nan, dtype=float)
    cost = np.full(n_voxels, np.nan, dtype=float)
    nfev = np.zeros(n_voxels, dtype=np.int32)
    success = np.zeros(n_voxels, dtype=bool)

    lower_bounds = np.array([-np.inf, 0.0, -max_frequency_hz, -np.inf], dtype=float)
    upper_bounds = np.array([np.inf, max_r2star, max_frequency_hz, np.inf], dtype=float)

    def fit_one(idx):
        return idx, _fit_complex_decay_1d(
            flat_signal[idx],
            echo_times_seconds,
            lower_bounds=lower_bounds,
            upper_bounds=upper_bounds,
            max_nfev=max_nfev,
        )

    def store_result(idx, result):
        if result is None:
            return
        s0[idx] = result['s0']
        r2star[idx] = result['r2star']
        frequency_hz[idx] = result['frequency_hz']
        phase0[idx] = result['phase0']
        cost[idx] = result['cost']
        nfev[idx] = result['nfev']
        success[idx] = result['success']

    voxel_indices = np.flatnonzero(flat_mask)
    if n_threads == 1:
        iterator = (fit_one(idx) for idx in voxel_indices)
        for idx, result in tqdm(iterator, total=voxel_indices.size, desc='complex NLLS samples'):
            store_result(idx, result)
    else:
        # The per-voxel scipy fit is GIL-bound Python, so threads give no real
        # speedup. Use joblib's loky (process) backend for true parallelism and
        # pin BLAS threads to 1 in the workers to avoid oversubscription.
        from joblib import Parallel, delayed, parallel_config

        chunks = list(_chunk_indices(voxel_indices, n_threads))
        tasks = (
            delayed(_fit_complex_decay_chunk)(
                chunk,
                flat_signal[chunk],
                echo_times_seconds,
                lower_bounds,
                upper_bounds,
                max_nfev,
            )
            for chunk in chunks
        )
        with (
            parallel_config(backend='loky', inner_max_num_threads=1),
            tqdm(total=voxel_indices.size, desc='complex NLLS samples') as pbar,
        ):
            for chunk_results in Parallel(n_jobs=n_threads, return_as='generator')(tasks):
                for idx, result in chunk_results:
                    store_result(idx, result)
                pbar.update(len(chunk_results))

    with np.errstate(divide='ignore', invalid='ignore'):
        t2star = 1.0 / r2star
    t2star[~np.isfinite(t2star)] = np.nan

    # Casting to float32 can overflow for very large T2* (R2* ~ 0); ignore the
    # overflow (those voxels become inf and are treated as unfit downstream).
    with np.errstate(over='ignore'):
        return {
            'r2star': r2star.reshape(leading_shape).astype(np.float32),
            't2star': t2star.reshape(leading_shape).astype(np.float32),
            's0': np.abs(s0).reshape(leading_shape).astype(np.float32),
            'frequency_hz': frequency_hz.reshape(leading_shape).astype(np.float32),
            'phase0': phase0.reshape(leading_shape).astype(np.float32),
            'success': success.reshape(leading_shape).astype(np.float32),
            'cost': cost.reshape(leading_shape).astype(np.float32),
            'nfev': nfev.reshape(leading_shape).astype(np.float32),
        }
