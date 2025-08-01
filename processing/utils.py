import os


def run_command(command, env=None):
    """Run a given shell command with certain environment variables set.

    Copied from XCP-D.
    """
    import subprocess

    merged_env = os.environ
    if env:
        merged_env.update(env)

    process = subprocess.Popen(
        command.split(),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        shell=False,
        env=merged_env,
    )
    while True:
        line = process.stdout.readline()
        line = str(line, 'utf-8')[:-1]
        print(line)
        if line == '' and process.poll() is not None:
            break

    if process.returncode != 0:
        raise RuntimeError(
            f'Non zero return code: {process.returncode}\n{command}\n\n{process.stdout.read()}'
        )


def to_bidsuri(filename, dataset_dir, dataset_name):
    return f'bids:{dataset_name}:{os.path.relpath(filename, dataset_dir)}'


def get_filename(name_source, layout, out_dir, entities, dismiss_entities=None):
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


def coregister_to_t1(name_source, layout, in_file, t1_file, out_dir, source_space, target_space):
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
        dismiss_entities=['acquisition', 'inv', 'reconstruction','mt', 'echo', 'part'],
    )
    shutil.copyfile(transform, transform_file)

    return transform_file


def plot_coregistration(
    name_source,
    layout,
    in_file,
    t1_file,
    out_dir,
    source_space,
    target_space,
    wm_seg=None,
):
    """Plot the coregistration of an image to a T1w image."""
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


def fit_monoexponential(in_files, echo_times):
    """Fit monoexponential decay model to MESE data.

    Parameters
    ----------
    in_files : list of str
        List of paths to MESE data.
    echo_times : list of float
        List of echo times in seconds.

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
    t2s_limited, s0_limited, _, _ = decay.fit_loglinear(
        data_cat=data_cat,
        echo_times=echo_times_ms,
        adaptive_mask=masksum,
        report=False,
    )
    # Limit infinite values to 0.5 seconds (500 ms)
    t2s_limited[np.isinf(t2s_limited)] = 500
    s0_limited[np.isinf(s0_limited)] = 0

    r_squared = calculate_r_squared(np.squeeze(data_cat), echo_times_ms, s0_limited, t2s_limited)

    t2s_s = t2s_limited / 1000

    r2s_hz = 1 / t2s_s

    t2s_s_img = io.new_nii_like(ref_img, t2s_s)
    r2s_hz_img = io.new_nii_like(ref_img, r2s_hz)
    s0_img = io.new_nii_like(ref_img, s0_limited)
    r_squared_img = io.new_nii_like(ref_img, r_squared)
    return t2s_s_img, r2s_hz_img, s0_img, r_squared_img


def plot_scalar_map(underlay, overlay, mask, out_file, dseg=None, vmin=None, vmax=None, cmap='Reds'):
    import matplotlib.pyplot as plt
    import nibabel as nb
    import pandas as pd
    import seaborn as sns
    from matplotlib import cm
    from nilearn import image, masking, plotting
    from nireports.reportlets.utils import cuts_from_bbox

    if not os.path.isdir(os.path.dirname(out_file)):
        os.makedirs(os.path.dirname(out_file), exist_ok=True)

    cuts = cuts_from_bbox(nb.load(underlay), cuts=6)
    z_cuts = cuts['z']
    overlay_masked = masking.unmask(masking.apply_mask(overlay, mask), mask)
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
            f'(img == {tissue_type_val}).astype(int)',
            img=dseg,
        )
        tissue_type_vals = masking.apply_mask(overlay, mask_img)
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

    xticks = [i for i in xticks if i.get_position()[0] <= xlim[1] and i.get_position()[0] >= xlim[0]]
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


def calculate_r_squared(data, echo_times, s0, t2s):
    """Calculate R-squared from data and T2*/S0 estimates.

    R-squared is a measure of goodness of fit of a monoexponential model to the data.

    Parameters
    ----------
    data : numpy.ndarray of shape (n_samples, n_echos)
        Data to calculate R-squared for.
    echo_times : list of float
        Echo times in seconds.
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
    ss_resid = np.sum(residuals ** 2, axis=1)

    # Calculate mean signal per voxel
    mean_signal = np.mean(data, axis=1)

    # Total sum of squares per voxel (sum of squared deviations from mean)
    ss_total = np.sum((data - mean_signal[:, np.newaxis]) ** 2, axis=1)

    # Handle division by zero
    ss_total[ss_total == 0] = np.spacing(1)

    # R-squared = 1 - (SS_residual / SS_total)
    r_squared = 1 - (ss_resid / ss_total)

    return r_squared
