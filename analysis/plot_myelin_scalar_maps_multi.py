"""Plot selected scalar maps as multiple rows in one figure (group mean)."""

import json
import math
import os
import sys
import warnings
from glob import glob

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import templateflow.api as tflow
from nilearn import image, maskers, plotting

# Keys in patterns.json / name_mapper.json; order is row order (top to bottom).
MULTI_KEYS = [
    'NODDI ICVF',
    'MPRAGE-MyelinW',
    'R1-B1c',
    'ihMTsat-B1c',
    "QSM-SEPIA-E5",
]

CUT_COORDS = [-30, -15, 0, 15, 30, 45, 60]


def _pattern_for_key(filename_mapper, key):
    for patterns in filename_mapper.values():
        if key in patterns:
            return patterns[key]
    raise KeyError(f'No pattern group contains key {key!r}')


def _strip_trailing_zeroes(num_str):
    """Drop unnecessary trailing zeros and a trailing decimal point (e.g. '1.200' -> '1.2')."""
    if 'e' in num_str.lower():
        return num_str
    if '.' not in num_str:
        return num_str
    return num_str.rstrip('0').rstrip('.')


def _colorbar_axis(cbar):
    """Tick *values* for a vertical colorbar are on y; horizontal uses x."""
    orient = getattr(cbar, 'orientation', 'vertical')
    if orient == 'vertical':
        return cbar.ax.yaxis, 'y'
    return cbar.ax.xaxis, 'x'


def _colorbar_ticks(cbar, vmin, vmax0, percentile):
    axis, tick_axis = _colorbar_axis(cbar)
    if percentile:
        if vmin == 0:
            cbar.set_ticks([0, vmax0])
            cbar.set_ticklabels(['0', '98th Percentile'])
        else:
            cbar.set_ticks([vmin, 0, vmax0])
            cbar.set_ticklabels(['-98th Percentile', '0', '98th Percentile'])
    else:
        if vmin == 0:
            cbar.set_ticks([0, np.mean([0, vmax0]), vmax0])
        else:
            cbar.set_ticks([vmin, 0, vmax0])

        # Match tick scale to the largest |tick| so mantissas stay in a readable range.
        tick_max = max(abs(vmin), abs(vmax0))
        if tick_max == 0:
            exponent = 0
        else:
            exponent = int(math.floor(math.log10(tick_max)))
        scale = 10**exponent

        def _fmt(val, pos, s=scale):
            if s == 0:
                return ''
            if abs(val) < 1e-15:
                return '0'
            t = _strip_trailing_zeroes(f'{val / s:.3f}')
            if t in ('', '-'):
                return '0'
            return t

        axis.set_major_formatter(mpl.ticker.FuncFormatter(_fmt))
        if exponent != 0:
            cbar.ax.text(
                0.5,
                1.02,
                f'$\\times 10^{{{exponent}}}$',
                transform=cbar.ax.transAxes,
                va='bottom',
                ha='center',
                fontsize=10,
            )

    cbar.ax.tick_params(axis=tick_axis, labelsize=10, length=0)


if __name__ == '__main__':
    _script_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, os.path.join(_script_dir, '..'))
    from configuration.config import load_config

    _cfg = load_config()

    PERCENTILE = False

    in_dir = os.path.join(_cfg['project_root'], 'scalars')
    out_dir = os.path.abspath(os.path.join(_script_dir, '..', 'figures', 'scalars'))
    template = tflow.get(
        'MNI152NLin2009cAsym', resolution='01', desc='brain', suffix='T1w', extension='nii.gz'
    )
    mask = tflow.get(
        'MNI152NLin2009cAsym', resolution='01', desc='brain', suffix='mask', extension='nii.gz'
    )

    os.makedirs(out_dir, exist_ok=True)

    with open('name_mapper.json', 'r') as fo:
        name_mapper = json.load(fo)

    with open('patterns.json', 'r') as fo:
        filename_mapper = json.load(fo)

    rows = []
    for key in MULTI_KEYS:
        title = name_mapper[key]
        pattern = _pattern_for_key(filename_mapper, key)
        temp_pattern = pattern.format(subject='*', session='*')

        scalar_maps = sorted(glob(os.path.join(in_dir, temp_pattern)))
        scalar_maps = [f for f in scalar_maps if 'PILOT' not in f]
        if len(scalar_maps) > 44:
            raise Exception(temp_pattern)

        if not scalar_maps:
            raise FileNotFoundError(f'No scalar maps found for {key!r} ({title})')

        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            mask_img = image.resample_to_img(mask, scalar_maps[0], interpolation='nearest')

        masker = maskers.NiftiMasker(mask_img, resampling_target='data')
        mean_img = image.mean_img(scalar_maps, copy_header=True)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            mean_arr = masker.fit_transform(mean_img)

        if 'Chi Map' in title:
            # Invert chi maps
            mean_arr *= -1

        mean_arr[np.isnan(mean_arr)] = 0
        mean_arr[np.isinf(mean_arr)] = 0
        mean_img = masker.inverse_transform(mean_arr)
        vmax0 = np.percentile(np.abs(mean_arr), 98)
        if 'Chi Map' in title:
            kwargs = {'symmetric_cbar': True, 'vmin': None}
            vmin = -vmax0
        else:
            kwargs = {'symmetric_cbar': False, 'vmin': 0}
            vmin = 0

        rows.append((mean_img, title, vmax0, vmin, kwargs))

    n = len(rows)
    fig, axs = plt.subplots(
        n,
        2,
        figsize=(17.5, 4.2 * n),
        width_ratios=[1, 0.035],
        gridspec_kw={'wspace': 0.04, 'hspace': 0.35},
    )
    cmap = mpl.cm.viridis

    for i, (mean_img, title, vmax0, vmin, kwargs) in enumerate(rows):
        if 'Chi Map' in title:
            title += " (Inverted)"

        ax_map = axs[i, 0]
        cax = axs[i, 1]
        plotting.plot_stat_map(
            mean_img,
            bg_img=template,
            display_mode='z',
            cut_coords=CUT_COORDS,
            axes=ax_map,
            figure=fig,
            vmax=vmax0,
            cmap='viridis',
            annotate=False,
            black_bg=False,
            resampling_interpolation='nearest',
            colorbar=False,
            **kwargs,
        )
        ax_map.set_title(title, fontsize=18)
        norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax0)
        cbar = fig.colorbar(
            mpl.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax, orientation='vertical'
        )
        _colorbar_ticks(cbar, vmin, vmax0, PERCENTILE)

    fname = 'multi_panel_scalars'
    fig.savefig(os.path.join(out_dir, f'{fname}.png'), bbox_inches='tight')
    fig.savefig(os.path.join(out_dir, f'{fname}.pdf'), bbox_inches='tight')
    plt.close()
