"""Plot scalar maps from myelin measures."""

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


if __name__ == '__main__':
    _script_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, os.path.join(_script_dir, '..'))
    from configuration.config import load_config

    _cfg = load_config()

    in_dir = os.path.join(_cfg['project_root'], 'derivatives')
    out_dir = os.path.join(_script_dir, '..', 'figures', 'scalars')
    PERCENTILE = False

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

    for group, patterns in filename_mapper.items():
        for key, pattern in patterns.items():
            title = name_mapper[key]
            temp_pattern = pattern.format(subject='*', session='*')

            # Get all scalar maps
            scalar_maps = sorted(glob(os.path.join(in_dir, temp_pattern)))
            scalar_maps = [f for f in scalar_maps if 'PILOT' not in f]
            print(f'{title}: {len(scalar_maps)}')

            # Mask out non-brain voxels
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                mask_img = image.resample_to_img(mask, scalar_maps[0], interpolation='nearest')

            masker = maskers.NiftiMasker(mask_img, resampling_target='data')
            mean_img = image.mean_img(scalar_maps, copy_header=True)
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                mean_arr = masker.fit_transform(mean_img)

            # Get vmax (98th percentile) across both sessions
            mean_arr[np.isnan(mean_arr)] = 0
            mean_arr[np.isinf(mean_arr)] = 0
            mean_img = masker.inverse_transform(mean_arr)
            vmax0 = np.percentile(mean_arr, 98)
            print(f'\t{vmax0}')
            if 'Chi Map' in title:
                # Use two-directional colorbar
                kwargs = {'symmetric_cbar': True, 'vmin': None}
                vmin = -vmax0
            else:
                kwargs = {'symmetric_cbar': False, 'vmin': 0}
                vmin = 0

            # Plot mean from each session
            fig, axs = plt.subplots(2, 1, figsize=(17.5, 5), height_ratios=[2, 0.25])
            # Increase vertical space between the first two rows
            fig.subplots_adjust(hspace=-0.2)
            plotting.plot_stat_map(
                mean_img,
                bg_img=template,
                display_mode='z',
                cut_coords=[-30, -15, 0, 15, 30, 45, 60],
                axes=axs[0],
                figure=fig,
                vmax=vmax0,
                cmap='viridis',
                annotate=False,
                black_bg=False,
                resampling_interpolation='nearest',
                colorbar=False,
                **kwargs,
            )
            fig.suptitle(title, fontsize=20, y=0.9)

            # Plot the colorbars
            # Resize colorbar axis to be shorter and narrower
            cax = axs[1]
            pos = cax.get_position()
            new_width = pos.width * 0.75
            new_height = 0.05
            center_x = pos.x0 + pos.width / 2.0
            new_x0 = center_x - new_width / 2.0
            new_y0 = pos.y0 + (pos.height - new_height) / 2.0
            cax.set_position([new_x0, new_y0, new_width, new_height])

            cmap = mpl.cm.viridis

            norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax0)
            cbar = fig.colorbar(
                mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
                cax=cax,
                orientation='horizontal',
            )
            if PERCENTILE:
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

                exponent = int(math.floor(math.log10(abs(vmax0)))) if vmax0 != 0 else 0
                scale = 10**exponent
                cbar.ax.xaxis.set_major_formatter(
                    mpl.ticker.FuncFormatter(
                        lambda val, _, s=scale: '0' if val == 0 else f'{val / s:.3f}'
                    )
                )
                if exponent != 0:
                    cbar.ax.text(
                        1.01,
                        0.5,
                        f'$\\times10^{{{exponent}}}$',
                        transform=cbar.ax.transAxes,
                        va='center',
                        ha='left',
                        fontsize=14,
                    )

            cbar.ax.tick_params(labelsize=14, length=0)

            fname = title.lower().replace('/', '_').replace(' ', '_')
            fname = fname.replace('*', 'star').replace("'", 'prime')
            fname = fname.replace('(', '').replace(')', '')
            fig.savefig(os.path.join(out_dir, f'{fname}.png'), bbox_inches='tight')
            plt.close()
